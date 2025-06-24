import os
import gzip
import pickle
import argparse

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else lambda x: x

try:
    from crystallm import CIFTokenizer, sequences_from_lmdb
except Exception:
    CIFTokenizer = None
    sequences_from_lmdb = None

def load_pickle(path):
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rb') as f:
        return pickle.load(f)

def encode_sequences(cif_list, tokenizer, stoi, itos, keep_unknown=False):
    encoded = []
    for item in tqdm(cif_list, desc="encoding sequences"):
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[1], (str, bytes))
        ):
            cif = item[1]
        else:
            cif = item

        if isinstance(cif, bytes):
            cif = cif.decode("utf-8", errors="ignore")

        tokens = None

        if not isinstance(cif, str):
            if np is not None:
                arr = np.asarray(cif).reshape(-1)
                is_int_seq = np.issubdtype(arr.dtype, np.integer)
                values = arr.tolist()
            else:
                try:
                    values = list(cif)
                    is_int_seq = all(isinstance(v, int) for v in values)
                except TypeError:
                    is_int_seq = False
                    values = []

            if is_int_seq:
                tokens = [f"<amp_{int(v)}>" for v in values]

        if tokens is None:
            tokens = tokenizer.tokenize_cif(str(cif), keep_unknown=keep_unknown)

        for tok in tokens:
            if tok not in stoi:
                idx = len(stoi)
                stoi[tok] = idx
                itos[idx] = tok
            encoded.append(stoi[tok])

    if np is not None:
        return np.array(encoded, dtype=np.uint16)
    return encoded

def copy_embedding_lmdb_with_token_names(
    input_path,
    output_path,
    prefix="<amp_",
    *,
    include_db_index=False,
    dim=500,
):
    """Copy embeddings from an LMDB and rename keys using token strings.

    Parameters
    ----------
    input_path : str
        Path to the source LMDB containing the embeddings.
    output_path : str
        Destination LMDB where renamed embeddings will be written.
    prefix : str, optional
        Prefix to prepend to each token id. Defaults to ``"<amp_"``.
    include_db_index : bool, optional
        When ``True`` the sub-database index is appended to each key. The
        default ``False`` produces keys of the form ``<amp_0>`` which match the
        tokens produced by :func:`encode_sequences`.
    dim : int, optional
        Expected embedding dimensionality (unused but kept for backwards
        compatibility).
    """
    import lmdb
    import pickle

    env_in = lmdb.open(input_path, readonly=True, lock=False, max_dbs=10)
    env_out = lmdb.open(output_path, map_size=int(1e12))

    # Lire le nombre de sous-bases
    with env_in.begin() as txn:
        num_dbs = int(txn.get(b"num_dbs").decode())

    # Ouvrir toutes les sous-bases
    sub_dbs_in = [env_in.open_db(str(i).encode()) for i in range(num_dbs)]

    with env_out.begin(write=True) as txn_out:
        for db_idx, sub_db in enumerate(sub_dbs_in):
            with env_in.begin(db=sub_db) as txn_in:
                cursor = txn_in.cursor()
                for raw_key, raw_val in tqdm(cursor, desc=f"copying embeddings from sub_db {db_idx}"):
                    try:
                        vec = pickle.loads(raw_val)
                        key_str = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
                        if include_db_index:
                            new_key = f"{prefix}{key_str}_{db_idx}>"
                        else:
                            new_key = f"{prefix}{key_str}>"
                        txn_out.put(new_key.encode("utf-8"), pickle.dumps(vec))
                    except Exception as e:
                        print(f"Skipped key {raw_key} in sub_db {db_idx}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retokenize dataset with updated meta.pkl')
    parser.add_argument('--lmdb_path', default='', help='LMDB containing CIF strings')
    parser.add_argument('--sub_db', type=int, default=0, help='LMDB sub-database index')
    parser.add_argument('--train_fraction', type=float, default=0.9, help='Train split fraction when using LMDB')
    parser.add_argument('--meta_path', required=True, help='Path to meta.pkl from main dataset')
    parser.add_argument('--out_dir', required=True, help='Directory to write train.bin/val.bin')
    parser.add_argument('--keep_unknown', action='store_true', help='Keep unknown tokens instead of replacing them with <unk>')
    parser.add_argument('--output_embeddings', action='store_true', help='Copy existing LMDB embeddings and rename keys to match <amp_*> format')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']

    tokenizer = CIFTokenizer()

    if args.lmdb_path:
        sequences = sequences_from_lmdb(args.lmdb_path, sub_db=args.sub_db)
        if not sequences:
            raise ValueError('LMDB contains no sequences')

        rng = np.random.default_rng(0)
        rng.shuffle(sequences)
        split_idx = int(len(sequences) * args.train_fraction)
        train_pairs = sequences[:split_idx]
        val_pairs = sequences[split_idx:]
    else:
        parser.error('Currently only LMDB input is supported')

    train_ids = encode_sequences(train_pairs, tokenizer, stoi, itos, keep_unknown=args.keep_unknown)
    train_ids.tofile(os.path.join(args.out_dir, 'train.bin'))

    if val_pairs:
        val_ids = encode_sequences(val_pairs, tokenizer, stoi, itos, keep_unknown=args.keep_unknown)
        val_ids.tofile(os.path.join(args.out_dir, 'val.bin'))

    updated_meta = {
        'stoi': stoi,
        'itos': itos,
        'vocab_size': len(itos),
    }
    with open(os.path.join(args.out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(updated_meta, f)

    if args.output_embeddings:
        emb_path = os.path.join(args.out_dir, "amp_embeddings.lmdb")
        copy_embedding_lmdb_with_token_names("data/version_0_last.lmdb", emb_path)
        print(f"Copied embeddings to: {emb_path}")
