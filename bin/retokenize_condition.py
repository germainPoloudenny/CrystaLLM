import os
import gzip
import pickle
import argparse
try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    np = None
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else lambda x: x

try:
    from crystallm import CIFTokenizer, sequences_from_lmdb
except Exception:  # pragma: no cover - optional deps for unit tests
    CIFTokenizer = None
    sequences_from_lmdb = None


def load_pickle(path):
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rb') as f:
        return pickle.load(f)


def encode_sequences(cif_list, tokenizer, stoi, itos, keep_unknown=False):
    """Tokenize a list of CIF strings or integer sequences.

    Parameters
    ----------
    cif_list : list
        Iterable containing either CIF strings/tuples or sequences of integers.
    tokenizer : CIFTokenizer
        Tokenizer used for CIF strings.
    stoi : dict
        Existing string-to-index mapping (will be updated in-place).
    itos : dict
        Existing index-to-string mapping (will be updated in-place).
    keep_unknown : bool, optional
        Whether to keep unknown tokens instead of replacing them with ``<unk>``.
    """

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

        # Bytes may represent a CIF string
        if isinstance(cif, bytes):
            cif = cif.decode("utf-8", errors="ignore")

        tokens = None

        # Sequence of integers: map each unique integer to <amp_i>
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

        # Fallback: treat as CIF string
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retokenize dataset with updated meta.pkl')
    parser.add_argument('--lmdb_path', default='', help='LMDB containing CIF strings')
    parser.add_argument('--sub_db', type=int, default=0, help='LMDB sub-database index')
    parser.add_argument('--train_fraction', type=float, default=0.9, help='Train split fraction when using LMDB')
    parser.add_argument('--meta_path', required=True, help='Path to meta.pkl from main dataset')
    parser.add_argument('--out_dir', required=True, help='Directory to write train.bin/val.bin')
    parser.add_argument('--keep_unknown', action='store_true', help='Keep unknown tokens instead of replacing them with <unk>')
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
        if not args.train_fname:
            parser.error('Either --lmdb_path or --train_fname must be provided')
        train_pairs = load_pickle(args.train_fname)
        val_pairs = load_pickle(args.val_fname) if args.val_fname else []

    train_ids = encode_sequences(train_pairs, tokenizer, stoi, itos, keep_unknown=args.keep_unknown)
    train_ids.tofile(os.path.join(args.out_dir, 'train.bin'))

    if val_pairs:
        val_ids = encode_sequences(val_pairs, tokenizer, stoi, itos, keep_unknown=args.keep_unknown)
        val_ids.tofile(os.path.join(args.out_dir, 'val.bin'))

    # ✨ Update and save expanded vocabulary
    updated_meta = {
        'stoi': stoi,
        'itos': itos,
        'vocab_size': len(itos),
    }
    with open(os.path.join(args.out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(updated_meta, f)