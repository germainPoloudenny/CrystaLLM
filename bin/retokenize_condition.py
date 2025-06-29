import os
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

def encode_sequences(cif_list, tokenizer, stoi, itos):
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
            # keep_unknown=True ensures new tokens are added to the vocabulary
            tokens = tokenizer.tokenize_cif(str(cif), keep_unknown=True)

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
    parser.add_argument('--lmdb_path', required=True, help='LMDB containing CIF strings')
    parser.add_argument('--sub_db', type=int, default=0, help='LMDB sub-database index')
    parser.add_argument('--train_fraction', type=float, default=0.9, help='Train split fraction when using LMDB')
    parser.add_argument('--meta_path', required=True, help='Path to meta.pkl from main dataset')
    parser.add_argument('--out_dir', required=True, help='Directory to write train.bin/val.bin')
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

        #rng = np.random.default_rng(0)
        #rng.shuffle(sequences)
        split_idx = int(len(sequences) * args.train_fraction)
        train_pairs = sequences[:split_idx]
        val_pairs = sequences[split_idx:]
    else:
        parser.error('Currently only LMDB input is supported')

    num_sequences = len(sequences)
    print(num_sequences)
    sequence_lengths = [len(np.asarray(seq).reshape(-1)) for seq in sequences]
    condition_length = max(sequence_lengths) if sequence_lengths else 0
    print(condition_length)

    train_ids = encode_sequences(train_pairs, tokenizer, stoi, itos)
    train_ids.tofile(os.path.join(args.out_dir, 'train.bin'))

    if val_pairs:
        val_ids = encode_sequences(val_pairs, tokenizer, stoi, itos)
        val_ids.tofile(os.path.join(args.out_dir, 'val.bin'))

    updated_meta = {
        'stoi': stoi,
        'itos': itos,
        'vocab_size': len(itos),
        'num_sequences': num_sequences,
        'condition_length': condition_length,
    }
    with open(os.path.join(args.out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(updated_meta, f)



