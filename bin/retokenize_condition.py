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
    parser = argparse.ArgumentParser(
        description=(
            'Retokenize a conditioning dataset using the vocabulary from a ' 
            'main dataset and split it into train, validation and test sets.'
        )
    )
    parser.add_argument('--lmdb_path', required=True, help='LMDB containing CIF strings')
    parser.add_argument('--sub_db', type=int, default=0, help='LMDB sub-database index')
    parser.add_argument(
        '--val_fraction',
        type=float,
        default=0.10,
        help=(
            'Fraction of the training portion to use for validation. ' 
            'This is applied after the test split.'
        ),
    )
    parser.add_argument(
        '--test_fraction',
        type=float,
        default=0.0045,
        help='Fraction of the entire dataset to reserve for testing',
    )
    parser.add_argument('--meta_path', required=True, help='Path to meta.pkl from main dataset')
    parser.add_argument('--out_dir', required=True, help='Directory to write train.bin/val.bin/test.bin')
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

        num_sequences = len(sequences)
        num_test = int(num_sequences * args.test_fraction)
        num_train_val = num_sequences - num_test
        num_val = int(num_train_val * args.val_fraction)
        num_train = num_train_val - num_val

        train_pairs = sequences[:num_train]
        val_pairs = sequences[num_train:num_train + num_val]
        test_pairs = sequences[num_train + num_val:]
    else:
        parser.error('Currently only LMDB input is supported')

    print(num_sequences)
    sequence_lengths = [len(np.asarray(seq).reshape(-1)) for seq in sequences]
    condition_length = max(sequence_lengths) if sequence_lengths else 0
    print(condition_length)

    train_ids = encode_sequences(train_pairs, tokenizer, stoi, itos)
    train_ids.tofile(os.path.join(args.out_dir, 'train.bin'))

    if val_pairs:
        val_ids = encode_sequences(val_pairs, tokenizer, stoi, itos)
        val_ids.tofile(os.path.join(args.out_dir, 'val.bin'))

    if test_pairs:
        test_ids = encode_sequences(test_pairs, tokenizer, stoi, itos)
        test_ids.tofile(os.path.join(args.out_dir, 'test.bin'))

    updated_meta = {
        'stoi': stoi,
        'itos': itos,
        'vocab_size': len(itos),
        'num_sequences': num_sequences,
        'condition_length': condition_length,
    }
    with open(os.path.join(args.out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(updated_meta, f)



