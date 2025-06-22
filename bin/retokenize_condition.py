import os
import gzip
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from crystallm import CIFTokenizer, sequences_from_lmdb


def load_pickle(path):
    open_fn = gzip.open if path.endswith('.gz') else open
    with open_fn(path, 'rb') as f:
        return pickle.load(f)


def encode_sequences(cif_list, tokenizer, stoi, itos, keep_unknown=False):
    encoded = []
    for item in tqdm(cif_list, desc='encoding sequences'):
        cif = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else item
        if isinstance(cif, bytes):
            cif = cif.decode('utf-8', errors='ignore')
        tokens = tokenizer.tokenize_cif(str(cif), keep_unknown=keep_unknown)

        ids = []
        for tok in tokens:
            if tok not in stoi:
                idx = len(stoi)
                stoi[tok] = idx
                itos[idx] = tok  # ✨ Add new token to vocab
            ids.append(stoi[tok])
        encoded.extend(ids)
    return np.array(encoded, dtype=np.uint16)


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