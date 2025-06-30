import argparse
from crystallm import embeddings_from_lmdb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display the dimension of the embeddings stored in an LMDB database.")
    parser.add_argument(
        "lmdb_path",
        help="Path to the LMDB database directory (e.g. data/version_0_last.lmdb)")
    parser.add_argument(
        "--sub_db",
        type=int,
        default=None,
        help="Index of a sub-database to read from if applicable")
    args = parser.parse_args()

    embedding_data = embeddings_from_lmdb(args.lmdb_path, sub_db=args.sub_db)
    if not embedding_data:
        raise RuntimeError("LMDB database is empty or not readable")

    first_vec = next(iter(embedding_data.values()))
    print(f"Embedding dimension: {len(first_vec)}")
    print(f"Number of embeddings: {len(embedding_data)}")


if __name__ == "__main__":
    main()