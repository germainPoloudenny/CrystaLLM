import argparse
import gzip
import pickle
import io
import tarfile
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Construct AMP prefix prompts from a dataset.")
    parser.add_argument("dataset", type=str, help="Path to the dataset .pkl.gz file containing (id, text) tuples with AMP prefixes.")
    parser.add_argument("--out", "-o", required=True, help="Output .tar.gz file to store the prompt .txt files.")
    args = parser.parse_args()

    ds_fname = args.dataset
    out_fname = args.out

    print(f"loading data from {ds_fname}...")
    with gzip.open(ds_fname, "rb") as f:
        data = pickle.load(f)

    with tarfile.open(out_fname, "w:gz") as tar:
        for id, text in tqdm(data, desc="preparing AMP prompts..."):
            prefix = text.split("\n", 1)[0].strip()
            info = tarfile.TarInfo(name=f"{id}.txt")
            content = prefix.encode("utf-8")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

    print(f"AMP prompts written to {out_fname}")


if __name__ == "__main__":
    main()