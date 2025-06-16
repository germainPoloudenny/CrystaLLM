import argparse
import gzip
import pickle
import tarfile
import io

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CIFs from AMP+structure dataset and write to .tar.gz")
    parser.add_argument("name", help="Path to mp20_amp_*.pkl.gz file")
    parser.add_argument("--out", "-o", required=True, help="Path to output .tar.gz file containing .cif files")
    args = parser.parse_args()

    with gzip.open(args.name, "rb") as f:
        entries = pickle.load(f)

    with tarfile.open(args.out, "w:gz") as tar:
        count = 0
        for cif_id, text in entries:
            if "\n" in text:
                _, cif_str = text.split("\n", 1)
            else:
                cif_str = text

            cif_bytes = cif_str.encode("utf-8")
            info = tarfile.TarInfo(name=f"{cif_id}.cif")
            info.size = len(cif_bytes)

            tar.addfile(tarinfo=info, fileobj=io.BytesIO(cif_bytes))
            count += 1

    print(f"wrote {count} CIFs to {args.out}")
