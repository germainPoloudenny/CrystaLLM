import argparse
import gzip
import pickle
import tarfile
import io
import os


def load_entries_from_tar(tar_fname):
    entries = []
    with tarfile.open(tar_fname, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".cif"):
                fid = os.path.basename(member.name)
                cif_id = fid.replace(".cif", "")
                f = tar.extractfile(member)
                if f is not None:
                    text = f.read().decode("utf-8")
                    entries.append((cif_id, text))
    return entries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CIFs from AMP+structure dataset and write to .tar.gz"
    )
    parser.add_argument(
        "name",
        help="Path to input file (.pkl.gz or .tar.gz) containing AMP-prefixed CIFs",
    )
    parser.add_argument("--out", "-o", required=True, help="Path to output .tar.gz file containing .cif files")
    args = parser.parse_args()

    if args.name.endswith(".pkl.gz"):
        entries = load_entries_from_pkl(args.name)
    elif args.name.endswith(".tar.gz"):
        entries = load_entries_from_tar(args.name)
    else:
        raise ValueError("input file must end with .pkl.gz or .tar.gz")

    with tarfile.open(args.out, "w:gz") as tar:
        count = 0
        for cif_id, text in entries:
            lines = text.splitlines()
            start = 0
            for i, line in enumerate(lines):
                if line.lstrip().startswith("data_"):
                    start = i
                    break
            cif_str = "\n".join(lines[start:])

            cif_bytes = cif_str.encode("utf-8")
            info = tarfile.TarInfo(name=f"{cif_id}.cif")
            info.size = len(cif_bytes)

            tar.addfile(tarinfo=info, fileobj=io.BytesIO(cif_bytes))
            count += 1

    print(f"wrote {count} CIFs to {args.out}")


if __name__ == "__main__":
    main()
