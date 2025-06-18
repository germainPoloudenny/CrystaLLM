import os
import io
import tarfile
import argparse

import torch
from pymatgen.core import Structure
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


class MP20:
    def __init__(self, root: str, cache: bool = True):
        self.root = root
        self.raw_csv = os.path.join(root, "raw/all.csv")
        self.cache_path = os.path.join(root, "raw/all.pt")

        print(f"Chargement du dataset MP20 depuis {self.cache_path}...")
        if cache and os.path.exists(self.cache_path):
            self._data = torch.load(self.cache_path, weights_only=False)
            print(f"\u2192 {len(self._data)} structures charg\u00e9es.")
        else:
            raise FileNotFoundError(f"Fichier non trouv\u00e9: {self.cache_path}")

    def __getitem__(self, idx: int) -> Structure:
        return Structure.from_str(self._data[idx]["cif"], fmt="cif")

    def __len__(self) -> int:
        return len(self._data)


def write_dataset_to_tar(dataset: MP20, out_fname: str) -> None:
    with tarfile.open(out_fname, "w:gz") as tar:
        for i in tqdm(range(len(dataset)), desc="writing CIFs..."):
            cif_str = dataset[i].to(fmt="cif")
            cif_bytes = cif_str.encode("utf-8")
            info = tarfile.TarInfo(name=f"{i}.cif")
            info.size = len(cif_bytes)
            tar.addfile(info, io.BytesIO(cif_bytes))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert MP-20 dataset to tar.gz format.")
    parser.add_argument("--mp20_root", default="/home/gpoloudenny/Projects/all-atom-diffusion-transformer/data/mp_20")
    parser.add_argument("--out", "-o", required=True, help="Path to output tar.gz file.")
    args = parser.parse_args(argv)

    dataset = MP20(args.mp20_root)
    write_dataset_to_tar(dataset, args.out)
    print(f"\u2705 CIFs written to {args.out}")


if __name__ == "__main__":
    main()