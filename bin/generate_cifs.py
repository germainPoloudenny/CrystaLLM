import os
import argparse
import io
import tarfile
import multiprocessing as mp
import sys

from tqdm import tqdm
from contextlib import nullcontext
import torch

from crystallm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
    array_split,
)


def write_listener(queue, n, out_file):
    pbar = tqdm(total=n, desc="generating CIFs...")
    with tarfile.open(out_file, "w:gz") as tar:
        while True:
            message = queue.get()
            if message == "kill":
                break
            cif_id, cif_str = message
            cif_info = tarfile.TarInfo(name=f"{cif_id}.cif")
            cif_bytes = cif_str.encode("utf-8")
            cif_info.size = len(cif_bytes)
            tar.addfile(cif_info, io.BytesIO(cif_bytes))
            pbar.update(1)


def get_prompts_from_file(prompts_file):
    prompts = []
    with tarfile.open(prompts_file, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting prompts..."):
            f = tar.extractfile(member)
            if f is not None:
                content = f.read().decode("utf-8")
                filename = os.path.basename(member.name)
                cif_id = filename.replace(".txt", "")
                prompts.append((cif_id, content))
    return prompts


def generate(model_dir, seed, device, dtype, num_gens, temperature, top_k, max_new_tokens, num_amp_tokens, chunk_of_prompts, queue):
    # init torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init tokenizer
    tokenizer = CIFTokenizer(num_amp_tokens=num_amp_tokens)
    encode = tokenizer.encode
    decode = tokenizer.decode

    print(f"initializing model from {model_dir} on {device}...")
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    model = torch.compile(model)  # requires PyTorch 2.0

    with torch.no_grad():
        with ctx:
            for cif_id, prompt in chunk_of_prompts:
                start_ids = encode(tokenizer.tokenize_cif(prompt))
                x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                output = decode(y[0].tolist())
                queue.put((cif_id, output))


"""
This script takes a collection of prompts and generates a corresponding CIF file for each prompt.

If there are multiple GPUs available on the same machine, generation can be done in parallel by 
distributing the prompts across the GPUs, and collecting all the results at the end. The number
of GPUs to be used can be specified with the `--gpus` argument.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CIFs from the given prompts.")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to the directory containing the trained model checkpoint file.")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to the gzipped tarball where the generated CIF files will be stored. "
                             "It is recommended that the filename end in `.tar.gz`.")
    parser.add_argument("--prompts", type=str,
                        help="Path to the .tar.gz file containing the prompt .txt files. "
                             "If not provided, CIFs are generated ab initio.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="The top-k value to use during sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=3000,
                        help="The maximum number of tokens to generate per CIF.")
    parser.add_argument("--num_amp_tokens", type=int, default=0,
                        help="Number of <AMP*> tokens to include in tokenizer.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use.")
    parser.add_argument("--temperature", type=float, default=1.0, help="The sampling temperature.")
    parser.add_argument("--seed", type=int, default=1337, help="The random seed.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"],
                        help="The datatype to use.")
    parser.add_argument(
        "--num-gens",
        type=int,
        default=1,
        help=(
            "Number of CIFs to generate when no prompts are provided. "
            "Ignored when using the --prompts option."
        ),
    )
    parser.add_argument("--gpus", type=int,
                        help="The number of GPUs to use. "
                             "The number of GPUs specified must be available on the same machine.")

    args, _ = parser.parse_known_args()
    if args.device == "cuda":
        gpus_avail = torch.cuda.device_count()
    else:
        gpus_avail = 0
    parser.set_defaults(gpus=gpus_avail)
    args = parser.parse_args()

    model_dir = args.model
    prompts_file = args.prompts
    ab_initio = not args.prompts
    out_file = args.out
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    num_amp_tokens = args.num_amp_tokens
    device = args.device
    temperature = args.temperature
    seed = args.seed
    dtype = args.dtype

    if device.startswith("cuda") and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        print(
            "WARNING: CUDA device does not support bfloat16; falling back to float16",
            file=sys.stderr,
        )
        dtype = "float16"
    num_gens = args.num_gens
    gpus = args.gpus

    if device == "cuda" and gpus > gpus_avail:
        print(f"ERROR: There are {gpus_avail} GPU(s) available but {gpus} was specified.")
        sys.exit(1)

    workers = 1 if device == "cpu" else gpus

    if ab_initio:
        prompts = [(i + 1, "data_") for i in range(num_gens)]
    else:
        prompts = get_prompts_from_file(prompts_file)

    total_cifs = len(prompts)
    print(f"will generate {total_cifs} CIFs")

    chunks = array_split(prompts, workers)
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers + 1)  # add an extra worker for writing
    watcher = pool.apply_async(write_listener, (queue, total_cifs, out_file))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        dev = f"cuda:{i}" if device == "cuda" else device
        worker_seed = (seed + i) if ab_initio else seed
        job = pool.apply_async(
            generate,
            (
                model_dir,
                worker_seed,
                dev,
                dtype,
                num_gens,
                temperature,
                top_k,
                max_new_tokens,
                num_amp_tokens,
                chunk,
                queue,
            ),
        )
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put("kill")
    pool.close()
    pool.join()
