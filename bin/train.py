"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""
import os
from dataclasses import dataclass
from typing import Union
from torch.utils.tensorboard import SummaryWriter
import math
import time
import shutil

from crystallm import parse_config
from omegaconf import OmegaConf
import numpy as np
import torch
import pickle
from contextlib import nullcontext

from crystallm import (
    GPT,
    GPTConfig,
)

from typing import Optional

@dataclass
class TrainDefaults:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    tensorboard_dir: Optional[str] = None  # directory for TensorBoard logs
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # data
    dataset: str = ""  # the path to the folder containing the .bin files with encoded tokens
    cond_embeddings: str = ""  # LMDB file with conditioning embeddings
    condition_dataset: str = ""  # optional dataset with conditioning tokens
    condition_length: int = 0  # length of conditioning prefix
    dataset_fraction: float = 1.0  # proportion of the dataset to use
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95  # make a bit bigger because number of tokens per iter is small
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
    backend: str = "nccl"
    underrep_p: float = 0.0
    validate: bool = False  # whether to evaluate the model using the validation set
    use_ckpt_expanded: bool = False  # load ckpt_expanded.pt instead of ckpt.pt when resuming


def read_start_indices(
    max_start_index: int,
    data_dir: str,
    starts_fname: str,
    on_condition: bool = True,
    required: bool = False,
) -> Union[torch.Tensor, None]:
    start_indices = None
    starts_path = os.path.join(data_dir, starts_fname)
    if on_condition:
        if os.path.exists(starts_path):
            print(f"Reading start indices from {starts_path}...")
            with open(starts_path, "rb") as f:
                start_indices = torch.tensor(pickle.load(f))  # should be sorted
            # remove indices that would result in out-of-bounds sequences
            start_indices = start_indices[start_indices <= max_start_index]
        elif required:
            raise Exception(f"Expected to find a file in dataset dir named '{starts_fname}'")
    return start_indices


if __name__ == "__main__":
    C = parse_config(TrainDefaults)

    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    load_dir = C.out_dir
    C.out_dir = f"{C.out_dir}_{timestamp}"
    ckpt_filename = "ckpt_expanded.pt" if C.use_ckpt_expanded else "ckpt.pt"

    if C.init_from == "resume":
        print(f"Resuming training from {load_dir}...")
        print(f"Creating clone directory {C.out_dir}...")
        shutil.copytree(load_dir, C.out_dir, dirs_exist_ok=True)
    else:
        print(f"Creating {C.out_dir}...")
        os.makedirs(C.out_dir, exist_ok=True)

    if C.tensorboard_dir is None:
        C.tensorboard_dir = C.out_dir

    torch.distributed.init_process_group(backend=C.backend)
    ddp_rank = int(os.environ.get("RANK", 0))
    ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if "cuda" in C.device else C.device
    torch.cuda.set_device(device)
    C.device = device
    seed_offset = ddp_rank

    master_process = ddp_rank == 0

    writer = SummaryWriter(C.tensorboard_dir) if master_process and C.tensorboard_dir else None

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")

    train_data = np.memmap(os.path.join(C.dataset, "train.bin"), dtype=np.uint16, mode="r")
    
    val_data = np.memmap(os.path.join(C.dataset, "val.bin"), dtype=np.uint16, mode="r") if C.validate else None
    #print(len(train_data))
    cond_train_data = None
    cond_val_data = None
    if C.condition_dataset:
        cond_train_data = np.memmap(
            os.path.join(C.condition_dataset, "train.bin"), dtype=np.uint16, mode="r"
        )
        #print(len(train_data))
        cond_val_data = (
            np.memmap(os.path.join(C.condition_dataset, "val.bin"), dtype=np.uint16, mode="r")
            if C.validate
            else None
        )
        cond_meta_path = os.path.join(C.condition_dataset, "meta.pkl")
        if os.path.exists(cond_meta_path):
            with open(cond_meta_path, "rb") as f:
                cond_meta = pickle.load(f)
            if C.condition_length <= 0:
                C.condition_length = int(cond_meta.get("condition_length", 0))
        if C.condition_length <= 0:
            raise ValueError("condition_length must be > 0 when using condition_dataset")

    if not 0 < C.dataset_fraction <= 1.0:
        raise ValueError("dataset_fraction must be in the (0, 1] range")
    train_len = int(len(train_data) * C.dataset_fraction)
    train_data = train_data[:train_len]
    if val_data is not None:
        val_len = int(len(val_data) * C.dataset_fraction)
        val_data = val_data[:val_len]

    amp_token_ids = []

    cif_start_indices = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts.pkl",
    )

    cif_start_indices_val = read_start_indices(
        max_start_index=(len(val_data) - C.block_size) if C.validate else -1,
        data_dir=C.dataset,
        starts_fname="starts_val.pkl",
        on_condition=C.validate,
    )

    cif_start_indices_underrep = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts_underrep.pkl",
        on_condition=C.underrep_p > 0,
        required=True,
    )

    def get_batch(split):
        data = train_data if split == "train" else val_data
        cond_data = cond_train_data if split == "train" else cond_val_data

        ix = torch.randint(len(data) - C.block_size, (C.batch_size,))
        if split == "train":
            if C.underrep_p is not None and np.random.random() < C.underrep_p:
                ix = cif_start_indices_underrep[torch.randperm(len(cif_start_indices_underrep))[:C.batch_size]]
            elif cif_start_indices is not None:
                ix = cif_start_indices[torch.randperm(len(cif_start_indices))[:C.batch_size]]
        elif cif_start_indices_val is not None:
            ix = cif_start_indices_val[torch.randperm(len(cif_start_indices_val))[:C.batch_size]]

        x = torch.stack([torch.from_numpy((data[i:i + C.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + C.block_size]).astype(np.int64)) for i in ix])

        if cond_data is not None:
            cond_ix = torch.randint(len(cond_data) - C.condition_length, (C.batch_size,))
            cond_x = torch.stack([
                torch.from_numpy((cond_data[i:i + C.condition_length]).astype(np.int64))
                for i in cond_ix
            ])
            cond_y = torch.full_like(cond_x, -1)
            x = torch.cat([cond_x, x], dim=1)
            y = torch.cat([cond_y, y], dim=1)

        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(C.device, non_blocking=True), y.pin_memory().to(C.device, non_blocking=True)
        else:
            x, y = x.to(C.device), y.to(C.device)
        return x, y

    iter_num = 0
    best_val_loss = 1e9

    if C.condition_dataset:
        meta_path = os.path.join(C.condition_dataset, "meta.pkl")
    else:
        meta_path = os.path.join(C.dataset, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    cond_matrix = None
    if C.cond_embeddings:
        from crystallm import embeddings_from_lmdb
        print(f"Loading conditioning embeddings from {C.cond_embeddings}...")
        cond_data = embeddings_from_lmdb(C.cond_embeddings)
        first_vec = next(iter(cond_data.values()))
        cond_dim = len(first_vec)
        print(f"Conditioning embedding dimension = {cond_dim}")
        if meta_vocab_size is None:
            raise ValueError("conditioning requires meta.pkl with vocab")
        cond_matrix = torch.zeros((meta_vocab_size, cond_dim), dtype=torch.float32)
        for tok, idx in meta["stoi"].items():
            vec = cond_data.get(tok)
            if vec is not None:
                cond_matrix[idx] = torch.tensor(vec, dtype=torch.float32)

    total_block_size = C.block_size + (C.condition_length if C.condition_dataset else 0)

    model_args = dict(
        n_layer=C.n_layer,
        n_head=C.n_head,
        n_embd=C.n_embd,
        block_size=total_block_size,
        bias=C.bias,
        vocab_size=None,
        dropout=C.dropout,
    )
    if cond_matrix is not None:
        model_args["cond_emb_dim"] = cond_matrix.shape[1]
    if C.init_from == "scratch":
        print("Initializing a new model from scratch...")
        if meta_vocab_size is None:
            print("Defaulting to vocab_size of 371...")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 371
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif C.init_from == "resume":
        print(f"Loading checkpoint from {load_dir}...")
        ckpt_path = os.path.join(load_dir, ckpt_filename)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        checkpoint_model_args = checkpoint["model_args"]
        ckpt_block_size = checkpoint_model_args.get("block_size", total_block_size)
        # force these config attributes to be equal otherwise we can't even resume training;
        #  the rest of the attributes (e.g. dropout) can stay as desired
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "cond_emb_dim"]:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        if total_block_size > ckpt_block_size:
            model.expand_block_size(total_block_size)
            model_args["block_size"] = total_block_size

    if cond_matrix is not None:
        model.cond_embedding.weight.data.copy_(cond_matrix)
        model.cond_embedding.weight.requires_grad = False

    # crop down the model block size if desired, using model surgery
    if total_block_size < model.config.block_size:
        model.crop_block_size(total_block_size)
        model_args["block_size"] = total_block_size  # so that the checkpoint will have the right value
    if C.cond_embeddings:
        for name, param in model.named_parameters():
            param.requires_grad = False
    if cond_matrix is not None:
        for param in model.cond_proj.parameters():
            param.requires_grad = True
    model.to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    # if C.init_from == "resume":
    #     optimizer.load_state_dict(checkpoint["optimizer"])

    if C.compile:
        print("Compiling the model (takes a ~minute)...")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, eval_iters in [("train", C.eval_iters_train), ("val", C.eval_iters_val)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < C.warmup_iters:
            return C.learning_rate * it / C.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > C.lr_decay_iters:
            return C.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return C.min_lr + coeff * (C.learning_rate - C.min_lr)

    # training loop
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % C.eval_interval == 0:
            if master_process:
                losses = estimate_loss()
                if C.validate:
                    print(
                        f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )
                else:
                    print(f"step {iter_num}: train loss {losses['train']:.4f}")

                if writer:
                    writer.add_scalar("loss/train", losses["train"], iter_num)
                    if C.validate:
                        writer.add_scalar("loss/val", losses["val"], iter_num)

                should_save = False
                if C.validate and losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    should_save = True
                elif not C.validate and C.always_save_checkpoint:
                    should_save = True

                if should_save and iter_num > 0:
                    state = model.module.state_dict()
                    checkpoint = {
                        "model": state,
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": dict(C),
                    }
                    print(f"saving checkpoint to {C.out_dir}...")
                    torch.save(checkpoint, os.path.join(C.out_dir, ckpt_filename))
            if torch.is_tensor(best_val_loss):
                loss_tensor = best_val_loss.clone().detach().to(C.device)
            else:
                loss_tensor = torch.tensor(best_val_loss, device=C.device)
            torch.distributed.broadcast(loss_tensor, src=0)
            best_val_loss = loss_tensor.item()
        if iter_num == 0 and C.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % C.log_interval == 0 and master_process:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu_src = model.module
                mfu = mfu_src.estimate_mfu(C.batch_size * C.gradient_accumulation_steps * ddp_world_size, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
            if writer:
                writer.add_scalar("loss/train_step", lossf, iter_num)
                writer.add_scalar("lr", lr, iter_num)
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > C.max_iters:
            break

    torch.distributed.destroy_process_group()
    if writer:
        writer.close()
