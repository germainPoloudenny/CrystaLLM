import torch
from torch.distributed import destroy_process_group
import torch.distributed as dist
import signal
import sys


def safe_exit():
    if torch.distributed.is_initialized():
        dist.barrier()
        destroy_process_group()
    try:
        # Ajoutez ici des actions de nettoyage si nécessaire
        sys.exit(0)  # Quitter proprement avec un code de sortie
    except BaseException as e:
        raise