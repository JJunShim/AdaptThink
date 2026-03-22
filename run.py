import torch
import torch.multiprocessing as mp

size = 2**4


def burn(rank):
    device = f"cuda:{rank}"
    x = torch.randn(size, size, device=device)
    while True:
        x = x @ x


if __name__ == "__main__":
    mp.spawn(burn, nprocs=torch.cuda.device_count())
