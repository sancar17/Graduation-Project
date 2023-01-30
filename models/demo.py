import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os


def setup_process(rank, master_addr, master_port, world_size, backend='ncll'):
    print(f'setting up {rank=} {world_size=} {backend=}')

    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"{master_addr=} {master_port=}")

    # Initializes the default distributed process group, and this will also initialize the distributed package.
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    exit()
    print(f"{rank=} init complete")
#    dist.destroy_process_group()
    print(f"{rank=} destroy complete")
        
if __name__ == '__main__':
    world_size = 2
    master_addr = 'localhost'
    master_port = '12355'
    mp.spawn(setup_process, args=(master_addr,master_port,world_size,), nprocs=world_size)
