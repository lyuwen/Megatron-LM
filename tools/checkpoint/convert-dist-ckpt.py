import os
import argparse
try:
  from loguru import logger
except ImportError:
  import logging
  logger = logging.getLogger(__name__)
  logging.basicConfig(encoding='utf-8', level=logging.INFO)
import torch.distributed.run as distrib_run


def _check_megatron_path(path):
    return os.path.exists(f"{path}/megatron") and os.path.exists(f"{path}/pretrain_gpt.py")


def find_megatron_path(megatron_path):
    if megatron_path is not None:
        if _check_megatron_path(megatron_path):
            return megatron_path
        else:
            raise ValueError(f"Megatron not found in {megatron_path=}.")
    #1 Megatron lib is in current working directory
    megatron_path = os.getcwd()
    if _check_megatron_path(megatron_path):
        return megatron_path
    #2 Megatron lib is in the same directory as the script
    megatron_path = os.path.dirname(os.path.abspath(__file__))
    if _check_megatron_path(megatron_path):
        return megatron_path
    #3 Megatron lib is in the second outter directory from the script
    megatron_path = os.path.dirname(os.path.dirnamepath(megatron_path))
    if _check_megatron_path(megatron_path):
        return megatron_path
    raise ValueError(f"Megatron not found.")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", "-N", type=int, required=True, help="Number of nodes.")
    parser.add_argument("--nproc_per_node", "-n", type=int, required=True, help="Number of GPUs per node.")
    parser.add_argument("--save", type=str, required=True, help="Checkpoint save directory.")
    parser.add_argument("--load", type=str, required=True, help="Checkpoint load directory.")
    parser.add_argument('--ckpt-format', default='torch', choices=['torch', 'zarr'], help='Checkpoint format to use.')
    parser.add_argument('--megatron-path', help='Megatron code path.')

    args = parser.parse_args(args)

    megatron_path = find_megatron_path(args.megatron_path)
    args.megatron_path = megatron_path
    logger.info(f"Megatron-Core library found in {megatron_path}, updating environ variable.")
    os.environ["PYTHONPATH"] = "{megatron_path}:" + os.environ.get("PYTHONPATH", "")
    return args


def main(args=None):
    args = parse_args()
    margs = [
        f"--nproc_per_node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--standalone",
        f"{args.megatron_path}/pretrain_gpt.py",
        f"--skip-train",
        f"--save={args.save}",
        f"--load={args.load}",
        f"--bf16",
        f"--save-interval=1",
        f"--micro-batch-size=1",
        f"--global-batch-size=1024",
        f"--lr=0.1",
        f"--mock-data",
        f"--train-iters=1",
        f"--tokenizer-type=NullTokenizer",
        f"--vocab-size=1",
        f"--no-load-optim",
        f"--no-load-rng",
        f"--use-checkpoint-args",
        f"--ckpt-convert-format={args.ckpt_format}",
        f"--ckpt-convert-save={args.save}",
        ]
    logger.info(f"Running command: torchrun %s" % (" ".join(margs)))

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # distrib_run.main(margs)


if __name__ == '__main__':
  main()

