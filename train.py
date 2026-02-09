import os, sys, time
import hydra
import torch
import wandb
import logging
import torch.distributed as dist
from utils.trainer import Trainer


@hydra.main(version_base=None, config_path="config", config_name="default")
def run(cfg):
    trainer = Trainer(cfg)
    trainer.launch()
    dist.destroy_process_group()
    logging.info("DONE")


if __name__ == "__main__":
    run()
