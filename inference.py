import os, sys, time
import hydra
import torch
import wandb
import logging
import torch.distributed as dist
from utils.inferencer import Inferencer
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./config", config_name="default")
def run(cfg: DictConfig):
    inferencer = Inferencer(cfg)
    inferencer.launch()
    dist.destroy_process_group()
    logging.info("DONE")


if __name__ == "__main__":
    run()
