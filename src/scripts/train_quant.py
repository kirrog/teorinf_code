import gc
import os
import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN
import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml import CLoader

from src.data.make_dataset import ImageDataset
from src.modeling.get_model import init_model, load_model
from src.utils import (PerceptualLoss, display_images_and_save_pdf,
                       process_images, set_random_seed)

tqdm.pandas()

os.environ["WANDB_PROJECT"] = "codec_ITMO"

import os




@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):



    with open(config_file, "r") as f:
        args_config = yaml.load(f, Loader=CLoader)

    # run = wandb.init(
    #     project=os.getenv("WANDB_PROJECT"),
    #     config=args_config,
    #     name=args_config["training_args"]["run_name"],
    # )
    set_random_seed(args_config["training_args"]["seed"])

    device = (
        "cuda"
        if torch.cuda.is_available() and args_config["training_args"]["use_cuda"]
        else "cpu"
    )

    model = load_model(args_config["model"], model_path=args_config["model_dir"], device=device)
    model.to(device)
    model.eval()

    batch_size = args_config["training_args"]["batch_size"]

    train_dataset = ImageDataset(args_config["data"]["train_data_path"])
    test_dataset = ImageDataset(args_config["data"]["test_data_path"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")

    train_cluster_data = []



    db = DBSCAN(eps=0.009, min_samples=10)
    vector_counter = 0
    vector_aggregator = np.array([0.0] * 16)
    for step, train_batch in enumerate(tqdm(train_loader)):
        train_batch = train_batch.to(device)

        outputs = model.encode(train_batch)
        outputs_cpu = outputs.cpu().detach()
        for case_num in range(outputs_cpu.size()[0]):
            for x_cord in range(outputs_cpu.size()[2]):
                for y_cord in range(outputs_cpu.size()[2]):
                    vector = outputs_cpu[case_num, :, x_cord, y_cord].numpy()
                    vector_aggregator += abs(vector)
                    vector[6] = 0
                    train_cluster_data.append(vector)
                    vector_counter += 1

        if step + 1 % 10 == 0:
            db.fit(train_cluster_data)

            del train_cluster_data
            train_cluster_data = []
        del train_batch, outputs, outputs_cpu
        gc.collect()
    db.fit(train_cluster_data)
    print(f"Total number of vectors: {vector_counter}")
    print(f"Vectors aggregated: {vector_aggregator}")

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    logger.info("***** Training finished *****")


if __name__ == "__main__":
    main()
