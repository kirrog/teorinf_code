import os

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
from src.modeling.get_model import init_model
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

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        config=args_config,
        name=args_config["training_args"]["run_name"],
    )
    set_random_seed(args_config["training_args"]["seed"])

    device = (
        "cuda"
        if torch.cuda.is_available() and args_config["training_args"]["use_cuda"]
        else "cpu"
    )

    model = init_model(args_config["model"])
    model.to(device)

    b_t = args_config["training_args"]["b_t"]
    b = args_config["training_args"].get("b", 2)
    batch_size = args_config["training_args"]["batch_size"]
    learning_rate = args_config["training_args"]["learning_rate"]
    use_aux_loss = args_config["training_args"].get("use_aux_loss", False)
    if use_aux_loss:
        AUX_Loss = PerceptualLoss().to(device)
    aux_lambda = args_config["training_args"].get("aux_lambda", 0)

    train_dataset = ImageDataset(args_config["data"]["train_data_path"])
    test_dataset = ImageDataset(args_config["data"]["test_data_path"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {args_config['training_args']['epochs']}")

    global_step = 0

    for epoch in tqdm(range(args_config["training_args"]["epochs"])):
        model.train()

        for step, train_batch in enumerate(train_loader):
            train_batch = train_batch.to(device)

            optimizer.zero_grad()

            outputs = model(train_batch, b_t=b_t)
            loss = nn.MSELoss()(outputs, train_batch)
            if use_aux_loss:
                aux_loss = AUX_Loss(outputs, train_batch)
                loss += aux_lambda * aux_loss
                run.log({"train/aux_loss": aux_loss.item(), "epoch": epoch, "step": global_step}, step=global_step)

            run.log(
                {"train/loss": loss, "epoch": epoch, "step": global_step},
                step=global_step,
            )

            loss.backward()
            optimizer.step()

            global_step += 1

        if (epoch + 1) % args_config["training_args"]["eval_epochs"] == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for val_batch in tqdm(test_loader):
                    val_batch = val_batch.to(device)
                    val_outputs = model(val_batch, b_t=b_t)
                    test_loss += nn.MSELoss()(val_outputs, val_batch).item()

                    if use_aux_loss:
                        val_aux_loss = AUX_Loss(outputs, train_batch)
                        loss += aux_lambda * aux_loss
                        run.log({"eval/aux_loss": val_aux_loss.item(), "epoch": epoch, "step": global_step}, step=global_step)

            test_loss /= len(test_loader)
            imgs_decoded, imgsQ_decoded, bpp = process_images(
                test_loader, model, device, b
            )
            fig, psnr_decoded, psnr_decoded_q, _ = display_images_and_save_pdf(
                test_dataset, imgs_decoded, imgsQ_decoded, bpp
            )
            run.log(
                {
                    "eval/loss": test_loss,
                    "epoch": epoch,
                    "step": global_step,
                    "eval/cherry_pick": fig,
                    "eval/psnr_ae": psnr_decoded,
                    "eval/psnr_ae_q": psnr_decoded_q,
                    "eval/bpp": np.mean(bpp),
                },
                step=global_step,
            )

            output_dir = os.path.join(
                args_config["training_args"]["output_dir"],
                args_config["training_args"]["run_name"],
            )
            save_dir = f"epoch_{epoch}"
            model.save(directory=os.path.join(output_dir, save_dir))

    wandb.finish()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    main()
