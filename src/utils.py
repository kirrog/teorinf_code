import io
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from EntropyCodec import *
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models import vgg16

from src.quntization import de_quntization_by_mask, quntization_by_mask

to_pil_transform = transforms.ToPILImage()


def PSNR_RGB(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    mse = torch.mean(torch.square(y_pred - y_true))
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(max_pixel ** 2 / mse)).item()


def EntropyEncoder(enc_img, size_z, size_h, size_w):
    temp = enc_img.astype(np.uint8).copy()

    maxbinsize = size_h * size_w * size_z
    bitstream = np.zeros(maxbinsize, np.uint8)
    StreamSize = np.zeros(1, np.int32)
    HiddenLayersEncoder(temp, size_w, size_h, size_z, bitstream, StreamSize)
    return bitstream[: StreamSize[0]]


def EntropyDecoder(bitstream, size_z, size_h, size_w):
    decoded_data = np.zeros((size_z, size_h, size_w), np.uint8)
    FrameOffset = np.zeros(1, np.int32)
    HiddenLayersDecoder(decoded_data, size_w, size_h, size_z, bitstream, FrameOffset)
    return decoded_data


def process_images(test_loader, model, device, b, w=128, h=128, mask_quantization=False):
    imgs_encoded = []
    imgs_decoded = []

    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            encoded_images = model.encoder(test_batch)
            if mask_quantization:
                encoded_images = de_quntization_by_mask(quntization_by_mask(encoded_images))
            decoded_images = model.decoder(encoded_images)

            imgs_encoded.append(encoded_images.cpu().detach())
            imgs_decoded.append(decoded_images.cpu().detach())

    imgs_encoded = torch.vstack(imgs_encoded)
    if mask_quantization:
        imgs_encoded = quntization_by_mask(imgs_encoded)
    imgs_decoded = torch.vstack(imgs_decoded)

    max_encoded_imgs = imgs_encoded.amax(dim=(1, 2, 3), keepdim=True)  # ---
    # Normalize and quantize
    norm_imgs_encoded = imgs_encoded / max_encoded_imgs
    quantized_imgs_encoded = (torch.clip(norm_imgs_encoded, 0, 0.9999999) * pow(2, b)).to(
        torch.int32
    )
    quantized_imgs_encoded = quantized_imgs_encoded.numpy()

    # Encode and decode using entropy coding
    quantized_imgs_decoded = []
    bpp = []

    for i in range(quantized_imgs_encoded.shape[0]):
        size_z, size_h, size_w = quantized_imgs_encoded[i].shape
        encoded_bits = EntropyEncoder(quantized_imgs_encoded[i], size_z, size_h, size_w)
        byte_size = encoded_bits.nbytes
        bpp.append(byte_size * 8 / (w * h))
        quantized_imgs_decoded.append(EntropyDecoder(encoded_bits, size_z, size_h, size_w))
    quantized_imgs_decoded = torch.tensor(np.array(quantized_imgs_decoded, dtype=np.uint8))

    shift = 1.0 / pow(2, b + 1)
    dequantized_imgs_decoded = (quantized_imgs_decoded.to(torch.float32) / pow(2, b)) + shift
    dequantized_denorm_imgs_decoded = dequantized_imgs_decoded * max_encoded_imgs  # ---
    if mask_quantization:
        dequantized_denorm_imgs_decoded = de_quntization_by_mask(dequantized_denorm_imgs_decoded)

    imgsQ_decoded = []

    with torch.no_grad():
        for deq_img in dequantized_denorm_imgs_decoded:
            deq_img = deq_img.to(device)[None, :]
            decoded_imgQ = model.decoder(deq_img)[0]

            imgsQ_decoded.append(decoded_imgQ.cpu().detach())

    imgsQ_decoded = torch.stack(imgsQ_decoded)

    assert imgsQ_decoded.shape == imgs_decoded.shape
    assert imgsQ_decoded.shape[0] == len(bpp)

    return imgs_decoded, imgsQ_decoded, bpp


def JPEGRDSingleImage(torch_img, TargetBPP):
    image = to_pil_transform(torch_img)

    width, height = image.size
    realbpp = 0
    realpsnr = 0
    realQ = 0
    final_image = None

    for Q in range(101):
        img_bytes = io.BytesIO()
        image.save(img_bytes, "JPEG", quality=Q)
        img_bytes.seek(0)
        image_dec = Image.open(img_bytes)
        bytesize = len(img_bytes.getvalue())

        bpp = bytesize * 8 / (width * height)
        psnr = PSNR_RGB(np.array(image), np.array(image_dec))
        rbpp_bigger_bpp = abs(realbpp - TargetBPP) > abs(bpp - TargetBPP)
        # if abs(realbpp - TargetBPP) > abs(bpp - TargetBPP):
        realbpp = bpp
        realpsnr = psnr
        realQ = Q
        final_image = image_dec

    return final_image, realQ, realbpp, realpsnr


def display_images_and_save_pdf(test_dataset, imgs_decoded, imgsQ_decoded, bpp, filepath=None, NumImagesToShow=None):
    if NumImagesToShow is None:
        NumImagesToShow = len(test_dataset)
    cols = NumImagesToShow
    rows = 4

    fig = plt.figure(figsize=(2 * cols, 2 * rows))
    psnr_decoded = []
    psnr_decoded_q = []
    psnr_jpeg = []

    for i in range(NumImagesToShow):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(to_pil_transform(test_dataset[i]), interpolation="nearest")
        plt.title("", fontsize=10)
        plt.axis('off')

    for i in range(NumImagesToShow):
        psnr = PSNR(test_dataset[i], imgs_decoded[i])
        psnr_decoded.append(psnr)
        plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(to_pil_transform(imgs_decoded[i]), interpolation="nearest")
        plt.title(f"PSNR: {psnr:.2f}", fontsize=10)
        plt.axis('off')

    for i in range(NumImagesToShow):
        psnr = PSNR(test_dataset[i], imgsQ_decoded[i])
        psnr_decoded_q.append(psnr)
        plt.subplot(rows, cols, 2 * cols + i + 1)
        plt.imshow(to_pil_transform(imgsQ_decoded[i]), interpolation="nearest")
        plt.title(f"PSNR: {psnr:.2f} | BPP: {bpp[i]:.2f}", fontsize=10)
        plt.axis('off')

    for i in range(NumImagesToShow):
        jpeg_img, JPEGQP, JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(test_dataset[i], bpp[i])
        psnr_jpeg.append(JPEGrealpsnr)
        plt.subplot(rows, cols, 3 * cols + i + 1)
        plt.imshow(jpeg_img, interpolation="nearest")
        plt.title(f"PSNR: {JPEGrealpsnr:.2f} | BPP: {JPEGrealbpp:.2f}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, format='pdf')
    return fig, np.mean(psnr_decoded), np.mean(psnr_decoded_q), np.mean(psnr_jpeg)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:23]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, reconstructed, original):
        loss = F.mse_loss(self.feature_extractor(reconstructed), self.feature_extractor(original))
        return loss
