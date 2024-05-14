# This code is the simplest example of image compression based on neural networks
# Comparison with JPEG is provided as well
# It is a demonstation for Information Theory course
# Written by Evgeny Belyaev, February 2024.
import math
import os

import numpy
import tensorflow
# import C-implementation of Witten&Neal&Cleary-1987 arithmetic coding as a external module
from EntropyCodec import *
from keras import backend as K
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# source folder with test images
testfolder = "./test/"
# source folder with train images
trainfolder = "./train/"
# size of test and train images
w = 128
h = 128
# If 0, then the training will be started, otherwise the model will be readed from a file
LoadModel = 1
# Training parameters
batch_size = 64
# Number of bits for representation of the layers sample in the training process
bt = 3
# epochs = 3000
epochs = 5
# Model parameters
n1 = 128
n2 = 32
n3 = 16

# Number of images to be compressed and shown from the test folder
NumImagesToShow = 5

# Number of bits for representation of the layers sample
b = 3


# Compute PSNR in RGB domain
def PSNR_RGB(image1, image2):
    width, height = image1.size
    I1 = numpy.array(image1.getdata()).reshape(image1.size[0], image1.size[1], 3)
    I2 = numpy.array(image2.getdata()).reshape(image2.size[0], image2.size[1], 3)
    I1 = numpy.reshape(I1, width * height * 3)
    I2 = numpy.reshape(I2, width * height * 3)
    I1 = I1.astype(float)
    I2 = I2.astype(float)
    mse = numpy.mean((I1 - I2) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        psnr = 100.0
    else:
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    # print("PSNR = %5.2f dB" % psnr)
    return psnr


# Compute PSNR between two vectors
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (
            10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
    )


def LoadImagesFromFolder(foldername):
    dir_list = os.listdir(foldername)
    Nmax = 0
    for name in dir_list:
        try:
            with Image.open(os.path.join(foldername, name)) as img:
                Nmax += 1
        except IOError:
            print("Не удаётся открыть файл как изображение:", name)

    x = numpy.zeros([Nmax, w, h, 3])
    N = 0
    for name in dir_list:
        try:
            with Image.open(os.path.join(foldername, name)) as image:
                I1 = numpy.array(image.getdata()).reshape(
                    image.size[0], image.size[1], 3
                )
                x[N, :, :, :] = I1
                N += 1
                print("Прогресс: N = %i" % N)
        except IOError:
            print("Неизвестный формат изображения для файла:", name)

    return x


# Model training function
def ImageCodecModel(trainfolder):
    input = layers.Input(shape=(w, h, 3))
    # Encoder
    e1 = layers.Conv2D(n1, (7, 7), activation="relu", padding="same")(input)
    e1 = layers.MaxPooling2D((2, 2), padding="same")(e1)
    e2 = layers.Conv2D(n2, (5, 5), activation="relu", padding="same")(e1)
    e2 = layers.MaxPooling2D((2, 2), padding="same")(e2)
    e3 = layers.Conv2D(n3, (3, 3), activation="relu", padding="same")(e2)
    e3 = layers.MaxPooling2D((2, 2), padding="same")(e3)
    # add noise during training (needed for layer quantinzation)
    e3 = e3 + tensorflow.random.uniform(
        tensorflow.shape(e3), 0, tensorflow.math.reduce_max(e3) / pow(2, bt + 1)
    )

    # Decoder
    x = layers.Conv2DTranspose(
        n3, (3, 3), strides=2, activation="relu", padding="same"
    )(e3)
    x = layers.Conv2DTranspose(
        n2, (5, 5), strides=2, activation="relu", padding="same"
    )(x)
    x = layers.Conv2DTranspose(
        n1, (7, 7), strides=2, activation="relu", padding="same"
    )(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    encoder = Model(input, e3)
    decoder = Model(e3, x)
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.summary()

    if LoadModel == 0:
        xtrain = LoadImagesFromFolder(trainfolder)
        xtrain = xtrain / 255
        autoencoder.fit(
            xtrain, xtrain, epochs=epochs, batch_size=batch_size, shuffle=True
        )
        autoencoder.save("autoencodertemp.mdl")
        encoder.save("encoder.mdl")
        decoder.save("decoder.mdl")
    else:
        autoencoder = keras.models.load_model("autoencodertemp.mdl")
        encoder = keras.models.load_model("encoder.mdl")
        decoder = keras.models.load_model("decoder.mdl")
    return encoder, decoder


# Compresses input layer by multi-alphabet arithmetic coding using memoryless source model
def EntropyEncoder(filename, enclayers, size_z, size_h, size_w):
    temp = numpy.zeros((size_z, size_h, size_w), numpy.uint8, "C")
    for z in range(size_z):
        for h in range(size_h):
            for w in range(size_w):
                temp[z][h][w] = enclayers[z][h][w]
    maxbinsize = size_h * size_w * size_z
    bitstream = numpy.zeros(maxbinsize, numpy.uint8, "C")
    StreamSize = numpy.zeros(1, numpy.int32, "C")
    HiddenLayersEncoder(temp, size_w, size_h, size_z, bitstream, StreamSize)
    name = filename
    path = "./"
    fp = open(os.path.join(path, name), "wb")
    out = bitstream[0: StreamSize[0]]
    out.astype("uint8").tofile(fp)
    fp.close()


# Decompresses input layer by multi-alphabet arithmetic coding using memoryless source model
def EntropyDecoder(filename, size_z, size_h, size_w):
    fp = open(filename, "rb")
    bitstream = fp.read()
    fp.close()
    bitstream = numpy.frombuffer(bitstream, dtype=numpy.uint8)
    declayers = numpy.zeros((size_z, size_h, size_w), numpy.uint8, "C")
    FrameOffset = numpy.zeros(1, numpy.int32, "C")
    FrameOffset[0] = 0
    HiddenLayersDecoder(declayers, size_w, size_h, size_z, bitstream, FrameOffset)
    return declayers


# This function is searching for the JPEG quality factor (QF)
# which provides neares compression to TargetBPP
def JPEGRDSingleImage(X, TargetBPP, i):
    X = X * 255
    image = Image.fromarray(X.astype("uint8"), "RGB")
    width, height = image.size
    realbpp = 0
    realpsnr = 0
    realQ = 0
    for Q in range(101):
        image.save("test.jpeg", "JPEG", quality=Q)
        image_dec = Image.open("test.jpeg")
        bytesize = os.path.getsize("test.jpeg")
        bpp = bytesize * 8 / (width * height)
        psnr = PSNR_RGB(image, image_dec)
        if abs(realbpp - TargetBPP) > abs(bpp - TargetBPP):
            realbpp = bpp
            realpsnr = psnr
            realQ = Q
    JPEGfilename = "image%i.jpeg" % i
    image.save(JPEGfilename, "JPEG", quality=realQ)
    return realQ, realbpp, realpsnr


# Main function
if __name__ == "__main__":
    # Load test images
    xtest = LoadImagesFromFolder(testfolder)
    xtest = xtest / 255

    # Train the model
    encoder, decoder = ImageCodecModel(trainfolder)

    # Run the model for first NumImagesToShow images from the test set
    encoded_layers = encoder.predict(xtest, batch_size=NumImagesToShow)
    max_encoded_layers = numpy.zeros(NumImagesToShow, numpy.float16, "C")

    # normalization the layer to interval [0,1)
    for i in range(NumImagesToShow):
        max_encoded_layers[i] = numpy.max(encoded_layers[i])
        encoded_layers[i] = encoded_layers[i] / max_encoded_layers[i]

    # Quantization of layer to b bits
    encoded_layers1 = numpy.clip(encoded_layers, 0, 0.9999999)
    encoded_layers1 = K.cast(encoded_layers1 * pow(2, b), "int32")

    # Encoding and decoding of each quantized layer by arithmetic coding
    bpp = numpy.zeros(NumImagesToShow, numpy.float16, "C")
    declayers = numpy.zeros((NumImagesToShow, 16, 16, 16), numpy.uint8, "C")
    for i in range(NumImagesToShow):
        binfilename = "image%i.bin" % i
        EntropyEncoder(binfilename, encoded_layers1[i], 16, 16, 16)
        bytesize = os.path.getsize(binfilename)
        bpp[i] = bytesize * 8 / (w * h)
        declayers[i] = EntropyDecoder(binfilename, 16, 16, 16)

    # Dequantization and denormalization of each layer
    print(bpp)
    shift = 1.0 / pow(2, b + 1)
    declayers = K.cast(declayers, "float32") / pow(2, b)
    declayers = declayers + shift
    encoded_layers_quantized = numpy.zeros(
        (NumImagesToShow, 16, 16, 16), numpy.double, "C"
    )
    for i in range(NumImagesToShow):
        encoded_layers_quantized[i] = K.cast(
            declayers[i] * max_encoded_layers[i], "float32"
        )
        encoded_layers[i] = K.cast(encoded_layers[i] * max_encoded_layers[i], "float32")
    decoded_imgs = decoder.predict(encoded_layers, batch_size=NumImagesToShow)
    decoded_imgsQ = decoder.predict(
        encoded_layers_quantized, batch_size=NumImagesToShow
    )

    # Shows NumImagesToShow images from the test set
    # For each image the following results are presented
    # Original image
    # Image, represented by the model (without quantization)
    # Image, represented by the model with quantization and compression of the layers samples
    # Corresponding JPEG image at the same compression level
    for i in range(NumImagesToShow):
        title = ""
        plt.subplot(4, NumImagesToShow, i + 1).set_title(title, fontsize=10)
        plt.imshow(xtest[i, :, :, :], interpolation="nearest")
        plt.axis(False)
    for i in range(NumImagesToShow):
        psnr = PSNR(xtest[i, :, :, :], decoded_imgs[i, :, :, :])
        title = "%2.2f" % psnr
        plt.subplot(4, NumImagesToShow, NumImagesToShow + i + 1).set_title(
            title, fontsize=10
        )
        plt.imshow(decoded_imgs[i, :, :, :], interpolation="nearest")
        plt.axis(False)
    for i in range(NumImagesToShow):
        psnr = PSNR(xtest[i, :, :, :], decoded_imgsQ[i, :, :, :])
        title = "%2.2f %2.2f" % (psnr, bpp[i])
        plt.subplot(4, NumImagesToShow, 2 * NumImagesToShow + i + 1).set_title(
            title, fontsize=10
        )
        plt.imshow(decoded_imgsQ[i, :, :, :], interpolation="nearest")
        plt.axis(False)
    for i in range(NumImagesToShow):
        JPEGQP, JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(
            xtest[i, :, :, :], bpp[i], i
        )
        JPEGfilename = "image%i.jpeg" % i
        JPEGimage = Image.open(JPEGfilename)
        title = "%2.2f %2.2f" % (JPEGrealpsnr, JPEGrealbpp)
        plt.subplot(4, NumImagesToShow, 3 * NumImagesToShow + i + 1).set_title(
            title, fontsize=10
        )
        plt.imshow(JPEGimage, interpolation="nearest")
        plt.axis(False)
    plt.show()
