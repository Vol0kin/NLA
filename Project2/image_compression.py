import numpy as np
import imageio
import argparse

def bw_compression(image, rank):
    u, s, v = np.linalg.svd(image, full_matrices=False)
    compressed_image = np.dot(u[:, :rank], np.dot(np.diagflat(s[:rank]), v[:rank, :]))

    # Compute error of the low rank approximation using Frobenius norm
    image_norm = np.linalg.norm(image)
    error = np.linalg.norm(image - compressed_image) / image_norm

    return compressed_image, error


def color_compression(image, rank):
    compressed_image = []

    for d in range(3):
        image_single_channel = image[:, :, d]

        # Compress the channel
        u, s, v = np.linalg.svd(image_single_channel, full_matrices=False)
        compressed_channel = np.dot(u[:, :rank], np.dot(np.diagflat(s[:rank]), v[:rank, :]))
        compressed_image.append(compressed_channel)

    # Create compressed image by stacking each channel
    compressed_image = np.dstack(compressed_image)

    # Compute error
    image_norm = np.linalg.norm(image)
    error = np.linalg.norm(image - compressed_image) / image_norm

    return compressed_image, error
    

def compress_image(image_path, rank):
    image = imageio.imread(image_path)

    if image.ndim == 2:
        compressed_image, error = bw_compression(image, rank)
    else:
        compressed_image, error = color_compression(image, rank)

    # Save image
    imageio.imwrite(f'k{rank}_{round(error, 4)}_{image_path}', compressed_image)

    return error


# Create argument parser
arg_parser = argparse.ArgumentParser(
    description='Compress an image with a low rank approximation using SVD',
    allow_abbrev=False
)

arg_parser.add_argument(
    '-i',
    '--image',
    metavar='PATH',
    type=str,
    help='Path to the image'
)

arg_parser.add_argument(
    '-r',
    '--rank',
    metavar='RANK',
    type=int,
    help='Rank of the approximation'
)

args = arg_parser.parse_args()

error = compress_image(args.image, args.rank)

print('Error of the low rank approximation using Frobenius norm: ', error)