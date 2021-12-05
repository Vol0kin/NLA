import numpy as np
import imageio
import argparse

def compress_image(image_path, rank):
    image = imageio.imread(image_path)

    if image.ndim != 2:
        image = image[:, :, 0]

    # Use SVD to compute a low rank approximation of the original image
    u, s, v = np.linalg.svd(image, full_matrices=False)
    compressed_image = np.dot(u[:, :rank], np.dot(np.diagflat(s[:rank]), v[:rank, :]))

    # Compute error of the low rank approximation using Frobenius norm
    image_norm = np.linalg.norm(image)
    error = np.linalg.norm(image - compressed_image) / image_norm

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