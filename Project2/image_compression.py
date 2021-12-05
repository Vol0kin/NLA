import numpy as np
import imageio
import argparse

def compress_image(image_path, rank):
    image = imageio.imread(image_path)

    print(image.shape)

    u, s, v = np.linalg.svd(image, full_matrices=False)

    compressed_image = np.dot(u[:, :rank], np.dot(np.diagflat(s[:rank]), v[:rank, :]))

    imageio.imwrite(f'output_{rank}.jpeg', compressed_image)

# Create argument parser
arg_parser = argparse.ArgumentParser(
    description='Compress an image with a low rank approximation using SVD'
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

print(args)
compress_image(args.image, args.rank)