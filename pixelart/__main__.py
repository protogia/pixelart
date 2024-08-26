# PYTHON_ARGCOMPLETE_OK

from palettes import protogia
from PIL import Image
from argparse import ArgumentParser
from rich_argparse import RichHelpFormatter
from rich import print
from rich.prompt import Prompt
import argcomplete
import os
import cv2
import numpy as np

def create_pixelart(image_path, output_path, width=200, height=200, resolution=48, colorcount=16):
    try:
        img = Image.open(image_path)
        target_size = (width, height)
        # reduce resolution
        # https://stackoverflow.com/questions/62282695/reduce-image-resolution
        img.thumbnail([resolution, resolution])
        pilImage = img.transform(target_size, Image.EXTENT, (0,0, resolution, resolution))
        opencvImage = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)
        
        # reduce colorset
        target_image = kmeans_color_quantization(opencvImage, clusters=colorcount, rounds=1)
        
        # apply target-color-palette
        target_image = flann_palette_colorization(target_image)
        
        # save
        cv2.imwrite(output_path, target_image)
        print(f"[green]Pixel art image (size: {target_size[0]}x{target_size[1]}) saved to:[/green] {output_path}")
    except Exception as e:
        print(f"[red]Error processing image:[/red] {e}")



def kmeans_color_quantization(image, clusters=8, rounds=1):
    # https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


def flann_palette_colorization(image):
    width, height, _ = image.shape
    # set up FLANN
    # somewhat arbitrary parameters because under-documented
    norm = cv2.NORM_L2
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    fm = cv2.FlannBasedMatcher(index_params, search_params)

    # make up a palette and give it to FLANN
    protogiaColors = protogia.Palette()
    palette = np.uint8(protogiaColors.hex_to_rgb())
    
    fm.add(np.float32([palette])) # extra dimension is "pictures", unused
    fm.train()

    # find nearest neighbor matches for all pixels
    queries = image.reshape((-1, 3)).astype(np.float32)
    matches = fm.match(queries)

    # get match indices and distances
    assert len(palette) <= 256
    indices = np.uint8([m.trainIdx for m in matches]).reshape(height, width)
    dist = np.float32([m.distance for m in matches]).reshape(height, width)

    # indices to palette colors
    output = palette[indices]
    return output


def parse_arguments():
    parser = ArgumentParser(
        fromfile_prefix_chars=["@", "--"],
        formatter_class=RichHelpFormatter)

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str, default="pixelated_image.jpg",
        help="Path to save the pixelated image (default: pixelated_image.jpg)"
    )
    
    parser.add_argument("-r", "--resolution",
        type=int,
        default=48,
        help="Resolution of resulting image"
    )

    parser.add_argument("-c", "--colorcount",
        type=int,
        default=16,
        help="Maximum number of different colors in resulting image. If source-image contains 128 different colors and --colorcount is 16, the most matching colors will be grouped to target-colors"
    )

    argcomplete.autocomplete(parser)
    cliArgs = parser.parse_args()  
    return cliArgs
    

def main():
    cliArgs = parse_arguments()
    
    if os.path.exists(cliArgs.output):
        confirmation = Prompt.ask("[yellow]The output file already exists. Overwrite? (y/N)[/yellow]")
        if confirmation.lower() != "y":
            print("[info]No actions. Exit.")
            exit(0)

    create_pixelart(
        image_path=cliArgs.image_path,
        output_path=cliArgs.output,
        resolution=cliArgs.resolution,
        colorcount=cliArgs.colorcount
    )
    
    
if __name__ == "__main__":
    main()
