import os
import glob
from PIL import Image

# Set input and output directories
input_dir = './sort_stage'
state_dir = './hf'
output_dir = '.'

# Set GIF settings
gif_filename = 'output.gif'
fps = 10  # frames per second

# Get list of subdirectories
subdirs = [os.path.join(input_dir, str(i)) for i in range(1, 1000) if os.path.isdir(os.path.join(input_dir, str(i)))]

# Initialize image list
images = []

# Iterate over subdirectories and gather images
for subdir in subdirs:
    # Get the corresponding state directory
    state_subdir = os.path.join(state_dir, os.path.basename(subdir))

    # Check if the state directory exists and contains program_state.txt
    if os.path.exists(state_subdir) and os.path.exists(os.path.join(state_subdir, 'program_state.txt')):
        # Read the program state file to determine the image index
        with open(os.path.join(state_subdir, 'program_state.txt'), 'r') as f:
            lines = f.readlines()
            index = int(lines[0].strip().split()[0])  # assume the index is the first number on the first line

        # Load the corresponding image
        img_file = os.path.join(subdir, f'program_{index}.png')
        if os.path.exists(img_file):
            images.append(Image.open(img_file))

# Save GIF
if images:
    images[0].save(os.path.join(output_dir, gif_filename),
                   save_all=True,
                   append_images=images[1:],
                   optimize=False,
                   duration=int(1000/fps),
                   loop=0)
else:
    print("No images found.")