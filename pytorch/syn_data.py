import numpy as np
from PIL import Image
import os

input_size = 28
num_samples = 100
output_dir = 'graphics_programs'

def gen_program_matrix(program_open_set):
    program_matrix = np.zeros((input_size, input_size))
    
    for i, v in enumerate(program_open_set):
        program_matrix[1 + i *2, 1:9] = np.array([b for b in format(v, '08b')])

    return program_matrix

def render_program_matrix(program_open_set):
    program_matrix = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    
    for i, v in enumerate(program_open_set):
        binary = np.array([int(b) for b in format(v, '08b')])
        for j, bit in enumerate(binary):
            if bit == 1:
                program_matrix[1 + i * 2, 1 + j] = [0, 255, 0]  # Green color for ones
            else:
                program_matrix[1 + i * 2, 1 + j] = [128, 128, 128]  # Grey color for zeros

    return program_matrix

def place_scanlines(input_matrix, program_open_set):

    x, y, width, height = program_open_set
    height = min(input_size-y, height)
    width = min(input_size-x, width)

    indices = []
    for y_iter in range(y, y+height):
        if (y_iter-y) % 2 == 0:
            indices.extend([y_iter*input_size + x for x in range(x,x+width)])
    np.put(input_matrix, indices, 1)

    return input_matrix

def render_op(input_matrix, program_open_set):
    print(input_matrix.shape)
    x, y, width, height = program_open_set
    height = min(28 - y, height)
    width = min(28 - x, width)

    indices = []
    for y_iter in range(y, y+height):
        if (y_iter-y) % 2 == 0:
            indices.extend([y_iter*input_size + x for x in range(x,x+width)])

    # Set 1 on all depth channels
    for channel in range(1): # Red
        np.put(input_matrix[:, :, channel], indices, 255)

    return input_matrix

def render_search(input_matrix, program_open_set):
    print(input_matrix.shape)
    x, y, width, height = program_open_set
    height = min(28 - y, height)
    width = min(28 - x, width)

    indices = []
    for y_iter in range(y, y+height):
        if (y_iter-y) % 2 == 0:
            indices.extend([y_iter*input_size + x for x in range(x,x+width)])

    # Set yellow on pixels that are already red, and blue otherwise
    for index in indices:
        if input_matrix[:, :, 0].flat[index] == 255:  # Check if red channel is 255
            input_matrix[:, :, 1].flat[index] = 255  # Set green channel to 255 (yellow)
        else:
            input_matrix[:, :, 1].flat[index] = 255  # Set green channel to 255 (cyan)
            input_matrix[:, :, 2].flat[index] = 255  # Set blue channel to 255 (cyan)

    return input_matrix


if __name__ == "__main__":
    np.random.seed(42)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the synthetic dataset
    for i in range(num_samples):
        input_matrix = np.random.rand(input_size, input_size)

        x = np.random.randint(0, input_size-15)
        y = np.random.randint(0, input_size-15)

        width = np.random.randint(5, 15)
        height = np.random.randint(5, 15)

        output_matrix = place_scanlines(input_matrix.copy(), [x, y, width, height])
        program_matrix = gen_program_matrix([x, y, width, height])

        # Create an operation representation matrix (a black matrix with the rectangle drawn on it)
        op_matrix = np.zeros((input_size, input_size))
        op_matrix[(output_matrix - input_matrix) > 0] = 1

        # Concatenate the input, output, and operation representation matrices
        concat_matrix = np.concatenate((input_matrix, output_matrix, program_matrix, op_matrix), axis=1)

        # Save the concatenated matrix to file as a PNG image
        concat_img = Image.fromarray((concat_matrix * 255).astype(np.uint8))
        concat_img.save(os.path.join(output_dir, f'sample_{i}.png'))

        np.save(os.path.join(output_dir, f'input_{i}.npy'), input_matrix)
        np.save(os.path.join(output_dir, f'output_{i}.npy'), output_matrix)
        np.save(os.path.join(output_dir, f'op_{i}.npy'), op_matrix)
        np.save(os.path.join(output_dir, f'program_{i}.npy'), program_matrix)
        np.save(os.path.join(output_dir, f'program_vars_{i}.npy'), np.array([x, y, width, height]))