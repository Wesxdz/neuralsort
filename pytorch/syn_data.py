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

def place_scanlines(input_matrix, program_open_set):

    x, y, width, height = program_open_set
    height = min(input_size-y, height)
    width = min(input_size-x, width)

    indices = []
    for y in range(y, y+height):
        if y % 2 == 0:
            indices.extend([y*input_size + x for x in range(x,x+width)])
    np.put(input_matrix, indices, 1)

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