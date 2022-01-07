print("Please this code in a Linux system. Deepsparse does not support MacOS.")

from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs
import time
import argparse

parser = argparse.ArgumentParser(description='Convert model to deepsparse')
parser.add_argument('--onnx_path',
                        type = str,
                        default = "fom_gen.onnx",
                        help = 'path to the onnx file')

if __name__ == '__main__':
    args = parser.parse_args()
    onnx_filepath = args.onnx_path
    batch_size = 1

    # Generate random sample input
    inputs = generate_random_inputs(onnx_filepath, batch_size)

    # Compile and run
    engine = compile_model(onnx_filepath, batch_size)

    for i in range(0, 100):
        ss = time.time()
        outputs = engine.run(inputs)
        print("Deesparse inference time", time.time() - ss)

