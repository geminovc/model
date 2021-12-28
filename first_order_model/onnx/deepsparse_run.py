from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs
import time

onnx_filepath = "fom_gen.onnx"
batch_size = 1

# Generate random sample input
inputs = generate_random_inputs(onnx_filepath, batch_size)

# Compile and run
engine = compile_model(onnx_filepath, batch_size)

for i in range(0, 100):
	ss = time.time()
	outputs = engine.run(inputs)
	print("Deesparse inference time", time.time() - ss)                                                      
