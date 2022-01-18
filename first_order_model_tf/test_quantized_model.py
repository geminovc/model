import numpy as np
import tensorflow as tf
import time

# Load the TFLite model and allocate tensors.
generator_interpreter = tf.lite.Interpreter(model_path="tflite/vox_float16/generator.tflite")
generator_interpreter.allocate_tensors()
input_details = generator_interpreter.get_input_details()
output_details = generator_interpreter.get_output_details()
tt = []

for i in range(0, 100):
    for j in range(0, len(input_details)):
        # Test the model on random input data.
        input_shape = input_details[j]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        generator_interpreter.set_tensor(input_details[j]['index'], input_data)
    # Run inference
    start_time = time.time()
    generator_interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    output_data = generator_interpreter.get_tensor(output_details[0]['index'])
    tt.append(time.time() - start_time)

print("Average Inference time is:", sum(tt)/len(tt))


