import onnxruntime as rt
import numpy as np
from PIL import Image
import os
import onnx

# global current file path
g_file_path = os.path.abspath(os.path.dirname(__file__))

# Load the ONNX model
model_path = os.path.join(g_file_path, "./model/mnist-12.onnx")


# check model
# model = onnx.load(model_path)
# onnx.checker.check_model(model)
# print(f"Model {model_path} loaded successfully.")

# inference
session = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Prepare input data (assuming image is a 28x28 grayscale image)
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match MNIST input size
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array.reshape(1, 1, 28, 28)  # Add batch dimension
    img_array = img_array.astype(np.float32) / 256.0  # Normalize pixel values
    return img_array

# Load and preprocess the image
image_path = "./image_1.png"  # Replace with your image path
input_data = preprocess_image(image_path)

# Get input and output names from model metadata
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

for input in session.get_inputs():
    print(f'input:{input}, type:{type(input)}')
for output in session.get_outputs():
    print(f'output:{output}, type:{type(output)}')

# Run inference
result = session.run([output_name], {input_name: input_data})

# Process the result
print(f'result:{result}, type:{type(result)}')
output = result[0]
predicted_class = np.argmax(output)

print(f"Predicted digit: {predicted_class}")

# Optional: Display the image
import matplotlib.pyplot as plt
plt.imshow(Image.open(image_path), cmap='gray')
plt.title(f'Predicted: {predicted_class}')
plt.show()