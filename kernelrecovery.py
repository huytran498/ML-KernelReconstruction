!pip install torch torchvision matplotlib
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from ipywidgets import widgets, GridspecLayout, Output, VBox
from IPython.display import display, clear_output

# Load your original and convoluted images using PIL
original_img = Image.open("/content/stones.jpg")
convoluted_img = Image.open("/content/convoluted_image1.jpg")

# Convert them to tensors
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

original_image_tensor = transform(original_img)
convoluted_image_tensor = transform(convoluted_img)

class KernelRecoveryCNN(nn.Module):
    def __init__(self):
        super(KernelRecoveryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x

# Initialize the Kernel Recovery CNN model
model = KernelRecoveryCNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the number of epochs
num_epochs = 5000

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(original_image_tensor)
    loss = criterion(outputs, convoluted_image_tensor)
    loss.backward()
    optimizer.step()

# Calculate MSE
with torch.no_grad():
    model.eval()
    reconstructed_image = model(original_image_tensor)
    mse = criterion(reconstructed_image, convoluted_image_tensor)

# Retrieve the learned 3x3 kernel matrix from the model
recovered_kernel = model.conv1.weight.detach().squeeze().cpu().numpy()

# Create a grid of FloatText widgets to display the kernel values
custom_kernel_output = widgets.GridspecLayout(3, 3, width='1000px', height='150px')
for i in range(3):
    for j in range(3):
        formatted_value = "{:.7f}".format(recovered_kernel[i][j])
        custom_kernel_output[i, j] = widgets.FloatText(
            value=float(formatted_value),
            description=f"Row {i+1}, Col {j+1}",
            disabled=True,
            layout=widgets.Layout(width='auto')
        )

# Now you can display your FloatText grid
print("Recovered 3x3 Kernel Matrix:\n")
display(custom_kernel_output)

# Display the recovered 3x3 kernel matrix and MSE

print("Mean Squared Error (MSE):", mse.item())

# Visualize original image, convoluted image, and the kernel matrix
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original image
axes[0].imshow(original_img, cmap='gray')
axes[0].axis('off')  # Turn off axis
axes[0].set_title('Original Image')

# Convolved image
axes[1].imshow(convoluted_img, cmap='gray')
axes[1].axis('off')  # Turn off axis
axes[1].set_title('Convoluted Image')

# Kernel matrix
# Normalize the kernel matrix to values between 0 and 1
normalized_kernel = (recovered_kernel - np.min(recovered_kernel)) / (np.max(recovered_kernel) - np.min(recovered_kernel))
axes[2].imshow(normalized_kernel, cmap='viridis', interpolation='nearest')
axes[2].axis('off')  # Turn off axis
axes[2].set_title('Recovered Kernel Matrix')

# Show colorbar for the kernel matrix
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()