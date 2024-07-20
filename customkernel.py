!pip install torch torchvision matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from ipywidgets import widgets, GridspecLayout, Output, VBox
from IPython.display import display, clear_output

# Load the image
image_path = '/content/stones.jpg'  # Replace with your image path
image = Image.open(image_path)

# Define a transform to convert the image to a tensor
preprocess = transforms.Compose([transforms.ToTensor()])

# Preprocess the image
input_tensor = preprocess(image)
input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

# Initialize convolution kernels
kernel_names = ["Custom"]
kernels = {
    "Custom": np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], dtype=np.float32)
}

# Create a function to apply convolution and update the displayed images
class CustomConvolution(nn.Module):
    def __init__(self, kernel, channels):
        super(CustomConvolution, self).__init__()
        self.channels = channels
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=kernel.shape[0], bias=False) for _ in range(self.channels)])

        for layer in self.conv_layers:
            layer.weight = nn.Parameter(torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0))
            layer.weight.requires_grad = False

    def forward(self, x):
        convolved_channels = [layer(x[:, i:i+1, :, :]) for i, layer in enumerate(self.conv_layers)]
        return torch.cat(convolved_channels, dim=1)

# Create a grid-based widget for custom kernel input
custom_kernel_input = GridspecLayout(3, 3, width='1000px', height='150px')
custom_kernel_text_inputs = []

for i in range(3):
    row = []
    for j in range(3):
        row.append(widgets.FloatText(value=0, description=f"Row {i+1}, Col {j+1}"))
    custom_kernel_text_inputs.append(row)

for i in range(3):
    for j in range(3):
        custom_kernel_input[i, j] = custom_kernel_text_inputs[i][j]

# Initialize modified image variable as a global variable
modified_image = input_tensor.clone()

# Initialize the kernel with a default value of zeros
dynamic_kernel = np.zeros((3, 3), dtype=np.float32)

# Display the output images in one place
output_image = widgets.Output(layout={'border': '1px solid black'})
display(output_image)

# Function to visualize the kernel matrix as an image
def visualize_kernel(kernel, ax, cbar_ax):
    # Normalize the kernel matrix to values between 0 and 1
    normalized_kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
    # Display the kernel as an image
    im = ax.imshow(normalized_kernel, cmap='viridis', interpolation='nearest')
    ax.title.set_text('Kernel Visualization')
    ax.axis('off')
    plt.colorbar(im, cax=cbar_ax, orientation='vertical', fraction=0.046, pad=0.04)

# Function to apply convolution and update the displayed images
def apply_convolution(_):
    global modified_image, dynamic_kernel

    # Get the values from the custom kernel text inputs
    dynamic_kernel = np.array([[custom_kernel_text_inputs[i][j].value for j in range(3)] for i in range(3)], dtype=np.float32)
    custom_convolution = CustomConvolution(dynamic_kernel, channels=3)

    # Apply convolution
    with torch.no_grad():
        result = custom_convolution(input_tensor)
    modified_image = result.clone()

    with output_image:
        clear_output(wait=True)

        # Set up the figure layout with GridSpec
        fig = plt.figure(figsize=(12, 4))  # Adjust the figsize to scale your plots
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])  # Adjust width ratios to match the layout
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        cbar_ax = fig.add_subplot(gs[0, 3])

        # Original Image
        ax1.imshow(input_tensor.squeeze(0).permute(1, 2, 0))
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Convoluted Image
        ax2.imshow(modified_image.squeeze(0).permute(1, 2, 0).clip(0, 1))
        ax2.set_title('Convoluted Image')
        ax2.axis('off')

        # Kernel Visualization
        visualize_kernel(dynamic_kernel, ax3, cbar_ax)

        plt.tight_layout()
        plt.show()

def save_convolved_image(_):
    #clip the color to 0 and 1 then convert to uint8
    convolved_img = modified_image.squeeze(0).permute(1, 2, 0).detach().numpy().clip(0, 1) * 255
    print(convolved_img.size)
    # convolved_img = modified_image.squeeze(0).permute(1, 2, 0).detach().numpy() * 255
    convolved_img = convolved_img.astype(np.uint8)
    convolved_img_pil = Image.fromarray(convolved_img)
    convolved_img_pil.save('/content/convoluted_image1.jpg')
    print("Convoluted image saved successfully!")

# Function to update the custom kernel
def update_custom_kernel(kernel):
    for i in range(3):
        for j in range(3):
            custom_kernel_text_inputs[i][j].value = kernel[i, j]

# Create a dropdown widget for selecting the kernel type
kernel_dropdown = widgets.Dropdown(
    options=kernel_names,
    description="Kernel Type:",
    disabled=False,
)

# Register the apply_convolution function to be called when the dropdown value changes
def dropdown_change(change):
    kernel_type = change.new
    if kernel_type == "Custom":
        # Enable the custom kernel input
        for row in custom_kernel_text_inputs:
            for text_input in row:
                text_input.disabled = False
    else:
        # Disable the custom kernel input for predefined kernels
        for row in custom_kernel_text_inputs:
            for text_input in row:
                text_input.disabled = True
        # Update the custom kernel based on the selected type
        dynamic_kernel = kernels[kernel_type]
        update_custom_kernel(dynamic_kernel)
        # Apply the convolution with the updated custom kernel
        apply_convolution(None)

kernel_dropdown.observe(dropdown_change, names='value')

# Create a button to update the custom kernel
update_button = widgets.Button(description="Update Custom Kernel")
update_button.on_click(apply_convolution)  # Call apply_convolution when the button is clicked

# Create a button to save the convolved image
save_button = widgets.Button(description="Save Convoluted Image")
save_button.on_click(save_convolved_image)  # Call save_convolved_image when the button is clicked

# Assemble UI components in a vertical box layout
ui_elements = VBox([
    kernel_dropdown,
    custom_kernel_input,
    update_button,
    save_button
])

# Display the UI elements
display(ui_elements)

# Display the output image area
output_image = Output()
display(output_image)

# Initialize with default kernel
default_kernel = kernels['Custom']
update_custom_kernel(default_kernel)

# Apply convolution with the default kernel on start
apply_convolution(None)
