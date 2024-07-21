# README for Convolution Kernel Visualizer

## Overview
This project provides a set of Python scripts for visualizing and applying various convolution kernels to an image using PyTorch. It includes functionalities for defining custom kernels, selecting predefined kernels, and saving the convolved images. The project is designed to be interactive, utilizing widgets to adjust kernel values and observe the effects in real time.

## Files

### 1. `customkernel.py`
This script is designed to visualize and apply a custom convolution kernel to an image. Key functionalities include:

- Loading an image and converting it to a tensor.
- Defining and initializing a custom convolution kernel.
- Applying the custom kernel to the image using PyTorch's convolution operations.
- Displaying the image before and after applying the kernel.
- Providing a UI with widgets for custom kernel input and updating the convolution result interactively.

### 2. `kernelrecovery.py`
This script appears to be focused on recovering and visualizing the effects of known convolution kernels on an image. Key features include:

- Loading an image and converting it to a tensor.
- Predefining known kernels such as Gaussian, Sobel, and Laplacian.
- Applying these kernels to the image using PyTorch's convolution operations.
- Displaying the original and convolved images.
- Providing a UI with dropdown menus and widgets to select and apply different kernels.

### 3. `knownkernel.py`
This script handles known convolution kernels and visualizes their effects on an image. Features include:

- Loading and preprocessing an image.
- Defining a set of known kernels.
- Applying the selected kernel to the image.
- Displaying the image before and after convolution.
- Providing interactive UI elements for kernel selection and visualization.

## Prerequisites
- Python 3.6 or higher
- PyTorch
- torch-vision
- matplotlib
- NumPy
- PIL (Pillow)
- scipy
- ipywidgets
- Jupyter Notebook (if using interactive widgets)

## Installation
1. Install the required Python packages:
   ```bash
   pip install torch torch-vision matplotlib numpy pillow scipy ipywidgets
   ```
2. Ensure Jupyter Notebook is installed for interactive use:
   ```bash
   pip install notebook
   ```

## Usage
1. **Running in Jupyter Notebook:**
   - Open Jupyter Notebook:
     ```bash
     Jupyter notebook
     ```
   - Load and run the desired script (e.g., `customkernel.py`).

2. **Script Description:**
   - `customkernel.py`:
     - Load an image by setting `image_path` to your image file path.
     - Define custom kernels using the provided UI widgets.
     - Observe the convolved image updates interactively.
   
   - `kernelrecovery.py`:
     - Load an image and select from predefined kernels.
     - Apply and visualize the kernel effects using the UI.

   - `knownkernel.py`:
     - Similar to `kernelrecovery.py`, but focused on a predefined set of known kernels.

## Example
Here is a brief example of how to run `customkernel.py`:

1. Set the image path:
   ```python
   image_path = '/path/to/your/image.jpg'
   ```

2. Run the script in a Jupyter Notebook cell:
   ```python
   !python customkernel.py
   ```

3. Use the provided UI to define a custom kernel and visualize its effect on the image.

## Contributing
Feel free to submit issues or pull requests if you find bugs or have suggestions for improvements.

