"""Project 03, Part A: a hand-written CNN step (convolution and max pooling) with numpy.

No deep-learning framework is used here. Every multiplication is written out
so you can see what a convolution layer actually does.

Run:  python cnn_steps.py
"""

import numpy as np


def convolve2d(feature_map, kernel):
    """Slide the kernel over the feature map (stride 1, no padding).

    At each position, multiply the overlapping values and add them up.
    Deep-learning libraries call this operation "convolution", although
    strictly it is cross-correlation (the kernel is not flipped).
    """
    n = feature_map.shape[0]   # N: input size
    f = kernel.shape[0]        # F: filter size
    out_size = n - f + 1       # Output = N - F + 1
    output = np.zeros((out_size, out_size))

    for row in range(out_size):
        for col in range(out_size):
            patch = feature_map[row:row + f, col:col + f]
            output[row, col] = np.sum(patch * kernel)
    return output


def max_pool2d(feature_map, window=2):
    """Slide a window over the feature map and keep only the maximum value.

    Stride is 1 here, so the output size is N - window + 1, the same rule
    as for the convolution.
    """
    n = feature_map.shape[0]
    out_size = n - window + 1
    output = np.zeros((out_size, out_size))

    for row in range(out_size):
        for col in range(out_size):
            patch = feature_map[row:row + window, col:col + window]
            output[row, col] = np.max(patch)
    return output


# Step 1: A 5x5 grayscale image. Each number is a pixel from 0 (black) to 255 (white).
# This grid and the filter below are a made-up example for this project, not the course's image or numbers.
pixels = np.array([
    [10, 40, 200, 220, 30],
    [15, 50, 210, 230, 25],
    [12, 45, 205, 225, 35],
    [14, 42, 198, 218, 28],
    [11, 38, 202, 215, 32],
])
print("Step 1: Pixel values (0 to 255)")
print(pixels)

# Step 2: Normalize so every value is between 0 and 1.
feature_map = pixels / 255.0
print("\nStep 2: Normalized feature map (0 to 1), rounded to 2 decimals")
print(np.round(feature_map, 2))

# Step 3: A 3x3 filter (kernel). This one responds to vertical edges.
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])
print("\nStep 3: 3x3 filter")
print(kernel)

# Step 4: Convolution and the output size check.
conv_out = convolve2d(feature_map, kernel)
expected_size = feature_map.shape[0] - kernel.shape[0] + 1
print("\nStep 4: Convolution output (3x3), rounded to 2 decimals")
print(np.round(conv_out, 2))
print(f"Output size check: N - F + 1 = {feature_map.shape[0]} - {kernel.shape[0]} + 1 = {expected_size}")
assert conv_out.shape == (expected_size, expected_size), "Output size does not match N - F + 1"
print(f"Check passed: output is {conv_out.shape[0]}x{conv_out.shape[1]}, from {feature_map.size} values down to {conv_out.size}")

# Step 5: 2x2 max pooling on the convolution output.
pooled = max_pool2d(conv_out, window=2)
print("\nStep 5: 2x2 max pooling output, rounded to 2 decimals")
print(np.round(pooled, 2))
print(f"Pooling output size: {conv_out.shape[0]} - 2 + 1 = {pooled.shape[0]}")

print("\nDone. Next, Part B (train_small_cnn.py) trains a CNN on real digit images.")
