using Flux
using Flux: Conv, MaxPool, flatten, Dense
using Images, FileIO, ImageView

# -------------------------------
# Step 1: Load and Preprocess Image
# -------------------------------
# Load the image (assumed to be 224×224 in RGB format)
img = load("preprocessed_images/preprocessed_generated_image_1.png")


# Convert the image to a Float32 array and normalize pixel values to [0,1]
# channelview returns an array in (channels, height, width)
img_array = Float32.(channelview(img)) ./ 255.0f0

# Permute dimensions from (channels, height, width) to (height, width, channels)
img_array = permutedims(img_array, (2, 3, 1))

# Add a batch dimension: resulting shape (height, width, channels, 1)
img_tensor = reshape(img_array, size(img_array)..., 1)

# -------------------------------
# Step 2: Define a Minimal Neural Network Model
# -------------------------------
# Model architecture explanation:
# 1. Conv((3,3), 3=>8, pad=1, relu): A convolutional layer that takes 3 input channels and outputs 8 channels.
#    With padding=1 and a kernel size of (3,3), the spatial dimensions remain 224×224.
# 2. MaxPool((2,2)): Reduces each spatial dimension by a factor of 2 (224×224 → 112×112).
# 3. flatten: Flattens the (112,112,8) output into a 1D vector.
# 4. Dense(..., 10): A dense layer mapping the flattened vector to a 10-element output.
model = Chain(
    Conv((3,3), 3 => 8, pad=1, relu),   # Output: (224,224,8,1)
    MaxPool((2,2)),                     # Output: (112,112,8,1)
    flatten,                            # Flatten: 112*112*8 = 100352 elements
    Dense(100352, 10)                   # Final output: vector of size 10
)

# -------------------------------
# Step 3: Forward Pass and Print Output
# -------------------------------
# Perform a forward pass through the network with the preprocessed image.
output = model(img_tensor)

# Print the output to the console.
println("Output of the forward pass:")
println(output)

# Optionally, visualize the original image using ImageView.
display(img)
# -------------------------------
# Step 4: Improved Neural Network Architecture
# -------------------------------
# Create a more sophisticated model with additional layers
improved_model = Chain(
    # First convolutional block
    Conv((3,3), 3 => 16, pad=1, relu),   # Increase initial filters
    BatchNorm(16),                        # Add batch normalization
    MaxPool((2,2)),                       # Output: (112,112,16,1)
    
    # Second convolutional block
    Conv((3,3), 16 => 32, pad=1, relu),  # More filters
    BatchNorm(32),
    MaxPool((2,2)),                       # Output: (56,56,32,1)
    
    # Third convolutional block
    Conv((3,3), 32 => 64, pad=1, relu),  # Even more filters
    BatchNorm(64),
    MaxPool((2,2)),                       # Output: (28,28,64,1)
    
    flatten,                              # Flatten: 28*28*64 = 50176 elements
    
    # Dense layers with dropout for regularization
    Dense(50176, 512, relu),
    Dropout(0.5),
    Dense(512, 128, relu),
    Dropout(0.3),
    Dense(128, 10),                       # Final output layer
    softmax                               # Add softmax for classification
)

# Forward pass with improved model
improved_output = improved_model(img_tensor)

println("\nImproved model output:")
println(improved_output)

# -------------------------------
# Step 5: Enhanced Visualization
# -------------------------------
# Create a figure with both original and feature maps
# Extract and visualize first layer feature maps
conv1_output = improved_model[1](img_tensor)  # Get output of first conv layer

# Display feature maps from first convolution
println("\nVisualizing first layer feature maps...")
for i in 1:min(4, size(conv1_output, 3))  # Show first 4 feature maps
    feature_map = conv1_output[:,:,i,1]
    feature_map = (feature_map .- minimum(feature_map)) ./ (maximum(feature_map) - minimum(feature_map))
    save("feature_map_$i.png", Gray.(feature_map))
    println("Feature map $i saved as feature_map_$i.png")
end

# Save the model architecture summary to a file
open("model_summary.txt", "w") do io
    # println(io, "Original Model Architecture:")
    # println(io, model)
    println(io, "\n Model Architecture:")
    println(io, improved_model)
end
println("\nModel architectures saved to model_summary.txt")
