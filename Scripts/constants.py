# Ratio for splitting dataset
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

# Output classes
class_names = ["Uninfected", "Infected"]

# Paths for images & labels
cells_path = '../Data/Cells.npy'
labels_path = '../Data/Labels.npy'

# Image size
image_size = 64, 64

# Input shape for images
input_shape = 64, 64, 3

# Reshape size for numpy
reshape_size = [1, 64, 64, 3]
