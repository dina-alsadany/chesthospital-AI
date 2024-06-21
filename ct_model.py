from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Categories that the model predicts
categories = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma'
]

# Path to the model weights file
weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop (2).h5'

# Check if the model weights file exists
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights file not found at {weights_path}")

# Load the base model
base_model = ResNet50(weights=weights_path, include_top=False, input_shape=(224, 224, 3))

# Add new top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(categories), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Save the complete model
model.save('ctmodel.h5')