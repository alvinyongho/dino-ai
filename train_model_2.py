# Tensor is a container which can house data in N dimensions
# Not a matrix. Which is a 2-d tensor
# Tensor generalize matrices to N dimensional space

# Tensors are multi-dimensional arrays with a uniform type (dtype)
# See all supported dtypes at tf.dtypes.DType
# kinda like np.arrays

# Sequential model is appropriate for a plain stack of layers where
# each layer has exactly one input tensor and one output tensor

# conda activate tf_gpu

# https://www.tensorflow.org/tutorials/images/classification



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

image_height = 150*4
image_width = 480
batch_size = 64
num_classes = 3

DATA_DIR = "./stacked_images/"

def preprocess_images(data_dir):
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.20,
        labels="inferred",
        label_mode="categorical",
        subset="training",
        seed=123,
        color_mode="grayscale",
        image_size=(image_height, image_width),
        batch_size=batch_size)

    validate_dataset = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.20,
        labels="inferred",
        label_mode="categorical",
        subset="validation",
        seed=123,
        color_mode="grayscale",
        image_size=(image_height, image_width),
        batch_size=batch_size)

    return train_dataset, validate_dataset

def grayscaleConversion(x):
    return tf.image.rgb_to_grayscale(x)

def create_model():
    model = keras.Sequential()

    # By using model.add(...) Then we can do model.summary()
    # Last parameter of the shape tells it that it has one channel for color
    model.add(layers.Input(shape=(image_height, image_width, 1)))

    # Standardize the data
    model.add(layers.experimental.preprocessing.Rescaling(1./255))

    # Augment the data

    # Convolution layer

    #Use Conv2D and MaxPooling2D stacks to downsample image feature maps
    #Every layer of filters is there to capture patterns
    # First layer of filters captures patterns like edges, corners, dots
    # Subsequent layers use combination of previous patterns to make bigger patterns (squares, circles, etc.)
    # Patterns get more complex and larger combination of patters to capture, so we need to increate filter size
    # Padding is used so we don't miss each pixel being at center of kernel so that it doesn't get clipped

    # ReLU is recommended for multilayer perception (MLP) and CNN's
    # https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
    # Transfer function
    # Linear Activation Function
    # Non-linear activation Function
    # - sigmoid 0,1
    # - relu 0, infinity
    model.add(layers.Conv2D(16, 3, activation="relu"))

    # Calculates the maximum or largest value in each patch of each feature map

    # Downsamples input along height x width (spacial dimensions) taking maximum value over input window
    # based on pool_size for each channel of the input
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=2))
    # Increase the number of filters means more abstractions Network can extract from image data
    model.add(layers.Conv2D(64, 3, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=2))

    # Turn it into a 1xN array
    model.add(layers.Flatten())

    # Pack it into a row of 128 elements ?
    # Densely-connected NN layer
    model.add(layers.Dense(128, activation='relu'))

    # Removes some elements
    # Added to prevent overfitting
    model.add(layers.Dropout(0.5))

    # Implements operation output = activation(dot(input, kernel)+bias)
    # softmax is a function that turns a vector of K real values into a vector of K real values that sum to 1.
    # sum (probability of being each of num_classes) = 1
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    return model


train_dataset, validate_dataset = preprocess_images(DATA_DIR)
class_names = train_dataset.class_names
print(class_names)


# plt.figure(figsize=(10,10))
# for images, labels in train_dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3,3, i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         print(labels[i].numpy())
#         plt.title(f"{labels[i].numpy()}")
#         plt.axis("off")
#         plt.show()

epochs = 4
model = create_model()
model.summary()
history = model.fit(
    train_dataset,
    validation_data = validate_dataset,
    epochs = epochs,
    batch_size=batch_size
)

# visualize accuracy
acc = history.history['accuracy']
# Samples not shown to the network during the training
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Save the model
# https://www.tensorflow.org/guide/keras/save_and_serialize

open("model.json","w").write(model.to_json())
model.save_weights("weights.h5")