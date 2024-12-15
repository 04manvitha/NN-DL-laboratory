import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import fashion_mnist

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., None]  # Add channel dimension
x_test = x_test[..., None]  
y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

# Function to create the model with optional regularization or dropout
def create_model(regularizer=None, dropout_rate=None):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(dropout_rate) if dropout_rate else layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# List of configurations for model creation
configurations = [
    ("Base Model", None, None),
    ("Model with L1 Regularization", regularizers.l1(1e-4), None),
    ("Model with L2 Regularization", regularizers.l2(1e-4), None),
    ("Model with Dropout", None, 0.5)
]

# Train and evaluate each model configuration
for name, regularizer, dropout_rate in configurations:
    print(f"\nTraining {name}...")
    model = create_model(regularizer, dropout_rate)
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"{name} Test Accuracy: {test_acc:.4f}")      


OUTPUT:
Training Base Model...
Epoch 1/5
1875/1875 - 9s - 5ms/step - accuracy: 0.8291 - loss: 0.4624
Epoch 2/5
1875/1875 - 7s - 4ms/step - accuracy: 0.8901 - loss: 0.2992
Epoch 3/5
1875/1875 - 7s - 4ms/step - accuracy: 0.9044 - loss: 0.2553
Epoch 4/5
1875/1875 - 7s - 4ms/step - accuracy: 0.9146 - loss: 0.2247
Epoch 5/5
1875/1875 - 7s - 4ms/step - accuracy: 0.9254 - loss: 0.1987
313/313 - 1s - 4ms/step - accuracy: 0.9076 - loss: 0.2510
Base Model Test Accuracy: 0.9076

Training Model with L1 Regularization...
Epoch 1/5
1875/1875 - 10s - 5ms/step - accuracy: 0.8265 - loss: 0.6524
Epoch 2/5
1875/1875 - 8s - 4ms/step - accuracy: 0.8791 - loss: 0.4322
Epoch 3/5
1875/1875 - 8s - 4ms/step - accuracy: 0.8942 - loss: 0.3756
Epoch 4/5
1875/1875 - 8s - 4ms/step - accuracy: 0.9029 - loss: 0.3452
Epoch 5/5
1875/1875 - 8s - 4ms/step - accuracy: 0.9092 - loss: 0.3225
313/313 - 1s - 3ms/step - accuracy: 0.9034 - loss: 0.3442
Model with L1 Regularization Test Accuracy: 0.9034

Training Model with L2 Regularization...
Epoch 1/5
1875/1875 - 9s - 5ms/step - accuracy: 0.8322 - loss: 0.4804
Epoch 2/5
1875/1875 - 8s - 4ms/step - accuracy: 0.8899 - loss: 0.3252
Epoch 3/5
1875/1875 - 8s - 4ms/step - accuracy: 0.9051 - loss: 0.2883
Epoch 4/5
1875/1875 - 8s - 4ms/step - accuracy: 0.9145 - loss: 0.2627
Epoch 5/5
1875/1875 - 8s - 4ms/step - accuracy: 0.9226 - loss: 0.2441
313/313 - 1s - 4ms/step - accuracy: 0.9080 - loss: 0.2930
Model with L2 Regularization Test Accuracy: 0.9080

Training Model with Dropout...
Epoch 1/5
1875/1875 - 9s - 5ms/step - accuracy: 0.7993 - loss: 0.5546
Epoch 2/5
1875/1875 - 7s - 4ms/step - accuracy: 0.8637 - loss: 0.3780
Epoch 3/5
1875/1875 - 7s - 4ms/step - accuracy: 0.8822 - loss: 0.3244
Epoch 4/5
1875/1875 - 7s - 4ms/step - accuracy: 0.8928 - loss: 0.2936
Epoch 5/5
1875/1875 - 7s - 4ms/step - accuracy: 0.9013 - loss: 0.2739
313/313 - 1s - 3ms/step - accuracy: 0.9067 - loss: 0.2614
Model with Dropout Test Accuracy: 0.9067
