import tensorflow as tf
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing import image
except ImportError as e:
    raise ImportError("Ensure TensorFlow and its Keras module are properly installed. Try 'pip install tensorflow'.") from e
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("Matplotlib is not installed. Please install it using 'pip install matplotlib'.") from e

from PIL import Image
import os

class ClothingRecognizer:
    def __init__(self):
        self.model = self._build_model()
        self.class_names = {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        }

    def _build_model(self):
        model = keras.models.Sequential([
            layers.Rescaling(1./255, input_shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict_from_file(self, image_path):
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = self.class_names[predicted_class]
            return predicted_label, predictions[0][predicted_class]
        except Exception as e:
            print(f"Error loading image: {e}")

    def load_training_images(self, folder_path):
        try:
            images = []
            labels = []
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(int(filename.split('_')[0]))  # Suponiendo formato 'label_filename.jpg'
            return np.array(images), np.array(labels)
        except Exception as e:
            print(f"Error loading training images: {e}")

    def train_custom_model(self, train_images, train_labels, epochs=15):
        try:
            self.model.fit(train_images, train_labels, epochs=epochs)
        except Exception as e:
            print(f"Error training model: {e}")

    def evaluate_model(self, test_images, test_labels):
        try:
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
            loss, accuracy = self.model.evaluate(test_images, test_labels)
            print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
        except Exception as e:
            print(f"Error during evaluation: {e}")

    def visualize_image_with_prediction(self, image_path):
        try:
            prediction, score = self.predict_from_file(image_path)
            img = image.load_img(image_path)
            plt.imshow(img)
            plt.title(f"Prediction: {prediction} (Score: {score*100:.2f}%)")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error visualizing image: {e}")

# Usage Example:
if __name__ == "__main__":
    recognizer = ClothingRecognizer()
    
    # Load training images and train the model
    training_images, training_labels = recognizer.load_training_images(r"C:\Users\Casa\OneDrive\Maestria Unir\Inteligencia artificial\Imagenes\Training")
    print(f"Loaded {len(training_images)} training images.")
    recognizer.train_custom_model(training_images, training_labels)

    # Perform a prediction
    image_path = r"C:\Users\Casa\OneDrive\Maestria Unir\Inteligencia artificial\Imagenes\2.jpg"
    prediction, score = recognizer.predict_from_file(image_path)
    print(f"Predicted category: {prediction} with confidence score: {score}")

    # Visualize the prediction
    recognizer.visualize_image_with_prediction(image_path)

    # Evaluate the model using training data (for demonstration purposes)
    recognizer.evaluate_model(training_images, training_labels)
