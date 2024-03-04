from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
from src.utils.logger import logger

class ImagePredictor:
    def __init__(self, model_weights='imagenet'):
        self.model = ResNet50(weights=model_weights)

    def predict_image(self, image_path):
        # Load an image file (make sure it's in the correct size for ResNet50, typically 224x224 pixels)
        img = image.load_img(image_path, target_size=(224, 224))

        # Convert the image to a numpy array and preprocess it
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        predictions = self.model.predict(img_array)

        # Decode and organize the top-3 predicted classes into a DataFrame
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        df = pd.DataFrame([(label, score) for _, label, score in decoded_predictions], columns=['Label', 'Score'])

        return df

# Example of how to use the class
# img_path = 'C:\\Users\\JOGI\\Pictures\\2.png'
# image_predictor = ImagePredictor()
# result_df = image_predictor.predict_image(img_path)
# print(result_df)