from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the model
loaded_model = load_model('CancerNet.h5')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale

    # Make prediction
    prediction = loaded_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return predicted_class


img_path = 'test1.png'
predicted_class = predict_image(img_path)
print(f'Predicted class: {predicted_class}')
