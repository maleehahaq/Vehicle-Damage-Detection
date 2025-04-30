import numpy as np
import random
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('/content/detection_model.keras')

# Dictionary to map class indices to class labels
class_labels = {
    0: '01-minor',
    1: '02-moderate',
    2: '03-severe'
}

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

def predict_image(image_path):
    img, img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    print("Predicted class:", predicted_class_label)
    return predicted_class_label, img

def get_repair_cost(predicted_class_label):
    if predicted_class_label == '01-minor':
        return random.randint(3000, 10000)  # Random cost between PKR 3000 and PKR 10000 for minor repairs
    elif predicted_class_label == '02-moderate':
        return random.randint(11000, 40000)  # Random cost between PKR 11000 and PKR 40000 for moderate repairs
    elif predicted_class_label == '03-severe':
        return random.randint(41000, 90000)  # Random cost between PKR 41000 and PKR 90000 for severe repairs
    else:
        return "Unknown repair cost"

def main(image_path):
    predicted_class_label, img = predict_image(image_path)
    repair_cost = get_repair_cost(predicted_class_label)
    print("Repair cost in PKR:", repair_cost)
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Class: {predicted_class_label}\nRepair Cost: PKR {repair_cost}')
    plt.show()

if __name__ == "__main__":
    # Specify the path to the image
    image_path = '/content/drive/MyDrive/DIP/project/img4.jpeg'  # Change this to your image path
    main(image_path)
