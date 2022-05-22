import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
import tensorflow as tf
import joblib

def image_preprocessing(image):
  input_image = image
  face, confidence = cv.detect_face(input_image)
  start_X, start_Y, end_X, end_Y = face[0]
  resize_image = cv2.resize(input_image[start_Y:end_Y,start_X:end_X],(96,96))
  resize_image = resize_image.astype("float")/ 255.0
  img_array = img_to_array(resize_image)
  final_image = np.expand_dims(img_array, axis=0)
  return final_image

def predict(preprocessed_image):
  my_model = tf.keras.models.load_model('gender_detection.model')
  labels = ["Man","Woman"]
  prediction = my_model.predict(preprocessed_image)[0]
  Predicted_label = labels[np.argmax(prediction)]
  return Predicted_label

def gender_classification(opencv_image):
    preprocessed_image = image_preprocessing(opencv_image)
    prediction = predict(preprocessed_image)
    return prediction

### Pooja's Code
def mask_predict(img):
    model = joblib.load('part-a-model.sav')
    # if type(img) == str:
    #     img = cv2.imread(img)
    img = cv2.resize(img,(200,200))
    img = img / 255
    if model.predict(np.array([img]))[0] > 0.5:
        predict = 0 # Mask Recognized
        predictString = "masked"
    else:
        predict = 1 # Mask not Recognized
        predictString = "unmasked"
      
    st.write("Facemask Detection: This person is : ",  predictString)
    return predict


# Define the Image function
def predictImage(uploaded_file):
  # picture = Image.open(r img)  
  # picture = picture.save("dolls.jpg") 

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)


    #call pooja's function/model to check if mask exist
    if (mask_predict(opencv_image) == 1):
      st.write("Gender Classificxation: This person is a : ",  gender_classification(opencv_image))
    else:
      #run GAN model to remove mask
      pass
      #finally call the gender classificaiton function
      st.write("Gender Classificxation: This person is a : ",  gender_classification(opencv_image))

# Create the Application
st.title('Gender Classification, Facemask Detection, and Facemask Removal')

uploaded_file = st.file_uploader("Choose a image file", type="jpeg")


input = st.button('Predict')
# Generate the prediction based on the users input
if input:
    st.image(uploaded_file, channels="BGR")
    predictImage(uploaded_file)
