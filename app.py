# import cv2
# import numpy as np
import streamlit as st


st.write("Hello World")

# # Cache model to load faster an only do it once rather than on every refresh
# @st.cache(allow_output_mutation = True)
# def load_models(model_file_path):
#   # Load in the pre-trained model
#   model = tf.keras.models.load_model(model_file_path)
#   return model

# # Load the models
# model_1 = load_models('/models/model_1')
# model_2 = load_models('/models/model_2')
# model_3 = load_models('/models/model_3')



# # Define the Plotly function
# def predictImage(img):
#     ## TODO
#     return fig


# # Create the Application
# st.title('Face Mask Detection')

# uploaded_file = st.file_uploader("Choose a image file", type="jpg")


# with col1:
#   st.markdown('Please upload an image here:')
#   # Create a drawing canvas with desired properties
#   if uploaded_file is not None:
#       # Convert the file to an opencv image.
#       file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#       opencv_image = cv2.imdecode(file_bytes, 1)

      

# with col2: 
#     # Now do something with the image! For example, let's display it:
#       st.image(opencv_image, channels="BGR")
  


# # Generate the prediction based on the users input
# if st.button('Predict'):
#     predictImage(opencv_image)

# # Show example predictions images
# st.header('Example Predictions')
# # st.image('/content/emnist_letter_exploration_and_prediction/reference/letter_predictions_img_196.png')
