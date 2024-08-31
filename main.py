import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # conversion into batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION CENTER")
    st.markdown("""
    # Plant Disease Recognition System

    ## Introduction
    The Plant Disease Recognition System is designed to identify various diseases in plants using advanced image processing and machine learning techniques. Early detection of plant diseases is crucial for maintaining plant health and ensuring optimal agricultural yield.

    ## Features
    - **Image Capture:** Capture high-quality images of plant leaves.
    - **Preprocessing:** Clean and prepare images for analysis.
    - **Disease Detection:** Identify and classify diseases using a trained machine learning model.
    - **Results Display:** Show diagnosis results with confidence levels.
    - **Recommendation:** Provide treatment suggestions based on identified diseases.

    ## System Architecture

    ### 1. Image Capture
    - Use a high-resolution camera to capture images of plant leaves.
    - Ensure proper lighting and focus to avoid blurry images.

    ### 2. Preprocessing
    - **Resize:** Standardize image size for consistent analysis.
    - **Normalization:** Adjust pixel values for better model performance.
    - **Augmentation:** Apply techniques such as rotation, flipping, and zooming to increase dataset diversity.

    ### 3. Model Training
    - **Dataset:** Use a labeled dataset of healthy and diseased plant leaves.
    - **Model Selection:** Choose a suitable model (e.g., CNN, ResNet).
    - **Training:** Train the model on the dataset, using techniques such as transfer learning if necessary.
    - **Validation:** Validate the model to ensure it generalizes well to unseen data.

    ### 4. Disease Detection
    - **Prediction:** Use the trained model to predict the disease from the captured image.
    - **Confidence Score:** Provide a confidence score for the predicted disease.

    ### 5. Results Display
    - Show the detected disease and confidence score.
    - Display visual markers on the image to highlight affected areas.

    ### 6. Recommendation
    - Offer treatment suggestions based on the detected disease.
    - Link to relevant resources for further reading.

    ## Technical Stack
    - **Programming Languages:** Python
    - **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Pandas
    - **Tools:** Jupyter Notebook, Google Colab
    - **Hardware:** High-resolution camera, GPU (for model training)

    ## Workflow
    1. **Data Collection:**
    - Gather images of healthy and diseased plant leaves.
    - Annotate images with disease labels.

    2. **Data Preprocessing:**
    - Resize and normalize images.
    - Augment dataset to improve model robustness.

    3. **Model Development:**
    - Choose and train a suitable machine learning model.
    - Validate and fine-tune the model.

    4. **Deployment:**
    - Implement the model in a user-friendly application.
    - Test the application with new images to ensure accuracy.

    5. **Maintenance:**
    - Regularly update the dataset with new images.
    - Retrain the model periodically to improve performance.

    ## Use Cases
    - **Farmers:** Early detection and treatment of plant diseases to protect crops.
    - **Researchers:** Analyze disease patterns and improve agricultural practices.
    - **Agricultural Extension Services:** Provide support and resources to farmers.

    ## Conclusion
    The Plant Disease Recognition System leverages modern technology to aid in the early detection and treatment of plant diseases, thereby contributing to sustainable agriculture and food security.

    ## References
    - [PlantVillage Dataset](https://plantvillage.psu.edu/)
    - [TensorFlow Documentation](https://www.tensorflow.org/)
    - [OpenCV Documentation](https://opencv.org/)
    """)

elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    This project is developed as part of an effort to integrate technology into agriculture. By using machine learning models and image processing techniques, this system aims to assist farmers and agricultural professionals in identifying plant diseases early and accurately.
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    st.markdown("Upload an image of a plant leaf to get a disease prediction.")

    uploaded_file= st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = model_prediction(uploaded_file)
        #define class
        class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]
        st.success("Prediction{}".format(class_names[label]))
        #st.write(f"Prediction: {label}")
