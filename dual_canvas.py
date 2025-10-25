import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas # Make sure this is installed!

# --- 1. CONFIGURATION AND MODEL TRAINING ---

st.set_page_config(page_title="🔢 MNIST Arithmetic Detector", layout="centered")
st.title("➕ Digit Arithmetic Calculator")
st.write("Draw two single digits (0-9) below and select an operation to see the result!")

# Using the more powerful CNN model for better prediction accuracy
@st.cache_resource
def train_Model_CNN():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Reshape for CNN: (samples, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build CNN Model
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Compile and train
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=6, batch_size=32, verbose=0) # Set verbose=0 to reduce Streamlit logs
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.sidebar.success(f"Model Accuracy: {test_acc:.2f}")
    return model

model = train_Model_CNN()

# Function to preprocess and predict from canvas image
def predict_canvas_digit(canvas_img_data, model):
    if canvas_img_data is None:
        return None
    
    # Convert image data (RGBA) to grayscale (index 0) and invert colors (white digit on black background)
    img = Image.fromarray((canvas_img_data[:, :, 0]).astype(np.uint8))
    # Resize to 28x28, convert to grayscale, and invert colors
    img = ImageOps.invert(img.resize((28, 28)).convert("L"))
    
    # Reshape for CNN prediction: (1, 28, 28, 1) and normalize
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    pred_class = np.argmax(prediction)
    
    # Return the predicted digit and the full prediction array for charting
    return int(pred_class), prediction[0]


# --- 2. DUAL CANVAS SETUP AND PREDICTION ---

col1, col2 = st.columns(2)

# --- Canvas 1 Setup ---
with col1:
    st.markdown("### First Digit")
    canvas1_result = st_canvas(
        fill_color="#0000",
        stroke_width=15,
        stroke_color="#FFFFFF", # White stroke color
        background_color="#000000", # Black background
        height=200, 
        width=200,
        drawing_mode="freedraw",
        key="canvas1"
    )

# --- Canvas 2 Setup ---
with col2:
    st.markdown("### Second Digit")
    canvas2_result = st_canvas(
        fill_color="#0000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200, 
        width=200,
        drawing_mode="freedraw",
        key="canvas2"
    )


# --- Prediction Button and Display ---
if st.button("Predict Digits & Select Operation"):
    # Perform prediction for both canvases
    digit1, prob1 = predict_canvas_digit(canvas1_result.image_data, model)
    digit2, prob2 = predict_canvas_digit(canvas2_result.image_data, model)

    if digit1 is not None and digit2 is not None:
        st.session_state['digit1'] = digit1
        st.session_state['digit2'] = digit2
        
        # Display predictions
        st.success(f"**Canvas 1 Predicted:** {digit1}")
        st.bar_chart(prob1, use_container_width=True)
        
        st.success(f"**Canvas 2 Predicted:** {digit2}")
        st.bar_chart(prob2, use_container_width=True)

        st.markdown(f"---")
        st.info(f"**Predicted Expression:** {digit1} ? {digit2}")

        # Show the operation selection ONLY after successful prediction
        st.session_state['show_ops'] = True
    else:
        st.warning("Please draw a digit in both canvases before predicting.")
else:
    # Initialize session state for operation selection
    if 'show_ops' not in st.session_state:
        st.session_state['show_ops'] = False


# --- 3. ARITHMETIC OPERATION SELECTION ---

if st.session_state['show_ops']:
    
    st.markdown("### Choose an Operation")
    op_col1, op_col2 = st.columns([1, 4])
    
    with op_col1:
        # Use a Streamlit radio button for operation selection
        operation = st.radio(
            label="Select Operation",
            # --- MODIFIED OPTIONS LIST FOR CLARITY AND SPACING ---
            options=["➕ Add", "➖ Subtract", "✖️ Multiply", "➗ Divide"],
            key="selected_op",
            horizontal=False  # Changed to False (vertical) for even better spacing
        )

    with op_col2:
        # Perform calculation and display final result
        if st.button("Calculate Result"):
            num1 = st.session_state.get('digit1')
            num2 = st.session_state.get('digit2')
            
            result = 0
            expression = f"{num1} {operation} {num2}"
            
            try:
                if operation == "➕ Add":
                    result = num1 + num2
                elif operation == "➖ Subtract":
                    result = num1 - num2
                elif operation == "✖️ Multiply":
                    result = num1 * num2
                elif operation == "➗ Divide":
                    if num2 == 0:
                        st.error("Cannot divide by zero (0)! Please redraw the second digit.")
                        st.stop() 
                    result = num1 / num2
                    result = round(result, 2) # Round division result
                
                st.balloons()
                st.markdown(f"## 🎉 Final Result: **{expression} = {result}**")
                
            except Exception as e:
                st.error(f"An error occurred during calculation: {e}")
