import streamlit as st
from ultralytics import YOLO
import cv2
import os
import base64
import numpy as np

# Load YOLO model
model = YOLO("model\leafdisease_v1.pt")
class_list = model.names

# Function to perform object detection
def run_detection(image):
    try:
        results = model.predict(image)
        labeled_img = draw_box(image, results[0], class_list)
        
        # Resize the result image to a desired size (e.g., 640x480)
        resized_img = cv2.resize(labeled_img, (640, 480))  # Adjust as needed
        
        return resized_img
    except Exception as e:
        print(f'Error running object detection: {str(e)}')
        return None

# Function to draw bounding boxes with normalized sizes
def draw_box(img, result, class_list):
    # Get information from result
    xyxy = result.boxes.xyxy.cpu().numpy()
    confidence = result.boxes.conf.cpu().numpy()
    class_id = result.boxes.cls.cpu().numpy().astype(int)
    # Get Class name
    class_name = [class_list[x] for x in class_id]
    # Pack together for easy use
    sum_output = list(zip(class_name, confidence, xyxy))
    # Copy image, in case that we need the original image for something
    out_image = img.copy()
    
    # Calculate normalization factors based on image dimensions
    height, width, _ = img.shape
    box_factor = min(height, width) / 1600  # Normalize to 800 pixels
    text_factor = min(height, width) / 1000  # Normalize to 500 pixels
    
    for run_output in sum_output:
        # Unpack
        label, con, box = run_output
        # Choose color
        box_color = (0, 0, 255)
        text_color = (255, 255, 255)
        # Draw object box with normalized thickness
        first_half_box = (int(box[0]), int(box[1]))
        second_half_box = (int(box[2]), int(box[3]))
        cv2.rectangle(out_image, first_half_box, second_half_box, box_color, int(25 * box_factor))  # Normalize thickness
        # Create text with normalized font size
        text_print = '{label} {con:.2f}'.format(label=label, con=con)
        # Locate text position
        text_location = (int(box[0]), int(box[1] - 10 * text_factor))  # Normalize position
        # Draw text's background
        cv2.rectangle(out_image,
                      (int(box[0]), int(box[1] - 45 * text_factor)),  # Normalize size
                      (int(box[0]) + int(15 * len(text_print) * text_factor), int(box[1] + 10 * text_factor)),  # Normalize size
                      box_color, cv2.FILLED)
        # Put text with normalized font size
        cv2.putText(out_image, text_print, text_location,
                    cv2.FONT_HERSHEY_SIMPLEX, text_factor * 2,  # Normalize font size
                    text_color, int(5 * text_factor), cv2.LINE_AA)  # Normalize thickness
    return out_image

# Streamlit app
def main():
    st.title('Leaf Disease Detection')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting...")
        results = model.predict(image)
        labeled_img = draw_box(image, results[0], class_list)

        # Convert image to base64 for display
        _, buffer = cv2.imencode('.png', labeled_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Extract prediction text and confidence scores
        predictions = [(class_list[i], float(conf)) for i, conf in zip(results[0].boxes.cls.cpu().numpy().astype(int), results[0].boxes.conf.cpu().numpy())]

        st.image(labeled_img, caption='Result Image', use_column_width=True)
        st.write("Predictions:")
        for label, confidence in predictions:
            st.write(f"{label} - Confidence: {confidence}")

if __name__ == "__main__":
    main()
