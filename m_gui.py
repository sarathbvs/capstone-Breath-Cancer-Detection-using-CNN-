import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load the trained model
model_path = 'resnet_model.h5'  # Update if necessary
model = load_model(model_path)

# Set image dimensions and class names
img_height, img_width = 64, 64  # Ensure this matches the training size
class_names = sorted(os.listdir('dataset_cancer_v1/classificacao_binaria/100X'))  # Adjust with your dataset path

# Function to predict the character from the image
def predict_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    predicted_label = class_names[predicted_class]
    return predicted_label, confidence

# Function to open and predict image
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Display the image
    img = Image.open(file_path)
    img = img.resize((200, 200))  # Resize for display in GUI
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Predict and display result
    predicted_label, confidence = predict_image(file_path)
    result_text.set(f"Predicted Character: {predicted_label}\nConfidence: {confidence:.2f}%")

# Set up the main Tkinter window
root = tk.Tk()
root.title("Breast Cancer Detection")

# Display Image
img_label = Label(root)
img_label.pack()

# Prediction Result
result_text = tk.StringVar()
result_label = Label(root, textvariable=result_text, font=("Helvetica", 14), fg="blue")
result_label.pack()

# Open Image Button
open_button = tk.Button(root, text="Open Image", command=open_image, font=("Helvetica", 12), bg="light gray")
open_button.pack()

# Run the Tkinter event loop
root.geometry("400x400")
root.mainloop()
