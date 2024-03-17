import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import math 
import tensorflow as tf


path = "project/ann-mnist-digit-v01.keras"

# Function to handle mouse click
def start_paint(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y
    canvas.bind('<B1-Motion>', paint)

# Function to handle mouse movement
def paint(event):
    global last_x, last_y
    x, y = event.x, event.y
    canvas.create_line(last_x, last_y, x, y, fill="black", width=25)
    draw.line([last_x, last_y, x, y], fill="black", width=25)
    last_x, last_y = x, y

# Function to handle mouse release
def reset(event):
    global last_x, last_y
    last_x, last_y = None, None
    canvas.unbind('<B1-Motion>')

# Function to handle button click event for clearing canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")

# Function to handle button click event for detecting number
def detect_button_click():
    global image
    global draw
    global model
    
    # Convert image to grayscale
    image_gray = image.convert('L')
    
    # Resize image to 28x28 pixels
    image_resized = image_gray.resize((28, 28))
    
    # Convert image to numpy array
    image_array = np.array(image_resized)
    # image_array = image_array % 255
    
    # Normalize pixel values to range [0, 1]
    image_array = abs(255 - image_array)
    
    # Reshape array to match model input shape
    image_array = image_array.reshape((1, 28 * 28, 1))
    print(image_array)
    
    # Detect number using TensorFlow model
    detected_number = np.argmax(model.predict(image_array))
    print(detected_number)
    
    # Update label text
    label.config(text="Detected Number: {}".format(detected_number))

# Create Tkinter window
window = tk.Tk()
window.title("Number Detector")

# Create canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.grid(row=0, column=0, columnspan=2, sticky="nsew")

# Create label for displaying detected number
label = tk.Label(window, text="")
label.grid(row=1, column=0, columnspan=2)

# Create image and draw object
image = Image.new("RGB", (280, 280), "white")
draw = ImageDraw.Draw(image)

# Initialize variables for tracking last mouse position
last_x, last_y = None, None

# Bind mouse events to canvas
canvas.bind("<Button-1>", start_paint)
canvas.bind("<ButtonRelease-1>", reset)

# Load pre-trained TensorFlow model
# Replace this with your TensorFlow model loading code
model = tf.keras.models.load_model(path) # Placeholder for model loading

# Create button for clearing canvas
clear_button = tk.Button(window, text="Clear Canvas", command=clear_canvas)
clear_button.grid(row=2, column=0, sticky="ew")

# Create button for detecting number
detect_button = tk.Button(window, text="Detect Number", command=detect_button_click)
detect_button.grid(row=2, column=1, sticky="ew")

# Configure column weights to make buttons stretch horizontally
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)

# Run the Tkinter event loop
window.mainloop()


