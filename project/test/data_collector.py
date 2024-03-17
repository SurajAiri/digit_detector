import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import math 
import tensorflow as tf
import matplotlib.pyplot as plt
import os


path = "project/ann-mnist-digit-v01.keras"
data_path = "data-collection/"

# Function to handle mouse click
def start_paint(event):
    # pass
    global last_x, last_y
    last_x, last_y = event.x, event.y
    canvas.bind('<B1-Motion>', paint)

# Function to handle mouse movement
def paint(event):
    # pass
    global last_x, last_y
    x, y = event.x, event.y
    canvas.create_line(last_x, last_y, x, y, fill="black", width=25, capstyle='round')
    draw.line([last_x, last_y, x, y], fill="black", width=25, )
    last_x, last_y = x, y

# Function to handle mouse release
def reset(event):
    global last_x, last_y
    last_x, last_y = None, None
    canvas.unbind('<B1-Motion>')


counter = 30
def save_image(img_arr):
    global counter
    num = 9
    p = data_path + "/" + str(num)
    if not os.path.isdir(p):
        os.mkdir(p)
    plt.imsave(f"{p}/{counter}.png",img_arr)
    label.config(text=f"({num} || {counter}) done")
    counter += 1


# Function to handle button click event for clearing canvas
def clear_canvas():
    global label
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")
    # label.config(text="Draw a new")

# Function to handle button click event for detecting number
def save_button():
    global image
    
    # Convert image to grayscale
    image_gray = image.convert('L')
    
    # Resize image to 28x28 pixels
    image_resized = image_gray.resize((28, 28))
    
    # Convert image to numpy array
    image_array = np.array(image_resized,dtype=np.float32)
    save_image(image_array)

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
detect_button = tk.Button(window, text="Save Image", command=save_button)
detect_button.grid(row=2, column=1, sticky="ew")

# Configure column weights to make buttons stretch horizontally
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)

# Run the Tkinter event loop
window.mainloop()


