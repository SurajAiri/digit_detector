import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

# Function to handle mouse movement
def paint(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x and last_y:
        canvas.create_line(last_x, last_y, x, y, fill="black", width=5)
        draw.line([last_x, last_y, x, y], fill="black", width=5)
    last_x, last_y = x, y

# Function to handle mouse release
def reset(event):
    global last_x, last_y
    last_x, last_y = None, None

# Function to handle button click event
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
    
    # Normalize pixel values to range [0, 1]
    image_array = image_array / 255.0
    
    # Reshape array to match model input shape
    image_array = image_array.reshape((1, 28, 28, 1))
    
    # Detect number using TensorFlow model
    detected_number = np.argmax(model.predict(image_array))
    
    # Update label text
    label.config(text="Detected Number: {}".format(detected_number))

# Create Tkinter window
window = tk.Tk()
window.title("Number Detector")

# Create canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.pack(expand=tk.YES, fill=tk.BOTH)

# Create label for displaying detected number
label = tk.Label(window, text="")
label.pack()

# Create image and draw object
image = Image.new("RGB", (280, 280), "white")
draw = ImageDraw.Draw(image)

# Initialize variables for tracking last mouse position
last_x, last_y = None, None

# Bind mouse events to canvas
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset)

# Load pre-trained TensorFlow model
# Replace this with your TensorFlow model loading code
model = None  # Placeholder for model loading

# Create button for detecting number
button = tk.Button(window, text="Detect Number", command=detect_button_click)
button.pack()

# Run the Tkinter event loop
window.mainloop()
