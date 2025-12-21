import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import neuralNetwork as NN
import constants as con

# Load the trained model (Run datasetToCSV.py then trainNeuralNetwork.py first)
weights, biases = NN.load_model(con.NPZ_FILE)

# GUI setup
CANVAS_SIZE = 300
MODEL_SIZE = con.IMAGE_SIZE
PREDICT_INTERVAL = 75  # ms (live prediction speed)

root = tk.Tk()
root.title("Doodle Guesser")
root.configure(bg="#f5f5f5")

main_frame = tk.Frame(root, bg="#f5f5f5", padx=15, pady=15)
main_frame.pack()

# Create Title
title = tk.Label(
    main_frame,
    text="Doodle Guesser",
    font=("Arial", 20, "bold"),
    bg="#f5f5f5"
)
title.pack(pady=(0, 5))

# Create subtitle
subtitle = tk.Label(
    main_frame,
    text="Draw one of the known objects",
    font=("Arial", 11),
    bg="#f5f5f5"
)
subtitle.pack(pady=(0, 5))

# List Drawable objects
instructions = tk.Label(
    main_frame,
    text=f"Known objects: {', '.join(con.CLASS_NAMES)}",
    font=("Arial", 9),
    bg="#f5f5f5",
    fg="#666666",
    wraplength=CANVAS_SIZE
)
instructions.pack(pady=(0, 10))

# Create Canvas
canvas = tk.Canvas(
    main_frame,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="white",
    bd=2,
    relief="ridge"
)
canvas.pack()

# Create Labels for Predictions
pred_label = tk.Label(
    main_frame,
    font=("Arial", 11),
    justify="left",
    bg="#f5f5f5"
)
pred_label.pack(pady=10)

# Create image variable
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
draw = ImageDraw.Draw(image)

# Store stroke history for undo
stroke_history = []  # List of (canvas_items, image_state)
current_stroke_items = []  # Canvas items in current stroke

last_x, last_y = None, None
predict_job = None
is_drawing = False

# --- DRAWING SECTION ---
# Start
def start_draw(event):
    global last_x, last_y, is_drawing, predict_job, current_stroke_items
    last_x, last_y = event.x, event.y
    is_drawing = True
    current_stroke_items = []

    # Starts prediction loop
    if predict_job is None:
        prediction_loop()

# Draw lines
def draw_motion(event):
    global last_x, last_y
    if last_x is not None:
        # Draw on canvas and store the item ID
        line_id = canvas.create_line(
            last_x, last_y, event.x, event.y,
            width=6,
            capstyle=tk.ROUND,
            smooth=True
        )
        current_stroke_items.append(line_id)
        
        # Draw on PIL image
        draw.line(
            [last_x, last_y, event.x, event.y],
            fill=0,
            width=16
        )

    last_x, last_y = event.x, event.y

# Reset the Draw 
def reset_draw(event):
    global last_x, last_y, is_drawing, predict_job, current_stroke_items
    
    # Save the stroke to history when finished
    if current_stroke_items:
        # Save a copy of the current image state
        image_copy = image.copy()
        stroke_history.append((current_stroke_items.copy(), image_copy))
        current_stroke_items = []
    
    last_x, last_y = None, None
    is_drawing = False

    if predict_job:
        root.after_cancel(predict_job)
        predict_job = None

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_motion)
canvas.bind("<ButtonRelease-1>", reset_draw)

# --- Preprocess the images --- 
def preprocess_image(img):
    bbox = img.getbbox()
    if not bbox:
        return None

    img = img.crop(bbox)

    img.thumbnail(MODEL_SIZE, Image.LANCZOS)

    new_img = Image.new("L", MODEL_SIZE, 255)
    x = (MODEL_SIZE[0] - img.size[0]) // 2
    y = (MODEL_SIZE[1] - img.size[1]) // 2
    new_img.paste(img, (x, y))

    return new_img

# --- Predict the Drawing ---
def predict_doodle():
    # Preprocess
    processed = preprocess_image(image)
    if processed is None:
        return

    # Process img to same format used by dataset
    arr = np.array(processed) / 255.0
    arr = arr.flatten().reshape(con.NUM_PIXELS, 1)

    # Predict image class
    _, acts = NN.forward_propagation(arr, weights, biases)
    probs = acts[-1].flatten()

    top = np.argsort(probs)[::-1][:5]

    if probs[top[0]] < 0.3:
        pred_label.config(text="Drawing...")
        return

    text = "\n".join(
        f"{con.CLASS_NAMES[i]}: {probs[i]*100:.1f}%"
        for i in top
    )
    pred_label.config(text=text)

# Start the prediction loop
def prediction_loop():
    global predict_job
    if is_drawing:
        predict_doodle()
        predict_job = root.after(PREDICT_INTERVAL, prediction_loop)

# --- Add Controls ---
# Clear command
def clear_canvas():
    global image, draw, predict_job, is_drawing, stroke_history, current_stroke_items
    # Delete
    canvas.delete("all")
    # Reset variables
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(image)
    pred_label.config(text="")
    is_drawing = False
    stroke_history = []
    current_stroke_items = []

    if predict_job:
        root.after_cancel(predict_job)
        predict_job = None

# Undo Command
def undo_stroke():
    global image, draw, stroke_history
    
    # Check if empty
    if not stroke_history:
        return  # Nothing to undo
    
    # Remove the last stroke from canvas
    last_stroke_items, _ = stroke_history.pop()
    for item_id in last_stroke_items:
        canvas.delete(item_id)
    
    # Restore image state to before the last stroke
    if stroke_history:
        # Get the image state from the previous stroke
        _, image = stroke_history[-1]
        image = image.copy() 
    else:
        # No more strokes, reset to blank
        image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    
    # Redraw canvas
    draw = ImageDraw.Draw(image)
    
    # Update prediction
    predict_doodle()

# Detect keyboard input
def handle_keypress(event):
    if event.keysym in ('c', 'C', 'Escape'):
        clear_canvas()
    elif event.keysym in ('z', 'Z') and event.state & 0x4:  # Ctrl+Z
        undo_stroke()

root.bind('<Key>', handle_keypress)

# --- Add Buttons ---
button_frame = tk.Frame(main_frame, bg="#f5f5f5")
button_frame.pack(pady=10)

# Clear Canvas
clear_btn = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 11),
    width=10,
    command=clear_canvas
)
clear_btn.pack(side=tk.LEFT, padx=5)

#Undo Stroke
undo_btn = tk.Button(
    button_frame,
    text="Undo",
    font=("Arial", 11),
    width=10,
    command=undo_stroke
)
undo_btn.pack(side=tk.LEFT, padx=5)

root.mainloop()