import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os
from gtts import gTTS

# Load the model
try:
    model = tf.keras.models.load_model("modified_model_savedmodel")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)

my_w = tk.Tk()
my_w.geometry("400x400")
my_w.title(' Detection System')
my_font1 = ('times', 18, 'bold')

filename = ""
uploaded_image_label = tk.Label(my_w)

def upload_file():
    global filename
    f_types = [('All Files', '*.*')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    if filename:
        image = Image.open(filename)
        imgs = image.resize((224, 224))
        img = ImageTk.PhotoImage(imgs)
        uploaded_image_label.config(image=img)
        uploaded_image_label.image = img
        uploaded_image_label.grid(row=9, column=1, padx=5, pady=5)
        print("Uploaded file:", filename)

def predict():
    global model, filename

    if not filename:
        messagebox.showwarning("Warning", "Please upload an image first.")
        return

    try:
        img = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        disease_mapping = {
            0: "10 rupees",
            1: "100 rupees",
            2: "20 rupees",
            3: "200 rupees",
            4: "50 rupees",
            5: "Fake rupee",
            6: "Real rupee"
        }

        predicted_disease = disease_mapping.get(predicted_class, "Unknown Disease")
        out = f"Result for the given Image: {predicted_disease}"
        print(out)
        messagebox.showinfo("Result", out)

        # Use gTTS to convert text to speech
        language = 'en'
        tts = gTTS(text=out, lang=language, slow=False)
        tts.save("result.mp3")

        # Play the generated audio file
        os.system("start result.mp3")
    except Exception as e:
        print("Error predicting:", e)
        messagebox.showerror("Error", "Failed to predict. Please try again.")

l1 = tk.Label(my_w, text='Give Images', width=30, font=my_font1)
l1.grid(row=1, column=1)

b1 = tk.Button(my_w, text='Upload File', width=20, command=upload_file)
b1.grid(row=2, column=1, padx=5, pady=5)

b3 = tk.Button(my_w, text='Predict Output', width=20, command=predict)
b3.grid(row=6, column=1, padx=5, pady=5)

my_w.mainloop()
