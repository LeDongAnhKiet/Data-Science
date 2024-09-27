from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import threading

model = load_model('mnist.keras')

def predict_digit(img):
    img = img.resize((28, 28)).convert('L')
    img = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # Ensure the image is valid and not empty
    if np.sum(img) == 0:  # Check if the image is completely blank
        return 0, 0.0  # Return a default digit and accuracy

    res = model.predict(img)[0]
    
    # Handling the case where the model prediction is invalid
    if np.any(np.isnan(res)):
        return 0, 0.0  # Default digit and accuracy for NaN values
    
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.title('Simple MNIST')
        self.canvas = tk.Canvas(self, width=300, height=300, bg='white', cursor='cross')
        self.label = tk.Label(self, text='Thinking..', font=('Helvetica', 30))
        self.classify_btn = tk.Button(self, text='Recognise', command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text='Clear', command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # Bind events
        self.canvas.bind('<B1-Motion>', self.draw_lines)
        self.canvas.bind('<Button-3>', lambda event: self.clear_all())  # Right-click to clear
        self.bind('<space>', lambda event: self.clear_all())  # Space key to clear
        self.bind('<Return>', lambda event: self.classify_handwriting())  # Enter key to recognize

    def clear_all(self):
        self.canvas.delete('all')

    def classify_handwriting(self):
        threading.Thread(target=self._classify_handwriting_thread).start()

    def _classify_handwriting_thread(self):
        HWND = self.canvas.winfo_id()
        rect = self.canvas.bbox('all')
        im = ImageGrab.grab(bbox=(self.winfo_rootx() + rect[0], self.winfo_rooty() + rect[1], 
                                   self.winfo_rootx() + rect[2], self.winfo_rooty() + rect[3]))

        digit, acc = predict_digit(im)
        self.label.configure(text=f'Dự đoán: {digit}\nChính xác: {int(acc * 100)}%')

    def draw_lines(self, event):
        r = 8
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='black')

app = App()
mainloop()

