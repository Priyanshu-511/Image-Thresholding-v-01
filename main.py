import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys

class ThresholdApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Thresholding App")
        self.root.geometry("800x600")

        # Exit cleanly if GUI is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize image variables
        self.img = None
        self.binary = None

        # Upload button
        self.upload_btn = ttk.Button(root, text="Upload Image", command=self.load_image)
        self.upload_btn.pack(pady=10)

        # Threshold slider
        self.thresh_slider = ttk.Scale(root, from_=0, to=255, orient="horizontal", command=self.update_threshold)
        self.thresh_slider.set(128)
        self.thresh_slider.pack(fill="x", padx=20, pady=10)

        self.label = ttk.Label(root, text="Threshold: 128")
        self.label.pack()

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(1,2, figsize=(8,4))
        for a in self.ax:
            a.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif")])
        if not path:
            return

        self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.update_threshold(self.thresh_slider.get())

    def update_threshold(self, val):
        if self.img is None:
            return

        th = int(float(val))
        self.label.config(text=f"Threshold: {th}")

        _, self.binary = cv2.threshold(self.img, th, 255, cv2.THRESH_BINARY)

        self.ax[0].cla()
        self.ax[0].imshow(self.img, cmap="gray")
        self.ax[0].set_title("Original")
        self.ax[0].axis("off")

        self.ax[1].cla()
        self.ax[1].imshow(self.binary, cmap="gray")
        self.ax[1].set_title(f"Thresholded (T={th})")
        self.ax[1].axis("off")

        self.canvas.draw()

    def on_closing(self):
        """Clean exit when GUI window is closed"""
        plt.close('all')   # Close matplotlib windows
        self.root.destroy()
        sys.exit(0)        # Ensure process exits

if __name__ == "__main__":
    root = tk.Tk()
    app = ThresholdApp(root)
    root.mainloop()
