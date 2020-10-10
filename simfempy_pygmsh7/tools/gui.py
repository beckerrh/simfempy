import numpy as np
import tkinter
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure


class FempyGui():
    def __init__(self, solver=None):
        self.solver = solver
        self.root = tkinter.Tk()
        self.root.wm_title("FemPy")
        self.fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        button = tkinter.Button(master=self.root, text="Quit", command=self.quit)
        button.pack(side=tkinter.BOTTOM)
        button = tkinter.Button(master=self.root, text="Draw", command=self.draw)
        button.pack(side=tkinter.BOTTOM)
        self.run()
    def quit(self):
        self.root.quit()
        self.root.destroy()
    def run(self):
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        tkinter.mainloop()
    def on_key_press(self, event):
        print("you pressed {}".format(event.key))
    def draw(self):
        t = np.arange(0, 3, .01)
        self.fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
        self.canvas.draw()



fempygui = FempyGui()




