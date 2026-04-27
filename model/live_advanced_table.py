import tkinter as tk
from tkinter import ttk
import pandas as pd

class LiveTableApp:
    def __init__(self, root, shared):
        self.root = root
        self.shared = shared  # A dictionary that holds the latest DataFrame
        self.root.title("Live RMFS Table")

        self.tree = ttk.Treeview(root)
        self.tree["show"] = "headings"
        self.tree.pack(expand=True, fill='both')

        self.refresh_data()

    def update_table(self, df):
        # Clear existing columns
        self.tree.delete(*self.tree.get_children())
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
        self.tree["columns"] = []

        # Set new columns
        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")

        # Insert rows
        for _, row in df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def refresh_data(self):
        df = self.shared["df"]
        if df is not None and not df.empty:
            self.update_table(df)
        self.root.after(1000, self.refresh_data)

def start_gui(shared):
    root = tk.Tk()
    app = LiveTableApp(root, shared)
    root.mainloop()
