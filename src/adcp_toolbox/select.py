import tkinter as tk
from tkinter import filedialog


def single_file():
    """
    Opens a file dialog to select a file and returns the file path.
    """
    # Create and hide the root window
    root = tk.Tk()
    root.attributes("-alpha", 0)
    root.attributes("-topmost", True)
    # Open a file dialog window
    output = filedialog.askopenfilename(
        parent=root,
        title="Select a file"
    )
    file_path = output if output else None
    # Close the root window and return the path of the selected file
    root.update_idletasks()
    root.destroy()
    
    return file_path


def multiple_files():
    """
    Opens a file dialog to select multiple files and returns their file paths.
    """
    # Create and hide the root window
    root = tk.Tk()
    root.attributes("-alpha", 0)
    root.attributes("-topmost", True)
    # Open a file dialog window
    output = filedialog.askopenfilenames(
        parent=root,
        title="Select files"
    )
    file_paths = output if output else None
    # Close the root window and return paths of the selected files
    root.update_idletasks()
    root.destroy()
    
    return file_paths


def folder():
    """
    Opens a folder dialog and returns the path of the selected folder.
    """
    # Create and hide the root window
    root = tk.Tk()
    root.attributes("-alpha", 0)
    root.attributes("-topmost", True)
    # Open a folder dialog window
    path = filedialog.askdirectory(parent=root, title="Select a folder")
    path = path if path else None
    # Close the root window and return the path of the selected folder
    root.update_idletasks()
    root.destroy()
    
    return path