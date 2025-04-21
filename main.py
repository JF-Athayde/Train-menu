import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
import neural_network
import sys
import threading
import json

# Generate a 2D list of random floats
def randn(limit_a, limit_b, size, subsize):
    return [[round(random.uniform(limit_a, limit_b), 3) for _ in range(subsize)] for _ in range(size)]

# Generate binary output (0 or 1) as labels
def binary(size):
    return [[random.randint(0, 1)] for _ in range(size)]

# Save current dataset to a JSON file
def save_dataset():
    dataset = {"X_train": X_train, "y_train": y_train}
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if file_path:
        with open(file_path, "w") as f:
            json.dump(dataset, f)
        messagebox.showinfo("Success", "Dataset saved successfully!")

# Load dataset from a JSON file
def load_dataset():
    global X_train, y_train
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        with open(file_path, "r") as f:
            dataset = json.load(f)
        X_train, y_train = dataset["X_train"], dataset["y_train"]
        result_label.config(text="Dataset loaded successfully!")
        print("Loaded Dataset:")
        print("X_train:", X_train)
        print("y_train:", y_train)

# Create a new dataset with given size and dimensions
def modify_dataset():
    global X_train, y_train
    size = int(entry_size.get())
    subsize = int(entry_subsize.get())
    print(f"Creating new dataset with size {size} and dimension {subsize}...")
    X_train = randn(0, 1, size, subsize)
    y_train = binary(size)
    result_label.config(text="Dataset successfully created!")
    print("New Dataset:")
    print("X_train:", X_train)
    print("y_train:", y_train)

# Train the neural network with the current dataset
def train_model():
    global X_train, y_train
    learning_rate = float(entry_lr.get())
    epochs = int(entry_epochs.get())
    hidden_layers = [int(entry_layer1.get()), int(entry_layer2.get())]
    activation = activation_var.get()
    
    model = neural_network.DeepLearning(X_train, y_train, learning_rate, hidden_layers, activation)
    model.train(epochs=epochs, verbose=True)

    X_test = X_train.copy()
    y_predict = model.predict(X_test)
    mse = neural_network.mean_squared_error(y_train, y_predict)
    result_label.config(text=f'MSE: {mse:.4f}')
    print(f"Training completed. MSE: {mse:.4f}")

# Redirect standard output to the GUI text widget
class RedirectedStdout:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

# GUI setup
root = tk.Tk()
root.title("Neural Network Playground")

# Menu bar
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# Dataset menu
dataset_menu = tk.Menu(menu_bar, tearoff=0)
dataset_menu.add_command(label="Create New Dataset", command=modify_dataset)
dataset_menu.add_command(label="Modify Dataset", command=modify_dataset)
dataset_menu.add_command(label="Load Dataset", command=load_dataset)
dataset_menu.add_command(label="Save Dataset", command=save_dataset)
menu_bar.add_cascade(label="Dataset", menu=dataset_menu)

# Training menu
train_menu = tk.Menu(menu_bar, tearoff=0)
train_menu.add_command(label="Run Training", command=lambda: threading.Thread(target=train_model).start())
menu_bar.add_cascade(label="Training", menu=train_menu)

# Input fields and labels
tk.Label(root, text="Dataset Size:").pack()
entry_size = tk.Entry(root)
entry_size.pack()
entry_size.insert(0, "10")

tk.Label(root, text="Input Dimension:").pack()
entry_subsize = tk.Entry(root)
entry_subsize.pack()
entry_subsize.insert(0, "3")

tk.Label(root, text="Learning Rate:").pack()
entry_lr = tk.Entry(root)
entry_lr.pack()
entry_lr.insert(0, "0.01")

tk.Label(root, text="Epochs:").pack()
entry_epochs = tk.Entry(root)
entry_epochs.pack()
entry_epochs.insert(0, "100000")

tk.Label(root, text="Neurons in Hidden Layer 1:").pack()
entry_layer1 = tk.Entry(root)
entry_layer1.pack()
entry_layer1.insert(0, "6")

tk.Label(root, text="Neurons in Hidden Layer 2:").pack()
entry_layer2 = tk.Entry(root)
entry_layer2.pack()
entry_layer2.insert(0, "6")

tk.Label(root, text="Activation Function:").pack()
activation_var = tk.StringVar(root)
activation_var.set("sigmoid")
activation_menu = ttk.Combobox(root, textvariable=activation_var, values=["sigmoid", "relu", "tanh"])
activation_menu.pack()

result_label = tk.Label(root, text="")
result_label.pack()

# Text terminal for displaying logs/output
device_frame = tk.Frame(root)
device_frame.pack(fill=tk.BOTH, expand=True)
text_terminal = tk.Text(device_frame, height=10, wrap=tk.WORD)
text_terminal.pack(fill=tk.BOTH, expand=True)
sys.stdout = RedirectedStdout(text_terminal)

root.mainloop()
