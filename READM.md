# Neural Network Playground

A simple graphical interface built with Tkinter to experiment with neural networks. This project allows you to generate datasets, load and save them, and train a custom neural network defined in a separate `neural_network` module.

## 🚀 Features

- Generate random datasets with specified size and dimension
- Save and load datasets as `.json` files
- Configure neural network architecture:
  - Number of hidden layers
  - Activation function (sigmoid, relu, tanh)
  - Learning rate and epochs
- Train the model and visualize the Mean Squared Error (MSE)
- View all training logs in a built-in terminal

## 🧠 Dependencies

- Python 3.x
- `neural_network.py` module (must contain `DeepLearning` class and `mean_squared_error` function)
- Tkinter (usually included with Python)


## 🛠 How to Use

1. Run `main.py`
2. Use the menu to:
   - Create a new dataset
   - Modify dataset size or dimension
   - Load or save datasets
3. Set your neural network parameters (learning rate, epochs, hidden layer sizes, activation)
4. Start training via the "Training" menu
5. View training logs and MSE results in the integrated terminal