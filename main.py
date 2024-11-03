from engine import Value
from nn import MLP

# Sample data (input features and target outputs)
xs = [[2.0, 3.0, -1.0], 
      [3.0, -1.0, 0.5], 
      [0.5, 1.0, 1.0], 
      [1.0, 1.0, -1.0]]  # Input data

ys = [0.0, 1.0, 1.0, 0.0]  # Target outputs

# Initialize the MLP with input and output sizes
n = MLP(3, [4, 4, 1])

