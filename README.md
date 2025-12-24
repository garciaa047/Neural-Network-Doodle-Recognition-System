# Neural Network Doodle Classifier

A fully custom neural network implementation built from scratch in NumPy to classify hand drawn doodles in real time. This project aims to gain a deeper understanding of neural network fundamentals by implementing backpropagation, optimization algorithms, and regularization techniques without relying on high level already existing ML frameworks.

## Project Highlights

- **Built from scratch**: Complete neural network implementation using only NumPy and Pandas (no TensorFlow/PyTorch)
- **90%+ accuracy** achieved on test data using only 500 images per class
- **Real time predictions**: Interactive GUI with live classification as you draw
- **Advanced techniques**: L2 regularization, mini-batch gradient descent, momentum optimization, Leaky ReLU activation, cross entropy loss calculation
- **Data augmentation**: Custom augmentation pipeline (rotation, translation, Gaussian noise) to triple training data (Original + 2 Copies)

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **90%+** |
| Training Time | 1-2 minutes |
| Training Samples | 7,500 (500 per class × 3 with augmentation) |
| Inference Speed | Real-time (~75ms prediction interval) |

### Classification Classes Included
`airplane` • `apple` • `clock` • `pants` • `tree`

## Architecture

**Input Layer**: 900 neurons (30×30 pixel grayscale images)  
**Hidden Layers**: 128 -> 64 neurons with Leaky ReLU activation  
**Output Layer**: 5 neurons with Softmax activation

### Key Features Implemented

#### 1. **Backpropagation Algorithm**
Implemented gradient computation through all layers with proper chain rule application for both hidden (Leaky ReLU) and output (Softmax) layers.

#### 2. **Regularization Technique**
- **L2 Regularization** (lambda=0.003) to prevent overfitting

#### 3. **Optimization Strategies**
- **Momentum** (beta=0.8) for faster convergence and reduced oscillation
- **He initialization** for proper weight scaling for leaky ReLU
- **Learning rate**: 0.001 for stable training
-  **Leaky ReLU** (alpha=0.1) to prevent dying neurons
- **Mini-batch gradient descent** (batch size: 64) for stable convergence

#### 4. **Data Augmentation Pipeline**
Each training image generates 2 augmented copies with:
- Random rotation (Up to 15 degrees)
- Random translation (Up to 5 pixels)
- Gaussian noise injection (alpha=0.01-0.03)

This effectively **triples the dataset** from 2,500 to 7,500 images.

# Getting Started

### Prerequisites

```bash
Python 3.10+
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/garciaa047/Neural-Network-Doodle-Recognition-System.git
cd Neural-Network-Doodle-Recognition-System
```

2. **Install dependencies**
```bash
pip install numpy pandas pillow
```

3. **Verify dataset structure**
```
project/
├── constants.py
├── datasetToCsv.py
├── neuralNetwork.py
├── trainNeuralNetwork.py
├── doodle_guesser.py
└── doodleDataset/
    ├── airplane/
    ├── apple/
    ├── clock/
    ├── pants/
    └── tree/
```

### Usage

#### Step 1: Prepare the Dataset
```bash
python datasetToCsv.py
```
This script:
- Loads images from `doodleDataset/` folder
- Applies data augmentation
- Exports normalized pixel data to `doodle_pixels.csv`

#### Step 2: Train the Model
```bash
python trainNeuralNetwork.py
```
Training progress will display:
```
Epoch 00 | Loss: 1.7083 | Train Acc: 0.2287
Epoch 50 | Loss: 1.4422 | Train Acc: 0.5433
Epoch 100 | Loss: 1.0624 | Train Acc: 0.8845
...
Epoch 450 | Loss: 0.2273 | Train Acc: 0.9295

Test Accuracy: 0.9333

Per class accuracy:
...
Per class confusion matrix:
...

Model saved as doodle_model.npz
```
Model weights and biases will be saved under doodle_model.npz

#### Step 3: Launch the Tkinter Drawing Interface
```bash
python doodle_guesser.py
```

**Controls**:
- **Draw**: Click and drag on the canvas
- **Clear**: Press `C` or `Esc` to clear canvas
- **Undo**: Press `Ctrl+Z` to undo last stroke

View the prediction in **real time**!

# Training Analysis

## Test Accuracy
```
Test Accuracy: 0.9333
```
Shows an impressive 93% test accuracy, especially when considering the small dataset (~7500 images), using a FNN instead of a CNN.

### Loss Value
```
1.7083 -> 0.2273
```
Loss shows a significant decrease, with it not yet plateauing, showing a potential for even better results if the number of epochs are increased.


### Per-Class Accuracy
```
Class 0: 93.86%
Class 1: 93.83%
Class 2: 94.72%
Class 3: 93.94%
Class 4: 90.30%
```
Shows a 90%+ accuracy for each class, with a relatively equal distribution.

### Confusion Matrix (Class 0 Example)
```
Class 0 (airplane) confusion matrix:
  Predicted as Class 0 (airplane): 275 times
  Predicted as Class 1 (clock): 0 times
  Predicted as Class 2 (tree): 16 times
  Predicted as Class 3 (pants): 2 times
  Predicted as Class 4 (apple): 0 times
```
Sample Confusion matrix shows what class 0 (airplane) gets confused with, noticeable confusion with tree.

# Forward & Backward Propagation Simplified

### Forward Propagation
```python
# For Hidden Layers
for i in range(len(weights)-1):
  Z = weights[i] @ activations[-1] + biases[i]
  A = leaky_relu(Z)
# For output layer
Z = weights[-1] @ activations[-1] + biases[-1]
A = softmax(Z)
```
Find the pre activation value by weight x value + bias, in the case of a matrix use dot product (@).

Then put that value through the activation function to get neuron values.

### Backpropagation with Regularization
```python
# Output layer gradient

# Output layer (with L2 regularization)
dZ = activations[-1] - Y # Cross entropy loss and softmax activation means derivative calculated easily by A3 - Y (one hot encoded labels)

# Output Layer gradients
grads_w[-1] = (1/m) * dZ @ activations[-2].T + (lambda_reg/m) * weights[-1]
grads_b[-1] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

# Hidden layers (with L2 regularization)
dA = weights[-1].T @ dZ
for l in reversed(range(L-1)):
  dZ = dA * leaky_relu_derivative(pre_activations[l])
  grads_w[l] = (1/m) * dZ @ activations[l].T + (lambda_reg/m) * weights[l]
  grads_b[l] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
  if l != 0:
    dA = weights[l].T @ dZ
...
```
Caclulate the dZ (Error at current layer), from dA (Previous layer's error) and derivative of activation function.

Then find the gradient of the weights, with dZ and the activated values, then add L2 regularization to reduce large weights. 

The bias gradient is then found easily by averaging out the sum of current layer's error.

Both are multiplied by (1/m) to find the average of each gradient over the mini batch.


### Update weights and biases with Momentum
```
for i in range(len(weights)):
  update_params.V_dw[i] = beta * update_params.V_dw[i] + (1 - beta) * grads_w[i]
  update_params.V_db[i] = beta * update_params.V_db[i] + (1 - beta) * grads_b[i]
  weights[i] -= learning_rate * update_params.V_dw[i]
  biases[i] -= learning_rate * update_params.V_db[i]
```
Update the velocity values for both weight and bias, using the following formula, 
- Vdw = beta * Vdw + (1 - beta) * gradW

Then update the actual weight and bias with the following formula, 
- w_new = w_old - learning_rate * Vdw

### Loss Function
```python
cross_entropy = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
l2_penalty = (lambda_reg / (2 * m)) * sum(np.sum(w ** 2) for w in weights)
loss = cross_entropy + l2_penalty
```
Use a combination of cross entropy and L2 regularization for loss to help ensure generalization.

## Configuration

Configuration is done simply through the use of the `constants.py` file:

```python
...
IMAGES_PER_ITEM = 500                               # Number of images per item you would like to train on
IMAGE_SIZE = (30, 30)                               # Image size in pixels, 30 x 30 pixels
...
TEST_SPLIT = 0.2                                    # Amount of data to test with, 0.2 -> 20% test and 80% Train
LEARNING_RATE = 0.001                               # Learning Rate, lower the more stable
HIDDEN_LAYERS = [128, 64]                        # Number of hidden layers and their size, currently (900 -> 128 -> 64 -> 5)
...
```

Easily able to change values such as the train and test split percentage or even the size or quantity of images.

## What I Learned

### Neural Network Fundamentals
- Implemented backpropagation from mathematical principles
- Understood the importance of proper weight initialization
- Learned how different activation functions affect training dynamics

### Optimization Challenges
- Experienced vanishing gradients with standard ReLU (solved with Leaky ReLU)
- Discovered the impact of regularization on generalization
- Balanced training speed vs. stability with learning rate tuning

### Practical ML Engineering
- Data augmentation significantly improves model robustness, while also provding additional data in small datasets
- Mini batch training provides better convergence and generalization than full batch
- Real time inference requires careful preprocessing pipeline design to conform to any forms of normalization done in the dataset

## Known Limitations

1. **Drawing Style**: The model struggles with drawings that don't match the Quick Draw dataset style

2. **Dataset Quality**: The Quick Draw dataset contains many poorly drawn or mislabeled samples, with some samples being unrecognizable, which impacts maximum achievable accuracy

3. **Limited Classes**: Currently uses only 5 classes, the introduction of more classes will increase the number of mislabels

## Potential Improvements

- [ ] Implement Convolutional Neural Network (CNN) architecture which perform better on spatial information
- [ ] Potentially add dropout layers for improved regularization
- [ ] Testing on other datasets
- [ ] Improving the real time drawing GUI

## Project Structure

```
.
├── constants.py              # Centralized configuration
├── datasetToCsv.py          # Dataset preprocessing & augmentation
├── neuralNetwork.py         # Core NN implementation
├── trainNeuralNetwork.py    # Training script with evaluation
├── doodle_guesser.py        # Real-time drawing GUI
├── doodle_pixels.csv        # Processed training data (Created after running datasetToCSV.py)
├── doodle_model.npz         # Trained model weights (Created after running trainNeuralNetwork.py)
└── doodleDataset/           # Raw image dataset
    ├── airplane/
    ├── apple/
    ├── clock/
    ├── pants/
    └── tree/
```

## Dataset

This project uses a curated subset of the [Google Quick, Draw! Dataset](https://www.kaggle.com/datasets/ashishjangra27/doodle-dataset), containing 3000 hand-drawn examples for each of 340 categories. The dataset is included in the repository. 

## Contributing

This is a personal learning project, but suggestions and feedback are welcome!