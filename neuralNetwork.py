import numpy as np
import constants as con

# --- ACTIVATIONS ---
def leaky_relu(Z, alpha=con.LEAKY_ALPHA):
    return np.maximum(alpha * Z, Z)
    
    # Normal relu
    # return np.maximum(0, Z)

def leaky_relu_derivative(Z, alpha=con.LEAKY_ALPHA):
    dz = np.ones_like(Z)
    dz[Z < 0] = alpha
    return dz
    
    # Normal relu
    # return np.where(Z > 0, 1, 0)

def softmax(Z):
    exp_vals = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

def one_hot(labels):
    one_hot_labels = np.zeros((labels.size, con.NUM_ITEMS))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels.T

# --- Init ---
def init_params(input_size=con.NUM_PIXELS, hidden_layers=con.HIDDEN_LAYERS, output_size=con.NUM_ITEMS):
    layers = [input_size] + hidden_layers + [output_size]
    weights = []
    biases = []

    for i in range(len(layers) - 1):
        scale = np.sqrt(2.0 / (layers[i] + layers[i+1]))
        w = np.random.randn(layers[i+1], layers[i]) * scale
        b = np.random.randn(layers[i+1], 1) * 0.01
        weights.append(w)
        biases.append(b)
    return weights, biases

# --- Forward Pass ---
def forward_propagation(X, weights, biases):
    activations = [X]
    pre_activations = []

    for i in range(len(weights)-1):
        Z = weights[i] @ activations[-1] + biases[i]
        A = leaky_relu(Z)
        pre_activations.append(Z)
        activations.append(A)

    Z = weights[-1] @ activations[-1] + biases[-1]
    A = softmax(Z)
    pre_activations.append(Z)
    activations.append(A)

    return pre_activations, activations

# --- Backwards Pass ---
def back_propagation(X, Y, pre_activations, activations, weights, lambda_reg=con.LAMBDA_REG):
    m = X.shape[1]
    L = len(weights)
    grads_w = [0] * L
    grads_b = [0] * L

    # Output layer (with L2 regularization)
    dZ = activations[-1] - Y
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

    return grads_w, grads_b

# --- Update using momentum ---
def update_params(weights, biases, grads_w, grads_b, learning_rate, beta=0.8):
    if not hasattr(update_params, "V_dw"):
        update_params.V_dw = [np.zeros_like(w) for w in weights]
        update_params.V_db = [np.zeros_like(b) for b in biases]

    for i in range(len(weights)):
        update_params.V_dw[i] = beta * update_params.V_dw[i] + (1 - beta) * grads_w[i]
        update_params.V_db[i] = beta * update_params.V_db[i] + (1 - beta) * grads_b[i]
        weights[i] -= learning_rate * update_params.V_dw[i]
        biases[i] -= learning_rate * update_params.V_db[i]

    return weights, biases

# --- Creation of minibatches --
def create_mini_batches(X, Y, batch_size):
    m = X.shape[1]
    perm = np.random.permutation(m)
    X_shuffled = X[:, perm]
    Y_shuffled = Y[:, perm]

    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        Y_batch = Y_shuffled[:, i:i+batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches


# --- Calculate loss ---
def compute_loss(Y_hat, Y, weights=None, lambda_reg=con.LAMBDA_REG):
    m = Y.shape[1]
    # Find loss using cross entrophy
    cross_entropy = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
    
    # Add the L2 regularization penalty to loss value
    if weights is not None:
        l2_penalty = (lambda_reg / (2 * m)) * sum(np.sum(w ** 2) for w in weights)
        return cross_entropy + l2_penalty
    
    return cross_entropy

# --- Predict ---
def predict(X, weights, biases):
    _, activations = forward_propagation(X, weights, biases)
    return np.argmax(activations[-1], axis=0)

# --- Accuracy ---
def get_accuracy(preds, labels):
    return np.mean(preds == labels)

# --- Gradient Descent ---
def gradient_descent(X, labels, hidden_layers=con.HIDDEN_LAYERS, epochs=con.EPOCHS, learning_rate=con.LEARNING_RATE, batch_size=con.BATCH_SIZE):
    # Init
    Y = one_hot(labels)
    weights, biases = init_params(X.shape[0], hidden_layers, con.NUM_ITEMS)

    # Reset momentum values
    if hasattr(update_params, "V_dw"):
        del update_params.V_dw
    if hasattr(update_params, "V_db"):
        del update_params.V_db

    m = X.shape[1]

    # MAIN LOOP
    for epoch in range(epochs):
        mini_batches = create_mini_batches(X, Y, batch_size)
        epoch_loss = 0

        for X_batch, Y_batch in mini_batches:
            # Forward prop
            pre_acts, acts = forward_propagation(X_batch, weights, biases)
            # Find loss
            loss = compute_loss(acts[-1], Y_batch, weights)
            # Back Prop
            grads_w, grads_b = back_propagation(X_batch, Y_batch, pre_acts, acts, weights)
            # Update Params
            weights, biases = update_params(weights, biases, grads_w, grads_b, learning_rate)

            # Accumulate loss 
            epoch_loss += loss

        # Average epoch loss
        epoch_loss /= len(mini_batches)

        # Print out status of training every 50 epochs
        if epoch % 50 == 0:
            preds = predict(X, weights, biases)
            acc = get_accuracy(preds, labels)
            print(
                f"Epoch {epoch:02d} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Train Acc: {acc:.4f}"
            )

    return weights, biases


# --- SAVE / LOAD ---
# Save into NPZ_FILE declared in contants.py
def save_model(weights, biases, filename=con.NPZ_FILE):
    savez_dict = {}
    for i, (w, b) in enumerate(zip(weights, biases)):
        savez_dict[f"w{i}"] = w
        savez_dict[f"b{i}"] = b
    np.savez(filename, **savez_dict)

# load from NPZ_FILE declared in constans.py
def load_model(filename=con.NPZ_FILE):
    data = np.load(filename, allow_pickle=True)
    weights, biases = [], []
    i = 0
    while f"w{i}" in data:
        weights.append(data[f"w{i}"])
        biases.append(data[f"b{i}"])
        i += 1
    return weights, biases

# --- Predict a single Image  ---
def predict_single_image(x, weights, biases):
    x = x.reshape(con.NUM_PIXELS, 1)
    pred = predict(x, weights, biases)
    return pred[0]
