import pandas as pd
import numpy as np
import neuralNetwork as NN
import constants as con

# --- Load csv ---
df = pd.read_csv(con.CSV_FILE)
labels = df["label"].to_numpy(dtype=int)
X = df.drop(columns=["label"]).to_numpy(dtype=float)

m = X.shape[0]

# Shuffle
perm = np.random.permutation(m)
X = X[perm]
labels = labels[perm]

# Train/test split
split_idx = int(m * con.TEST_SPLIT)
X_test = X[:split_idx].T
label_test = labels[:split_idx]

X_train = X[split_idx:].T
label_train = labels[split_idx:]

print(f"Training samples: {X_train.shape[1]}, Test samples: {X_test.shape[1]}")

# --- Train NN ---
print(f"Training with learning rate: {con.LEARNING_RATE}")
weights, biases = NN.gradient_descent(X_train, label_train, hidden_layers=con.HIDDEN_LAYERS, epochs=con.EPOCHS, learning_rate=con.LEARNING_RATE, batch_size=con.BATCH_SIZE)



# --- Evaluate the Model ---

preds_test = NN.predict(X_test, weights, biases)
_, activations_test = NN.forward_propagation(X_test, weights, biases)
confidence_scores = np.max(activations_test[-1], axis=0)

# Find Total test accuracy 
acc_test = NN.get_accuracy(preds_test, label_test)
print(f"\nTest Accuracy: {acc_test:.4f}")

# Get the per class accuracy
print("\nPer class accuracy:")
for cls in range(con.NUM_ITEMS):
    mask = label_test == cls
    if np.any(mask):
        class_acc = np.mean(preds_test[mask] == cls)
        print(f"  Class {cls}: {class_acc:.2%}")

for i in range(con.NUM_ITEMS):
    # Confusion analysis for Class i
    print(f"\nClass {i} ({con.CLASS_NAMES[i]}) confusion matrix:")
    class_mask = label_test == i
    class_preds = preds_test[class_mask]
    for cls in range(con.NUM_ITEMS):
        count = np.sum(class_preds == cls)
        print(f"  Predicted as Class {cls} ({con.CLASS_NAMES[cls]}): {count} times")


# --- Save models weights and biases ---
NN.save_model(weights, biases, filename=con.NPZ_FILE)
print("\nModel saved as ", con.NPZ_FILE)

# --- Neuron activation stats ---
# Was needed when debugging dying neurons
# _, acts = NN.forward_propagation(X_train, weights, biases)
# print("\nFinal activation statistics (Training Set):")
# for i, act in enumerate(acts[1:-1]):
#    active_neurons = np.sum(np.any(act != 0, axis=1))
#    total_neurons = act.shape[0]
#    print(f"Layer {i}: {active_neurons}/{total_neurons} neurons active ({active_neurons/total_neurons:.1%})")
#    print(f"  Mean: {np.mean(act):.4f}, Std: {np.std(act):.4f}")
