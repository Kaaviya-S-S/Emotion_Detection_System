from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from BuildingCNN import model
from Generator import validation_generator
import numpy as np


# Limit the number of batches to process
batch_limit = 10  # Process only 10 batches, adjust as needed

true_labels = []
predicted_labels = []

# Loop over the generator and collect data for a limited number of batches
for i, (batch_x, batch_y) in enumerate(validation_generator):
    if i >= batch_limit:  # Stop after processing the batch_limit
        break

    # Collect true labels
    true_labels.extend(np.argmax(batch_y, axis=1))

    # Predict for the current batch
    y_pred_batch = model.predict(batch_x)
    predicted_labels.extend(np.argmax(y_pred_batch, axis=1))

print("done")
# Convert to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

