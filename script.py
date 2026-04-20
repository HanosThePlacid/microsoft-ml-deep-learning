import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Loads the seeds.csv dataset into a pandas DataFrame
seeds = pd.read_csv('seeds.csv')

# Defines which columns are features and which is the target label
seed_features = ['area','perimeter','compactness','kernel_length','kernel_width','asymmetry_coefficient','groove_length']
seed_label = 'species'

X = seeds[seed_features].values   # Feature matrix
y = seeds[seed_label].values      # Labels (0,1,2)

# Splits data into 70% training and 30% test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)

# Creates and trains a 3-layer neural network (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='relu', 
                          learning_rate_init=0.01, learning_rate='adaptive', 
                          max_iter=1000)
mlp_model.fit(x_train, y_train)

# Makes predictions on test set and prints accuracy metrics
predictions = mlp_model.predict(x_test)
print("Metrics:\n", classification_report(y_test, predictions))

# Plots the confusion matrix
seed_classes = ['Kama Wheat', 'Rosa Wheat', 'Canadian Wheat']
mcm = confusion_matrix(y_test, predictions)
plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(len(seed_classes)), seed_classes, rotation=45)
plt.yticks(np.arange(len(seed_classes)), seed_classes)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()

# Saves the trained model to a file
joblib.dump(mlp_model, './mlp_model.pkl')

# Loads model (optional) and predicts on two new seed samples
x_new = np.array([[12.73,13.75,0.8458,5.412,2.882,3.533,5.067],
                  [17.63,15.98,0.8673,6.191,3.561,4.076,6.06]])
preds = mlp_model.predict(x_new)
for p in preds:
    print(p, '(', seed_classes[p], ')')