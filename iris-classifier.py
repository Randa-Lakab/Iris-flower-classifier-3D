import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# 3D Visualization with the user's flower
def plot_3d_data_with_user_flower(X_scaled, y, user_flower_scaled):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Data by class
    ax.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], X_scaled[y == 0][:, 2],
               color='red', label='Iris-setosa (0)')
    ax.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], X_scaled[y == 1][:, 2],
               color='blue', label='Iris-versicolor (1)')
    ax.scatter(X_scaled[y == 2][:, 0], X_scaled[y == 2][:, 1], X_scaled[y == 2][:, 2],
               color='purple', label='Iris-virginica (2)')

    # User's flower
    ax.scatter(user_flower_scaled[0], user_flower_scaled[1], user_flower_scaled[2],
               color='green', s=100, marker='X', label='User Flower')

    # Labels
    ax.set_title("3D Visualization of Flowers (First 3 Features)")
    ax.set_xlabel("Sepal Length (normalized)")
    ax.set_ylabel("Sepal Width (normalized)")
    ax.set_zlabel("Petal Length (normalized)")
    ax.legend()
    plt.show()

# Function to find the closest flower
def find_closest_flower(user_flower, X_scaled):
    first_three_flowers = X_scaled[:3]
    distances = cdist([user_flower], first_three_flowers, metric='euclidean')
    closest_index = np.argmin(distances)
    return closest_index, distances[0, closest_index]

# Main program
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Enter the characteristics of a flower:")
    sepal_length = float(input("Sepal length (cm): "))
    sepal_width = float(input("Sepal width (cm): "))
    petal_length = float(input("Petal length (cm): "))
    petal_width = float(input("Petal width (cm): "))

    user_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    user_flower_scaled = scaler.transform(user_flower)

    closest_index, distance = find_closest_flower(user_flower_scaled[0], X_scaled)

    print(f"\nThe closest flower is flower number {closest_index + 1}.")
    print(f"Euclidean distance: {distance:.4f}")
    print(f"Characteristics: {X[closest_index]}")
    print(f"Flower name: {target_names[y[closest_index]]}")

    # 3D visualization using the first 3 features
    plot_3d_data_with_user_flower(X_scaled, y, user_flower_scaled[0])
