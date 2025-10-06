# plotting.py

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA

def plot_pca_2d(X, y, save_path="pca_2d.png"):
    """
    Performs 2D PCA and saves a scatter plot of the results.
    """
    pca_2d = PCA(n_components=2)
    x_pca_2d = pca_2d.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca_2d[:, 0], x_pca_2d[:, 1], c=y, cmap='plasma')
    plt.title('2D PCA of Heart Condition Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(save_path)
    plt.close()
    print(f"2D PCA plot saved to {save_path}")

def plot_pca_3d(X, y, save_path="pca_3d.png"):
    """
    Performs 3D PCA and saves a scatter plot of the results.
    """
    pca_3d = PCA(n_components=3)
    x_pca_3d = pca_3d.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_pca_3d[:, 0], x_pca_3d[:, 1], x_pca_3d[:, 2], c=y, cmap='plasma')
    ax.set_title('3D PCA of Heart Condition Data')
    plt.savefig(save_path)
    plt.close()
    print(f"3D PCA plot saved to {save_path}")