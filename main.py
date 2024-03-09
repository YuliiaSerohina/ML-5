import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.utils import normalize
from keras.datasets import mnist
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'Training Data: {x_train.shape}')
print(f'Training Labels: {y_train.shape}')
x_train = normalize(x_train.reshape(x_train.shape[0], -1))
x_test = normalize(x_test.reshape(x_test.shape[0], -1))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

k_values = [4, 8, 10, 12]
inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_train)
    inertia_values.append(kmeans.inertia_)

plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

k_best = 10
kmeans = KMeans(n_clusters=k_best, random_state=42)
kmeans.fit(x_train)

cluster_to_label = {}
for cluster in range(k_best):
    cluster_samples_indices = np.where(kmeans.labels_ == cluster)[0]
    cluster_labels = y_train[cluster_samples_indices]
    most_common_label = np.bincount(cluster_labels).argmax()
    cluster_to_label[cluster] = most_common_label

correct_predictions = 0
total_samples = len(y_test)
for i in range(total_samples):
    cluster = kmeans.predict(x_test[i].reshape(1, -1))[0]
    predicted_label = cluster_to_label[cluster]
    if predicted_label == y_test[i]:
        correct_predictions += 1
accuracy = correct_predictions / total_samples
print("Classification Accuracy:", accuracy)

sample_images = []
for cluster in range(k_best):
    cluster_samples_indices = np.where(kmeans.labels_ == cluster)[0]
    sample_image_index = cluster_samples_indices[0]
    sample_image = x_train[sample_image_index].reshape(28, 28)
    sample_images.append(sample_image)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f'Cluster {i}')
plt.show()
