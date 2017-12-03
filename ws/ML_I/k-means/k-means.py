import numpy as np


def loadCsv():
    return np.genfromtxt('data.csv', delimiter=',')

def initialize_centroids(points, k):
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):

    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def main(points):
    num_iterations = 100
    k = 3

    centroids = initialize_centroids(points, 3)

    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)
    
    return centroids

if __name__ == '__main__':
    points = loadCsv()
    centroids = main(points)
    print(centroids)