from sklearn.mixture import GaussianMixture

def detect_anomalies_gmm(data):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(data)
    scores = gmm.score_samples(data)
    anomalies = data[scores < threshold]
    return anomalies

# Example usage:
data = [[1], [2], [3], [10], [15], [100]]
anomalies = detect_anomalies_gmm(data)
print("Anomalies:", anomalies)
