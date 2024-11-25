import matplotlib.pyplot as plt

def plot_similarity_scores(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(scores, marker='o')
    plt.title('Similarity Scores', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True)
    plt.show()
