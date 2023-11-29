import matplotlib.pyplot as plt
def calculate_frequency_distribution(likes):
    frequency_dict = {}
    for like in likes:
        frequency_dict[like] = frequency_dict.get(like, 0) + 1
    return frequency_dict
def plot_histogram(frequency_dict):
    likes = list(frequency_dict.keys())
    frequencies = list(frequency_dict.values())
    plt.bar(likes, frequencies, color='blue', alpha=0.7)
    plt.xlabel('Number of Likes')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution of Likes')
    plt.show()
if __name__ == "__main__":
    likes_data = [10, 15, 20, 10, 15, 25, 20, 15, 10, 20, 25, 30, 30, 25, 20, 15]
    frequency_distribution = calculate_frequency_distribution(likes_data)
    print("Frequency Distribution:")
    for likes, frequency in frequency_distribution.items():
        print(f"{likes} likes: {frequency} posts")
    plot_histogram(frequency_distribution)
