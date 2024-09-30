import json
import matplotlib.pyplot as plt

def plot_results(experiment_name):
    log_file = f"experiments/logs/{experiment_name}/results.json"

    with open(log_file, 'r') as f:
        results = json.load(f)

    total_rewards = [episode["total_reward"] for episode in results["episodes"]]
    steps = [episode["steps"] for episode in results["episodes"]]

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(total_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.suptitle(f'Experiment: {experiment_name}')
    plt.show()

if __name__ == "__main__":
    plot_results("linear_first_order_test")

