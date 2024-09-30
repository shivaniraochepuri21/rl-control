import os
from experiments.record_experiment import run_experiment
import environments  # Ensure environments are registered

def main():
    config = {
        "agent": "TD3_agent",  # Define which agent to use
        "environment": "runner_linearfirstorder",  # Define which environment to use
        "reward_function": "default",  # Define which reward function to use
        "experiment_name": "linear_first_order_test",
        "num_episodes": 10,  # Reduced for quick test
        "steps_per_episode": 200,  # Reduced for quick test
    }

    run_experiment(config)

if __name__ == "__main__":
    main()

