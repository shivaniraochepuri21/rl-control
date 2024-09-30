import os
import json
import importlib
import time
from datetime import datetime

def run_experiment(config):
    agent_module = importlib.import_module(f"agents.{config['agent']}")
    env_module = importlib.import_module(f"runners.{config['environment']}")


    env = env_module.SimpleRunner(num_episodes=config["num_episodes"])

    reward_function = config.get("reward_function", "default")

    # Experiment logging setup
    experiment_dir = os.path.join("experiments", "logs", config["experiment_name"])
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join(experiment_dir, "results.json")

    results = {
        "config": config,
        "episodes": []
    }

    for episode in range(config["num_episodes"]):
        start_time = time.time()
        episode_result = run_episode(env, config["steps_per_episode"])
        end_time = time.time()
        episode_result["duration"] = end_time - start_time
        results["episodes"].append(episode_result)

        with open(log_file, 'w') as f:
            json.dump(results, f, indent=4)

def run_episode(env, steps):
    env.run()
    return {
        "total_reward": env.score_history[-1] if env.score_history else 0,
        "steps": steps,
        "data": env.score_history
    }

