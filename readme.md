project/
│
├── agents/
│   ├── TD3.py
│   ├── Agent.py
│
├── environments/
│   ├── data_generation_for_reverse_mapping.py
│   ├── noise.py
│
├── metrics/
│   ├── mapping_metrics_and_weights.py
│
├── models/
│   ├── reverse_mapping_ANN.py
│
├── runners/
│   ├── runner_linearfirstorder.py
│   ├── runner_new_obs.py
│
├── experiments/
│   ├── record_experiment.py
│   ├── plot_results.py
│
├── main.py
│
└── requirements.txt


Agent.py defines an Agent class that uses a TD3 object from TD3.py and contains methods for acting, memorizing experiences, and learning from them.
TD3.py contains the definitions for the Actor and Critic networks and the logic for the TD3 algorithm.
runner_linearfirstorder.py defines a SimpleRunner class that sets up the environment, runs the episodes, and manages the training and evaluation of the agent.
