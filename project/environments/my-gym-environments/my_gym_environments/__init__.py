from gymnasium.envs.registration import register

register(
    id='LinearFirstOrder-v0',
    entry_point='my_gym_environments.envs:LinearFirstOrderEnv',
    max_episode_steps=300,
)

register(
    id='LinearSecondOrder-v0',
    entry_point='my_gym_environments.envs:LinearSecondOrderEnv',
    max_episode_steps=300,
)

register(
    id='MyPendulum-v0',
    entry_point='my_gym_environments.envs:MyPendulumEnv',
    max_episode_steps=300,
)







