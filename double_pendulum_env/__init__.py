from gym.envs.registration import register

register(
    id='DoubleInvertedPendulum-v1',
    entry_point='double_pendulum_env.double_inverted_pendulum:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)
