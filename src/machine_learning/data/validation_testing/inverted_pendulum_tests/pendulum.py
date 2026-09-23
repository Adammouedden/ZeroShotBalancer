import os
import gymnasium as gym

xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custompendulum.xml")

env = gym.make(
    "InvertedPendulum-v5",
    xml_file=xml_path,
    render_mode="human"
)

observation, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()