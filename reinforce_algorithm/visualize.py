import gym
import matplotlib.pyplot as plt
from IPython import display
from agent import Agent
from model import MLP
import argparse

parser = argparse.ArgumentParser(description="Visualize a trained agent for continuous cartpole problem.")
parser.add_argument("-f", "--file", help="file path to the model weights")
args = vars(parser.parse_args())

if __name__ == '__main__':

    file_path = args['file']

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    model = MLP(n_input=4, n_output=2, n_hidden=[16])
    agent = Agent(env, model)
    agent.load_best_model(file_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    state, _ = env.reset()
    img = ax.imshow(env.render())
    score = 0
    done = False

    s1, _ = env.reset()

    for t in range(1000):

        action, _ = agent.select_action(s1)
        s2, reward, done, info, _ = env.step(action)
        score += reward
        s1 = s2

        img.set_data(env.render())
        ax.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)

        if done:
            break

    print('Score Achieved:', score)

    env.close()
