import numpy as np
import torch
from collections import deque


def reinforce(agent, n_episodes=1000, m=1, gamma=1.0, t_max=1000, learning_rate=1e-2):
    scores = []
    scores_deque = deque(maxlen=100)
    TARGET_SCORE = 200.0

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate)

    for i_episode in range(1, n_episodes + 1):

        for trajectory in range(m):  # for single trajectory 1...m

            policy_losses = []
            policy_R = []

            R, log_probs_list = agent.calculate_return(gamma, t_max)

            trajectory_loss = []
            for log_prob in log_probs_list:
                trajectory_loss.append(-log_prob * R)

            trajectory_loss = torch.stack(trajectory_loss).sum()
            policy_losses.append(trajectory_loss)
            policy_R.append(R)

        # Save scores
        mean_R = np.mean(policy_R)
        scores.append(mean_R)
        scores_deque.append(mean_R)

        # Optimization
        mean_policy_losses = torch.stack(policy_losses).mean()
        optimizer.zero_grad()
        mean_policy_losses.backward()
        optimizer.step()

        if i_episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= TARGET_SCORE:
            torch.save(agent.model.state_dict(), 'checkpoint.pth')
            print(f'Agent reached the target score of {TARGET_SCORE} in {i_episode} episodes')
            break

    return scores
