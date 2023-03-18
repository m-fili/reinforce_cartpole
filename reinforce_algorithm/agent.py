import torch


class Agent:

    def __init__(self, env, model, stochastic_policy=True):
        self.env = env
        self.n_states = env.observation_space.shape[0]
        self.n_action = env.action_space.n
        self.stochastic_policy = stochastic_policy
        self.model = model
        self.device = torch.device('cpu')

    def select_action(self, state):
        state = torch.tensor(state).to(self.device)
        probs = self.model.forward(state).cpu()
        m = torch.distributions.Categorical(probs)
        if self.stochastic_policy:
            action = m.sample()
        else:
            action = torch.argmax(probs)
        log_prob = m.log_prob(action)
        return action.item(), log_prob

    def calculate_return(self, gamma=1.0, t_max=1000):
        """
        Calculates discounted return for a completed episode using the current policy in agent.
        """
        s1, _ = self.env.reset()
        rewards = []
        log_probs_list = []

        for t in range(t_max):
            action, log_prob = self.select_action(s1)
            s2, reward, done, info, _ = self.env.step(action)
            rewards.append(reward)
            log_probs_list.append(log_prob)
            s1 = s2
            if done:
                break
        # calculate the total discounted return for the episode
        discounts = [gamma ** i for i in range(len(rewards))]
        G = sum([a * b for a, b in zip(rewards, discounts)])

        return G, log_probs_list

    def load_best_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
