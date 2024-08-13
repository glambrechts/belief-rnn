import torch

from random import choices


class Trajectory:
    """
    Trajectory storing the observed action, observations, rewards, and
    termination flags.

    Arguments:
    - action_size: int
        The number of discrete actions for the environment.
    - observation_size: int
        The observation size for the environment.
    """

    def __init__(self, action_size, observation_size):

        self.action_size = action_size
        self.observation_size = observation_size
        self.observed = []
        self.terminal = False

    def add(self, action, reward, observation, terminal=False):
        """
        Adds a new action and its outcome to the trajectory.

        Arguments:
        - action: int
            The action played in the environment.
        - reward: float
            The reward obtained after playing this action.
        - observation: tensor
            The new observation.
        - terminal: bool
            Whether a terminal state has been reached.
        """
        assert not self.terminal

        one_hot = torch.zeros(self.action_size)
        if action is not None:
            one_hot[action] = 1.
        action = one_hot

        if reward is not None:
            reward = torch.tensor([reward], dtype=torch.float)
        else:
            reward = torch.tensor([0.], dtype=torch.float)

        self.observed.append(torch.cat((action, observation, reward)))

        if terminal:
            self.terminal = True

    @property
    def num_transitions(self):
        """
        Number of stored transitions.
        """
        return len(self.observed) - 1

    def get_cumulative_reward(self, gamma=1.0):
        """
        Returns the cumulative reward, discounted by gamma.

        Arguments:
        - gamma: float
            The discount factor.

        Returns:
        - cumulative_return: float
            The (discounted) cumulative return.
        """
        return sum(o[-1] * gamma ** t for t, o in enumerate(self.observed[1:]))

    def get_last_observed(self, number=None):
        """
        Returns the last observables (the last action and new observation).
        Note that the reward is not comprised in the observables.

        Arguments:
        - number: int
            Number of last oberservations.

        Returns:
        - observed: tensor
            The last observation(s).
        """
        if number is None:
            return self.observed[-1][:-1]
        else:
            truncated = [obs[:-1] for obs in self.observed[- number:]]
            if len(truncated) < number:
                padding = []
                for _ in range(number - len(truncated)):
                    padding.append(torch.zeros(self.observed[-1].size(0) - 1))
                truncated = padding + truncated
            return torch.stack(truncated)

    def get_transitions(self):
        """
        Returns the last observables (the last action and new observation).
        Note that the reward is not comprised in the observables.

        Returns:
        - transitions: list of tuples (eta, a, r', o', d', eta')
            The list of all transitions in the trajectory.
        """
        sequence = torch.stack(self.observed, dim=0)

        transitions = []
        for t in range(sequence.size(0) - 1):
            seq_bef = sequence[:t + 1, :-1]
            seq_aft = sequence[:t + 2, :-1]
            a = sequence[t + 1, :self.action_size]
            o = sequence[t + 1, self.action_size:-1]
            r = sequence[t + 1, -1]
            if a.sum() == 0:
                a = None
                r = None
            else:
                a = a.argmax()
                r = r.item()
            d = self.terminal and t == sequence.size(0) - 2
            transitions.append((seq_bef, a, r, o, d, seq_aft))

        return transitions


class ReplayBuffer:
    """
    Replay Buffer storing transitions.

    Arguments:
    - capacity: int
        The number of transitions that can be stored in the replay buffer.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.last = 0
        self.count = 0

    @property
    def is_full(self):
        """
        Whether the replay buffer is full.
        """
        return self.capacity == self.count

    def add_transition(self, transition):
        """
        Adds a transition to the replay buffer.

        Arguments:
        - transition: tuple (eta, a, r', o', d', eta')
            The transition to be added.
        """
        if self.count < self.capacity:
            self.buffer.append(transition)
            self.count += 1
        else:
            self.buffer[self.last] = transition
            self.last = (self.last + 1) % self.capacity

    def add(self, trajectory):
        """
        Adds all transitions of a trajectory to the replay buffer.

        Arguments:
        - trajectory: Trajectory
            The trajectory to be added.
        """
        assert isinstance(trajectory, Trajectory)
        for transition in trajectory.get_transitions():
            self.add_transition(transition)

    def sample(self, number):
        """
        Samples transitions from the replay buffer.

        Arguments:
        - number: int
            The number of transitions to be sampled.

        Returns:
        - transitions: list of tuples (eta, a, r', o', d', eta')
            The list of transitions sampled in the replay buffer.
        """
        return choices(self.buffer, k=number)
