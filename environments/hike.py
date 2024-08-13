import torch
import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

from random import choices, randrange


# Action space
ACTIONS = 0, 1, 2, 3
A_RIGHT, A_UP, A_LEFT, A_DOWN = ACTIONS


# Special states
START_POSITION = torch.tensor([-0.8, -0.8])
GOAL_POSITION = torch.tensor([0.8, 0.8])
LOWER_BOUND = torch.tensor([-1.0, -1.0])
UPPER_BOUND = torch.tensor([1.0, 1.0])


# Altitude function components
mvn_1 = mvn(mean=[0.0, 0.0], cov=[[0.005, 0.0], [0.0, 1.0]])
mvn_2 = mvn(mean=[0.0, -0.8], cov=[[1.0, 0.0], [0.0, 0.01]])
mvn_3 = mvn(mean=[0.0, 0.8], cov=[[1.0, 0.0], [0.0, 0.01]])
slope = torch.tensor([0.2, 0.2])


class MountainHike:
    """
    Creates a Mountain Hike environment
    ```
        +-----------------+ +1
        | - - - - - - - G |
        |        |        |
        |        |        |
        |        |        | 0
        |        |        |
        |        |        |
        | S - - - - - - - |
        +-----------------+ -1
       -1        0        +1
    ```

    Arguments:
        step_size: the norm of the expected displacement for any action
        transition_std: the transition noise standard deviation
        observation_std: the observation noise standard deviation
        bayes: whether to maintain the belief updated

    Example:
        >>> hike = MountainHike()
        >>> horizon = hike.horizon()
        >>> obs = hike.obs()
        >>> cum_rew = 0.0
        >>> for _ in range(horizon):
        ...     act = hike.exploration()
        ...     obs, rew, done = tmaze.step(act)
        ...     cum_rew += rew
        ...     if done:
        ...         break
    """

    gamma = 0.99
    observation_size = 1
    action_size = 4
    belief_type = "particles"

    def __init__(
        self,
        step_size=0.1,
        transition_std=0.05,
        observation_std=0.1,
        variations=None,
        bayes=False,
    ):

        self.bayes = bayes
        self.step_size = step_size
        self.transition_std = transition_std
        self.observation_std = observation_std
        self.variations = variations

        # Moves corresponding to actions
        self.M_RIGHT = torch.tensor([self.step_size, 0.0])
        self.M_UP = torch.tensor([0.0, self.step_size])
        self.M_LEFT = torch.tensor([-self.step_size, 0.0])
        self.M_DOWN = torch.tensor([0.0, -self.step_size])
        self.MOVES = torch.stack(
            [self.M_RIGHT, self.M_UP, self.M_LEFT, self.M_DOWN]
        )

    def horizon(self):
        """
        Returns the recommended truncation horizon for the environment.
        """
        manhattan_distance = 4.0
        if self.variations == "rotations":
            factor = 2
        elif self.variations == "permutations":
            factor = 3
        else:
            factor = 1
        return factor * int(manhattan_distance / self.step_size) * 2

    def exploration(self):
        """
        Returns a random action sampled according to the exploration policy
        of the T-Maze environment.
        """
        return choices(ACTIONS)[0]

    def reset(self):
        """
        Resets the environment for a new trajectory, by sampling a new
        initial state, and returns the initial observation provided by the
        environment.

        Returns:
        - observation: tensor
            The initial observation.
        """
        self._init_state()
        observation = self._observation()
        if self.bayes:
            self._init_belief(observation)
        return observation

    def step(self, action):
        """
        Samples a transition in the POMDP, according to the action chosen.

        Arguments:
        - action: int
            The action selected by the agent

        Returns:
        - observation: tensor
            the partial observation of the updated state.
        - reward: float
            the reward received for this transition.
        - done: bool
            whether the updated state is terminal.
        """
        self._check_action(action)

        self._transition(action)
        observation = self._observation()
        reward = self._reward(action)
        done = self._terminal()

        if self.bayes:
            self._update_belief(action, observation)

        return observation, reward, done

    def _check_action(self, action):
        """
        Checks if the action is valid.
        """
        if action < 0 or self.action_size <= action:
            size = self.action_size
            raise ValueError(f"The action should be in range [0, {size}[")

    def _init_state(self):
        """
        Samples an initial state according to p_0.
        """
        if self.variations is None:
            self.moves = self.MOVES
        elif self.variations == "rotations":
            rotation = np.roll(range(4), randrange(4))
            self.moves = self.MOVES[rotation, :]
        elif self.variations == "permutations":
            permutation = np.random.permutation(range(4))
            self.moves = self.MOVES[permutation, :]

        self.position = torch.clone(START_POSITION)

    def __altitude(self, position=None):
        """
        Returns the altitude at the current position (x, y).
        """
        if position is None:
            position = self.position.view(1, -1)

        mountains = [
            mvn_1.pdf(position),
            mvn_2.pdf(position),
            mvn_3.pdf(position),
        ]

        altitude = np.maximum.reduce(mountains)
        if np.isscalar(altitude):
            altitude = torch.tensor(altitude)
        else:
            altitude = torch.from_numpy(altitude)

        return (-torch.exp(-altitude)) + (position @ slope) - 0.02

    def _terminal(self, last=False):
        """
        Returns True if the current state is terminal.
        """
        position = self.last_position if last else self.position
        return np.linalg.norm(position - GOAL_POSITION) < self.step_size * 2

    def _transition(self, action):
        """
        Transitions to a new state using the action according to T(.|s, a).
        """
        self.last_position = torch.clone(self.position)

        # Terminal state are never updated
        if self._terminal():
            return

        noise = torch.randn(2) * self.transition_std
        self.position += self.moves[action] + noise
        self.position = torch.clamp(
            self.position,
            min=LOWER_BOUND,
            max=UPPER_BOUND,
        )

    def _reward(self, action):
        """
        Returns the reward resulting from the action according to R(s, a).
        """
        if self._terminal(last=True):
            return 0.0

        return self.__altitude()

    def _observation(self):
        """
        Samples an observation associated with the current state.
        """
        return self.__altitude() + torch.randn(1) * self.observation_std

    def _init_belief(self, observation, N=1000):
        """
        Initialises the belief b_0 according to the initial observation.
        """
        self.particles_positions = START_POSITION.repeat(N, 1)

        if self.variations is None:
            high = 1
        elif self.variations == "rotations":
            high = 4
        else:
            raise NotImplementedError

        self.particles_variations = torch.randint(low=0, high=high, size=(N,))

    def _update_belief(self, action, observation):
        """
        Updates the belief according to the action and new observation.
        """
        N = self.particles_positions.size(0)

        # Samples s_{t+1}
        noise = torch.randn(N, 2) * self.transition_std
        move_indices = (action + self.particles_variations) % 4
        self.particles_positions += self.MOVES[move_indices, :] + noise
        torch.clamp_(
            self.particles_positions,
            min=LOWER_BOUND,
            max=UPPER_BOUND,
        )

        # Computes importance weights
        weights = norm.pdf(
            observation,
            loc=self.__altitude(self.particles_positions),
            scale=self.observation_std,
        )
        weights /= weights.sum()

        # Resamples
        indices = np.random.choice(N, N, replace=True, p=weights)
        self.particles_positions = self.particles_positions[indices, :]
        self.particles_variations = self.particles_variations[indices]

    def get_belief(self):
        """
        Returns the current belief.
        """
        variations = self.particles_variations.float().unsqueeze(1)
        return (torch.cat((self.particles_positions, variations), dim=1),)


def main():
    """
    Allows to play with the Mountain Hike environment by manually selecting
    the actions.
    """
    import matplotlib.pyplot as plt

    env = MountainHike(bayes=True, variations="rotations")
    horizon = env.horizon()

    obs = env.reset()
    print(f"{obs=}")

    plt.ion()
    fig, ax = plt.subplots()
    ax.axis([-1, 1, -1, 1])
    plt.tight_layout()
    plt.show()

    # Subsample for plot
    idx = np.random.choice(np.arange(env.particles_positions.size(0)), 100)

    # Scatter plot particles
    ax.scatter(
        env.particles_positions[idx, 0],
        env.particles_positions[idx, 1],
    )
    ax.plot(env.position[0], env.position[1], "x", markersize=20, c="black")
    plt.draw()

    # Display variations particles
    print(np.unique(env.particles_variations, return_counts=True))

    cum_rew = 0.0
    for _ in range(horizon):

        act = None
        while True:
            try:
                act = int(input("action: "))
                obs, rew, done = env.step(act)
                break
            except ValueError:
                pass

        print(f"{act=}, {obs=}, {rew=}, {done=}")

        cum_rew += rew

        # Subsample for plot
        idx = np.random.choice(np.arange(env.particles_positions.size(0)), 100)

        # Scatter plot particles
        ax.scatter(
            env.particles_positions[idx, 0],
            env.particles_positions[idx, 1],
        )
        ax.plot(
            env.position[0],
            env.position[1],
            "x",
            markersize=20,
            c="black",
        )
        plt.draw()

        # Display variations particles
        print(np.unique(env.particles_variations, return_counts=True))

        if done:
            break


if __name__ == "__main__":
    main()
