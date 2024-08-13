import torch
import random

from math import ceil


# Observation space
OBSERVATIONS = torch.eye(4)
O_UP, O_DOWN, O_CORRIDOR, O_CROSSROAD = OBSERVATIONS

# Action space
ACTIONS = (0, 1, 2, 3)
A_RIGHT, A_UP, A_LEFT, A_DOWN = ACTIONS


class TMaze:
    """
        A T-Maze environment.
    ```
                    |?| -> L + 1
        |0|.|.|.|.|.|L|
                    |?| -> L + 2
    ```

    Arguments:
    - length: the T-Maze corridor length.
    - stochasticity: the transition stochasticity rate.
    - bayes: whether to maintain the belief updated

    Example:
        >>> tmaze = TMaze(10, stochasticity=0.2)
        >>> horizon = tmaze.horizon()
        >>> obs = tmaze.reset()
        >>> cum_rew = 0.0
        >>> for _ in range(horizon):
        ...     act = tmaze.exploration()
        ...     obs, rew, done = tmaze.step(act)
        ...     cum_rew += rew
        ...     if done:
        ...         break
    """

    gamma = 0.98
    observation_size = 4
    action_size = 4
    belief_type = "exact"

    def __init__(self, length=10, stochasticity=0.0, bayes=False):

        self.length = int(length)
        self.stochasticity = float(stochasticity)
        self.bayes = bayes

        self.T = self._transition_model()
        self.O = self._observation_model()

    def _observation_model(self):
        """
        Returns the observation model O such that O[o][s] is the probability
        O(o|s) to observe o in state s.
        """
        O = {}

        for o in range(len(OBSERVATIONS)):
            O[o] = torch.zeros(2 * (self.length + 3))

        for i in range(2 * (self.length + 3)):

            goal_up = 1 - int(i / (self.length + 3))
            position = i % (self.length + 3)

            if 0 < position < self.length:

                O[O_CORRIDOR.argmax().item()][i] = 1.0

            elif position == 0:

                if goal_up:
                    O[O_UP.argmax().item()][i] = 1.0
                else:
                    O[O_DOWN.argmax().item()][i] = 1.0

            elif position == self.length:

                O[O_CROSSROAD.argmax().item()][i] = 1.0

            elif self.length + 1 <= position <= self.length + 2:

                O[O_CROSSROAD.argmax().item()][i] = 1.0

        return O

    def _transition_model(self):
        """
        Returns the transition model O such that T[a][s,s'] is the
        probability T(s'|s,a) to transition from state s to state s' when
        selecting action a.
        """
        T = {}

        for ACTION in ACTIONS:
            T[ACTION] = torch.zeros(
                2 * (self.length + 3),
                2 * (self.length + 3),
            )

        for i in range(2 * (self.length + 3)):

            position = i % (self.length + 3)

            if 0 < position < self.length:

                T[A_RIGHT][i - 1, i] = self.stochasticity / 4
                T[A_RIGHT][i, i] = 2 * self.stochasticity / 4
                T[A_RIGHT][i + 1, i] = 1 - 3 * self.stochasticity / 4

                T[A_UP][i - 1, i] = self.stochasticity / 4
                T[A_UP][i, i] = 1 - 2 * self.stochasticity / 4
                T[A_UP][i + 1, i] = self.stochasticity / 4

                T[A_LEFT][i - 1, i] = 1 - 3 * self.stochasticity / 4
                T[A_LEFT][i, i] = 2 * self.stochasticity / 4
                T[A_LEFT][i + 1, i] = self.stochasticity / 4

                T[A_DOWN][i - 1, i] = self.stochasticity / 4
                T[A_DOWN][i, i] = 1 - 2 * self.stochasticity / 4
                T[A_DOWN][i + 1, i] = self.stochasticity / 4

            elif position == 0:

                T[A_RIGHT][i + 1, i] = 1 - 3 * self.stochasticity / 4
                T[A_RIGHT][i, i] = 3 * self.stochasticity / 4

                for A_OTHER in (A_UP, A_LEFT, A_DOWN):
                    T[A_OTHER][i, i] = 1 - 3 * self.stochasticity / 4
                    T[A_OTHER][i + 1, i] = self.stochasticity / 4

            elif position == self.length:

                T[A_RIGHT][i - 1, i] = self.stochasticity / 4
                T[A_RIGHT][i, i] = 1 - 3 * self.stochasticity / 4
                T[A_RIGHT][i + 1, i] = self.stochasticity / 4
                T[A_RIGHT][i + 2, i] = self.stochasticity / 4

                T[A_UP][i - 1, i] = self.stochasticity / 4
                T[A_UP][i, i] = self.stochasticity / 4
                T[A_UP][i + 1, i] = 1 - 3 * self.stochasticity / 4
                T[A_UP][i + 2, i] = self.stochasticity / 4

                T[A_LEFT][i - 1, i] = 1 - 3 * self.stochasticity / 4
                T[A_LEFT][i, i] = self.stochasticity / 4
                T[A_LEFT][i + 1, i] = self.stochasticity / 4
                T[A_LEFT][i + 2, i] = self.stochasticity / 4

                T[A_DOWN][i - 1, i] = self.stochasticity / 4
                T[A_DOWN][i, i] = self.stochasticity / 4
                T[A_DOWN][i + 1, i] = self.stochasticity / 4
                T[A_DOWN][i + 2, i] = 1 - 3 * self.stochasticity / 4

            elif self.length + 1 <= position <= self.length + 2:

                for ACTION in ACTIONS:
                    T[ACTION][i, i] = 1.0

        return T

    def horizon(self):
        """
        Returns the recommended truncation horizon for the environment.
        """
        return ceil(
            self.length / ((1.0 - self.stochasticity) * (1 / 2 - 1 / 6))
        )

    def exploration(self):
        """
        Returns a random action sampled according to the exploration policy
        of the T-Maze environment.
        """
        return random.choices(ACTIONS, weights=(1 / 2, 1 / 6, 1 / 6, 1 / 6))[0]

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
        if action < 0 or self.action_size <= action:
            size = self.action_size
            raise ValueError(f"The action should be in range [0, {size}[")

    def _init_state(self):
        """
        Samples an initial state according to p_0.
        """
        self.goal_up = random.random() < 0.5
        self.last_position = -1
        self.position = 0

    def _terminal(self, last=False):
        """
        Returns True if the current state is terminal.
        """
        position = self.last_position if last else self.position
        return self.length + 1 <= position <= self.length + 2

    def _transition(self, action):
        """
        Transitions to a new state using the action according to T(.|s, a).
        """
        self.last_position = self.position

        # Terminal states (L + 1 and L + 2) are never updated
        if self._terminal():
            return

        # If the environment is stochastic, the 'effective' action is random,
        # with a certain probability
        if random.random() < self.stochasticity:
            action = random.choice((A_RIGHT, A_UP, A_LEFT, A_DOWN))

        # Transition from the corridor
        if 0 < self.position < self.length:
            if action == A_RIGHT:
                self.position += 1
            elif action == A_LEFT:
                self.position -= 1

        # Transition from the first cell
        elif self.position == 0:
            if action == A_RIGHT:
                self.position += 1

        # Transition from the crossroad
        elif self.position == self.length:
            if action == A_LEFT:
                self.position = self.length - 1
            elif action == A_UP:
                self.position = self.length + 1
            elif action == A_DOWN:
                self.position = self.length + 2

        # Unexpected
        else:
            raise ValueError("Unexpected state")

    def _reward(self, action):
        """
        Returns the reward resulting from the action according to R(s, a).
        """
        # If the previous state was terminal, then the reward is zero.
        if self._terminal(last=True):
            return 0.0

        # Otherwise, if the agent hasn't move, it has bumped onto a wall.
        elif self.last_position == self.position:
            return -0.1

        # Otherwise, if it still is in the corridor or at the crossroad, the
        # reward is 0.0
        elif 0 <= self.position <= self.length:
            return 0.0

        # Finally, if it reaches a terminal state
        elif self.length + 1 <= self.position <= self.length + 2:
            if self.goal_up and self.position == self.length + 1:
                return 4.0
            elif not self.goal_up and self.position == self.length + 2:
                return 4.0
            else:
                return -0.1

        # Unexpected
        else:
            raise ValueError("Unexpected state")

    def _observation(self):
        """
        Samples an observation associated with the current state.
        """
        if self.position == 0:
            return O_UP if self.goal_up else O_DOWN
        elif 0 < self.position < self.length:
            return O_CORRIDOR
        elif self.length <= self.position <= self.length + 2:
            return O_CROSSROAD
        else:
            raise ValueError("Unexpected state")

    def _init_belief(self, observation):
        """
        Initialises the belief b_0 according to the initial observation.
        """
        self.belief = torch.zeros(2 * (self.length + 3))
        self.belief[0] = 0.5
        self.belief[self.length + 3] = 0.5

        O = self.O[observation.argmax().item()]
        self.belief = O * self.belief
        self.belief /= self.belief.sum()

    def _update_belief(self, action, observation):
        """
        Updates the belief according to the action and new observation.
        """
        T = self.T[action]
        O = self.O[observation.argmax().item()]
        self.belief = O * (T @ self.belief)
        self.belief /= self.belief.sum()

    def get_belief(self):
        """
        Returns the current belief.
        """
        return (self.belief.clone(),)


def main():
    """
    Allows to play with the T-Maze environment by manually selecting the
    actions.
    """
    env = TMaze(bayes=True, stochasticity=0.2, length=5)
    horizon = env.horizon()

    obs = env.reset()
    print(f"{obs=}")
    print(env.belief)

    cum_rew = 0.0
    for _ in range(horizon):

        act = int(input("action: "))

        obs, rew, done = env.step(act)
        print(f"{act=}, {obs=}, {rew=}, {done=}")
        print(env.belief)

        cum_rew += rew

        if done:
            break


if __name__ == "__main__":
    main()
