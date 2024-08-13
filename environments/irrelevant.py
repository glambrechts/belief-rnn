import torch


class Irrelevant:
    """
        Wrapper around an environment that adds a partially observable decision
    process that is irrelevant to optimal control. More specifically, this
    process does not intervene in the rewards.

    This process has linear dynamics and Gaussian noise.
    """

    def __init__(self, environment, state_size, bayes=True):

        self.environment = environment

        self.state_size = state_size
        self.bayes = bayes

        self.F = torch.eye(self.state_size, self.state_size)
        self.B = torch.zeros(self.state_size, self.action_size)

        self.H = torch.eye(self.state_size, self.state_size)
        self.Q = torch.eye(self.state_size, self.state_size) * 0.1
        self.R = torch.eye(self.state_size, self.state_size)

        self.I = torch.eye(self.state_size)

    @property
    def gamma(self):
        return self.environment.gamma

    @property
    def observation_size(self):
        return self.environment.observation_size + self.state_size

    @property
    def action_size(self):
        return self.environment.action_size

    @property
    def belief_type(self):
        return self.environment.belief_type

    def horizon(self):
        return self.environment.horizon()

    def exploration(self):
        return self.environment.exploration()

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
        self.environment._check_action(action)

    def _init_state(self):
        """
        Samples an initial state according to p_0.
        """
        mvn = torch.distributions.MultivariateNormal(
            torch.zeros(self.state_size), covariance_matrix=self.Q
        )
        self.state = mvn.sample()

        self.environment._init_state()

    def _terminal(self):
        """
        Returns True if the current state is terminal.
        """
        return self.environment._terminal()

    def _transition(self, action: int):
        """
        Transitions to a new state using the action according to T(.|s, a).
        """
        if self._terminal():
            return

        mvn = torch.distributions.MultivariateNormal(
            self.F @ self.state + self.B[:, action],
            covariance_matrix=self.Q,
        )
        self.state = mvn.sample()

        self.environment._transition(action)

    def _reward(self, action: int):
        """
        Returns the reward resulting from the action according to R(s, a).
        """
        return self.environment._reward(action)

    def _observation(self):
        """
        Samples an observation associated with the current state.
        """
        mvn = torch.distributions.MultivariateNormal(
            self.H @ self.state, covariance_matrix=self.R
        )
        observation = mvn.sample()

        return torch.cat((observation, self.environment._observation()))

    def __belief_correction(self, observation: torch.FloatTensor):
        y = observation - self.H @ self.mean
        S = self.H @ self.cov @ self.H.T + self.R
        K = self.cov @ self.H.T @ torch.inverse(S)
        self.mean += K @ y
        self.cov = (self.I - K @ self.H) @ self.cov

    def _init_belief(self, observation: torch.FloatTensor):
        """
        Initialises the belief b_0 according to the initial observation.
        """
        self.mean = torch.zeros(self.state_size)
        self.cov = self.Q

        self.__belief_correction(observation[: self.state_size])

        self.environment._init_belief(observation[self.state_size:])

    def _update_belief(self, action: int, observation: torch.FloatTensor):
        """
        Updates the belief according to the action and new observation.
        """
        self.mean = self.F @ self.mean + self.B[:, action]
        self.cov = self.F @ self.cov @ self.F.T + self.Q

        self.__belief_correction(observation[: self.state_size])

        self.environment._update_belief(action, observation[self.state_size:])

    def get_belief(self):
        """
        Returns the current belief.
        """
        return (
            torch.cat((self.mean, self.cov.flatten())),
        ) + self.environment.get_belief()


def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    import numpy as np
    import matplotlib.transforms as transforms
    from matplotlib.patches import Ellipse

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def main():
    """
    Allows to play with the Irrelevant environment with the T-Maze as
    underlying environment.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from .tmaze import TMaze

    env = TMaze(bayes=True, stochasticity=0.2, length=20)
    env = Irrelevant(env, state_size=2, bayes=True)

    plt.ion()
    fig, ax = plt.subplots()
    ax.axis([-10, 10, -10, 10])
    plt.tight_layout()
    plt.show()

    horizon = env.horizon()
    obs = env.reset()

    print(f"{obs=}")
    print(f"{env.state=}")
    print(*env.get_belief(), sep="\n")

    observations = [obs.tolist()[:2]]
    states = [env.state.tolist()]
    beliefs = [env.get_belief()[0].tolist()[:2]]

    (obs_line,) = ax.plot(*np.transpose(observations), label="observation")
    (state_line,) = ax.plot(*np.transpose(states), label="state")
    (belief_line,) = ax.plot(*np.transpose(beliefs), label="belief")

    mean = beliefs[-1]
    cov = np.array(env.get_belief()[0].tolist()[2:]).reshape(2, 2)
    confidence_ellipse(mean, cov, ax=ax, n_std=3.0, facecolor="C2", alpha=0.2)

    cum_rew = 0.0
    for _ in range(horizon):
        act = int(input("action: "))

        obs, rew, done = env.step(act)
        print(f"{act=}, {obs=}, {rew=}, {done=}")
        print(f"{env.state=}")
        print(*env.get_belief(), sep="\n")

        observations.append(obs.tolist()[:2])
        states.append(env.state.tolist())
        beliefs.append(env.get_belief()[0].tolist()[:2])

        obs_line.set_data(*np.transpose(observations))
        state_line.set_data(*np.transpose(states))
        belief_line.set_data(*np.transpose(beliefs))

        mean = beliefs[-1]
        cov = np.array(env.get_belief()[0].tolist()[2:]).reshape(2, 2)
        confidence_ellipse(
            mean,
            cov,
            ax=ax,
            n_std=1.0,
            facecolor="C2",
            alpha=0.2,
        )

        plt.draw()

        cum_rew += rew

        if done:
            break


if __name__ == "__main__":
    main()
