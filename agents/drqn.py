import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnns import RNNS
from .memory import Trajectory, ReplayBuffer
from utils import print_stats

from random import random


class DRQN:
    """
    Deep Recurrent Q-Network reinforcement learning agent.

    Arguments:
    - cell: str
        The name of the recurrent cell.
    - action_size: int
        The number of discrete actions in the environment.
    - observation_size: int
        The dimension of the observation in the environment.
    - **network_kwargs: dict
        Additional arguments for the recurrent cell.
    """

    def __init__(self, cell, action_size, observation_size, **network_kwargs):

        self.action_size = action_size
        self.observation_size = observation_size

        # Initialize Q network and target Q network
        input_size = action_size + observation_size
        self.Q = RNNS[cell](
            input_size=input_size,
            output_size=action_size,
            **network_kwargs,
        )
        self.Q_tar = RNNS[cell](
            input_size=input_size,
            output_size=action_size,
            **network_kwargs,
        )

    def eval(self, environment, num_rollouts):
        """
        Evaluates the (discounted) cumulative return over a certain number of
        rollouts.

        Arguments:
        - environment: Environment
            The environment on which the agent is evaluated.
        - num_rollouts: int
            The number of episodes over which the returns are averaged.

        Returns:
        - mean_return: float
            The average empirical cumulative return
        - mean_disc_return: float
            The average empirical discounted cumulative return
        """
        sum_returns, disc_returns = 0.0, 0.0

        for _ in range(num_rollouts):

            trajectory, = self.play(environment, epsilon=0.0)

            sum_returns += trajectory.get_cumulative_reward()
            disc_returns += trajectory.get_cumulative_reward(environment.gamma)

        mean_return = sum_returns / num_rollouts
        mean_disc_return = disc_returns / num_rollouts

        return mean_return, mean_disc_return

    def play(
        self,
        environment,
        epsilon,
        return_hiddens=False,
        return_beliefs=False,
    ):
        """
        Plays a trajectory in the environment with the current policy of
        the agent and some noise.

        Arguments:
        - environment: Environment
            The environment on which to play.
        - epsilon: float
            The exploration rate at each time step.
        - return_hiddens: bool
            Whether to return the hidden states along with the trajectory.
        - return_beliefs: bool
            Whether to return the beliefs along with the trajectory.

        Returns:
        - trajectory: Trajectory
            The trajectory resulting from the interaction with the environment.
        - hiddens: list
            The list of flattened hidden states at each time step of the trajectory.
        - beliefs: list
            The list of beliefs at each time step of the trajectory.
        """
        hiddens, beliefs = [], []

        o = environment.reset()
        trajectory = Trajectory(
            environment.action_size,
            environment.observation_size,
        )

        trajectory.add(None, None, o)

        hidden_states = None
        for _ in range(environment.horizon()):

            tau_t = trajectory.get_last_observed().view(1, 1, -1)
            with torch.no_grad():
                values, hidden_states = self.Q(tau_t, hidden_states)

            if return_hiddens:
                hiddens.append(hidden_states[0].detach().flatten().clone())

            if return_beliefs:
                beliefs.append(environment.get_belief())

            if random() < epsilon:
                a = environment.exploration()
            else:
                a = values.flatten().argmax().item()

            o, r, d = environment.step(a)

            trajectory.add(a, r, o, terminal=d)

            if d:
                break

        return_values = (trajectory,)

        if return_hiddens:
            return_values += (hiddens,)
        if return_beliefs:
            return_values += (beliefs,)

        return return_values

    def _targets(self, transitions, gamma):
        """
        Computes from a set of transitions, a list of inputs sequences,
        of masks indicating the considered time steps, and targets for those
        time steps computed with the target network.

        Arguments:
        - transitions: list of tuples (eta, a, r', o', d', eta')
            List of transitions
        - gamma: float:
            Discount factor for the targets

        Returns:
        - inputs: list
            List of histories
        - targets: list
            List of target values
        - masks: list
            List of time steps for which the loss should be computed
        """
        inputs, targets, masks = [], [], []

        # TODO: no loop but use padding
        for seq_bef, a, r, o, d, seq_aft in transitions:

            target = torch.tensor(r)

            if not d:
                with torch.no_grad():
                    Q_next, _ = self.Q_tar(seq_aft.unsqueeze(1))

                target += gamma * Q_next[-1, 0, :].max()

            target = target.view(1, -1)
            mask = torch.zeros(seq_bef.size(0), self.action_size,
                               dtype=torch.bool)
            mask[-1, a] = True

            inputs.append(seq_bef)
            targets.append(target)
            masks.append(mask)

        return inputs, targets, masks

    def _loss(self, inputs, targets, masks):
        """
        Computes the MSE loss between the predictions at the time steps
        specified by the masks and the targets.

        Arguments:
        - inputs: list
            List of histories
        - targets: list
            List of target values
        - masks: list
            List of time steps for which the loss should be computed

        Returns:
        - loss: float
            The loss resulting from the predictions and targets
        """
        inputs = nn.utils.rnn.pad_sequence(inputs)
        targets = nn.utils.rnn.pad_sequence(targets)
        masks = nn.utils.rnn.pad_sequence(masks)

        outputs, _ = self.Q(inputs)

        return F.mse_loss(outputs.transpose(0, 1)[masks.transpose(0, 1)],
                          targets.transpose(0, 1).flatten())

    def train(
        self,
        environment,
        run_id,
        logger,
        num_episodes,
        batch_size,
        learning_rate,
        num_gradient_steps,
        target_period,
        eval_period,
        num_rollouts,
        epsilon,
        buffer_capacity,
        fill_buffer=False,
    ):
        """
        Trains the reinforcement learning agent on the specified environment

        Arguments:
        - environment: Environment
            The environment on which to train the agent.
        - logger: function
            The function to call for logging the training statistics.
        - num_episodes: int
            The number of episodes to generate
        - batch_size: int
            The number of transitions in each minibatch
        - learning_rate: float
            The learning rate used in the Adam optimizer
        - num_gradient_steps: int
            The number of gradients steps made at each episode
        - target_period: int
            The period at which the target is updated in number of episodes
        - eval_period: the period at which the network is evaluated in number
            of episodes
        - num_rollouts: int
            The number of episodes for the evaluation
        - epsilon: float
            The exploration rate
        """
        optim = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        num_transitions = 0

        # Initialise replay buffer
        buffer = ReplayBuffer(buffer_capacity)

        # Eventually fill replay buffer
        if fill_buffer:
            while not buffer.is_full:
                trajectory, = self.play(environment, epsilon=1.0)
                buffer.add(trajectory)

        for episode in range(num_episodes):

            # Evaluate and save weights
            if episode % eval_period == 0:
                mean_return, mean_disc_return = self.eval(
                    environment,
                    num_rollouts,
                )

                print(f'Episode {episode:04d}')
                print_stats(
                    {'return': mean_return, 'disc_return': mean_disc_return}
                )

                logger(
                    {
                        'train/episode': episode,
                        'train/return': mean_return,
                        'train/disc_return': mean_disc_return,
                        'train/num_transitions': num_transitions,
                    }
                )

                self.save(run_id, episode=episode)

            # Update target
            if episode % target_period == 0:
                self.Q_tar.load_state_dict(self.Q.state_dict())

            # Generate trajectory
            trajectory, = self.play(environment, epsilon=epsilon)
            buffer.add(trajectory)
            num_transitions += trajectory.num_transitions

            # Optimize Q-network
            for _ in range(num_gradient_steps):

                transitions = buffer.sample(batch_size)

                inputs, targets, masks = self._targets(
                    transitions,
                    environment.gamma,
                )
                loss = self._loss(inputs, targets, masks)

                optim.zero_grad()
                loss.backward()
                optim.step()

        # Log and save last results
        mean_return, mean_disc_return = self.eval(environment, num_rollouts)

        print('Final evaluation')
        print_stats({'return': mean_return, 'disc_return': mean_disc_return})

        logger({'train/episode': episode,
                'train/return': mean_return,
                'train/disc_return': mean_disc_return,
                'train/num_transitions': num_transitions})

        self.save(run_id, episode=num_episodes)

    def save(self, run_id, episode=None):
        """
        Saves the weights of a trained agent on disk. Does not save the
        optimizer's momenta.

        Argument:
            run_id: str
                Unique identifier for this training run
            episode: int
                The episode at which the agent was saved
        """
        os.makedirs('weights', exist_ok=True)
        path = f'weights/{run_id}-{episode}-{{}}.pth'
        torch.save(self.Q.state_dict(), path.format('Q'))
        torch.save(self.Q_tar.state_dict(), path.format('Q_tar'))

    def load(self, run_id, episode=None):
        """
        Loads the weights of an agent saved on disk. Does not load the
        optimizer's momenta.

        Argument:
            run_id: str
                Unique identifier for this training run
            episode: int
                The episode at which the agent was saved
        """
        path = f'weights/{run_id}-{episode}-{{}}.pth'
        self.Q.load_state_dict(torch.load(path.format('Q')))
        self.Q_tar.load_state_dict(torch.load(path.format('Q_tar')))
