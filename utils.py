import torch
import wandb


class Namespace:
    """
    Creates a namespace from a dictionary.

    Arguments:
    - elements: dict
        The key and values in the namespace.
    """
    def __init__(self, elements):
        self.__dict__ |= elements


def generate_hiddens_and_beliefs(agent, environment, num_samples, epsilon=0.2,
                                 approximate=False):
    """
    Samples joint hidden states and beliefs using an epsilon-greedy policy
    based on the agent to sample the trajectories in the environment.

    Arguments
    - agent: Agent
        The agent whose epsilon-greedy policy is used.
    - environment: Environment
        The environment in which the trajectories are generated.
    - num_samples: int
        The number of hidden states and beliefs to generate.
    - epsilon: float
        The exploration rate of the epsilon-greedy policy.
    - approximate: bool
        Whether to use a faster approximation of the distributions where all
        beliefs and hidden states generated in any trajectory are returned.

    Returns
    - hiddens: tensor
        a batch of hidden states from the successive trajectories.
    - beliefs: tuple of tensors
        a batch of beliefs from the successive beliefs.
    """
    hiddens, beliefs = [], []
    while len(hiddens) < num_samples:
        _, hh, bb = agent.play(
            environment,
            epsilon=epsilon,
            return_hiddens=True,
            return_beliefs=True,
        )

        # TODO: this should be modified to sample a time step first
        # TODO: this should be modified to allow sampling past terminal states

        if approximate:
            for h, b in zip(hh, bb):
                hiddens.append(h)
                beliefs.append([bi for bi in b])
        else:
            t = torch.randint(len(hh), ()).item()
            hiddens.append(hh[t])
            beliefs.append([bi for bi in bb[t]])

    hiddens = hiddens[:num_samples]
    beliefs = beliefs[:num_samples]

    tuple_of_beliefs = [list() for _ in range(len(beliefs[0]))]
    for belief in beliefs:
        for i, b in enumerate(belief):
            tuple_of_beliefs[i].append(b)

    return torch.stack(hiddens), tuple(map(torch.stack, tuple_of_beliefs))


def print_stats(stats):
    """
    Print all key/value pairs in the dictionary, with colors (ANSI escape
    sequences) according to the sign.

    Arguments
        stats: statistics to be displayed.
    """
    for k, v in stats.items():

        postfix = '\033[0m'

        if v < 0.0:
            prefix = '\033[1;31m'
        elif v > 0.0:
            prefix = '\033[1;32m'
        else:
            prefix = '\033[1;1m'

        print(f'\t{k}: {prefix}{v:.4f}{postfix}')


def get_run_statistic(run_id):
    """
    Returns the configuration associated with the specified run.

    Arguments:
    - run_id: str
        The id of the considered run.

    Returns:
    - config: dict
        The configuration of the considered run.
    """
    api = wandb.Api()
    run = api.run(f'gsprd/belief-train/{run_id}')

    if run.state != 'finished':
        raise ValueError(
            f'The training session {run_id} has not finished its execution.')

    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    summary = {
        'episodes': run.summary['train/episode'],
        'transitions': run.summary['train/num_transitions'],
    }

    return Namespace(config | summary)
