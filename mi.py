import wandb
import torch

from environments.tmaze import TMaze
from environments.hike import MountainHike
from environments.irrelevant import Irrelevant

from agents.drqn import DRQN

from mine.mine import MutualInformationNeuralEstimator
from utils import generate_hiddens_and_beliefs, get_run_statistic

from argparse import ArgumentParser


def main(args):

    # Retrieve training arguments
    train_args = get_run_statistic(args.train_id)

    # Merge configurations
    config = vars(train_args) | vars(args)

    # Initialize logging
    wandb.init(
        project='belief-mi',
        name=args.name,
        config=config,
        save_code=True)
    config = wandb.config

    # Save all packages
    wandb.save('*.py')
    wandb.save('agents/*.py')
    wandb.save('environments/*.py')

    # Initialize environment
    if train_args.environment == 'tmaze':
        environment = TMaze(
            bayes=True,
            length=train_args.length,
            stochasticity=train_args.stochasticity)
    elif train_args.environment == 'hike':
        environment = MountainHike(
            bayes=True,
            variations=train_args.variations)
    else:
        environment = train_args.environment
        raise NotImplementedError(f'Unknown environment {environment}')

    # Add irrelevant variables
    if train_args.irrelevant != 0:
        environment = Irrelevant(environment, state_size=train_args.irrelevant,
                                 bayes=True)

    # Initialize agent
    if train_args.algorithm == 'drqn':
        network_kwargs = {
            'num_layers': train_args.num_layers,
            'hidden_size': train_args.hidden_size}

        agent = DRQN(
            cell=train_args.cell,
            action_size=environment.action_size,
            observation_size=environment.observation_size,
            **network_kwargs)
    else:
        raise NotImplementedError(f'Unknown algorithm {args.algorithm}')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print('Device:', device)

    # Evaluate MI on all available weights
    estimates = []
    print(config.episodes)
    for episode in range(0, config.episodes + 1, args.mine_period):

        agent.load(args.train_id, episode=episode)

        hiddens, beliefs = generate_hiddens_and_beliefs(
            agent, environment, num_samples=args.mine_num_samples,
            epsilon=args.epsilon, approximate=args.approximate)

        if len(beliefs) == 1 and args.belief_part is not None:
            print("No estimation of the MI on belief parts")
            return

        representation_sizes = []
        belief_sizes = []

        for belief_part in beliefs:

            belief_sizes.append(belief_part.size(-1))

            if belief_part.ndim == 2:
                representation_sizes.append(None)
            elif belief_part.ndim == 3:
                representation_sizes.append(args.representation_size)
            else:
                raise ValueError('Expected 2 or 3 dimensions for the belief '
                                 'representation')

        mine = MutualInformationNeuralEstimator(
            hs_sizes=hiddens.size(-1), belief_sizes=belief_sizes,
            hidden_size=args.mine_hidden_size, num_layers=args.mine_num_layers,
            alpha=args.mine_alpha, representation_sizes=representation_sizes,
            belief_part=args.belief_part, device=device)

        mine.optimize(
            hiddens, beliefs, num_epochs=args.mine_num_epochs,
            logger=wandb.log, learning_rate=args.mine_learning_rate,
            batch_size=args.mine_batch_size, lambd=args.mine_lambda,
            valid_size=args.valid_size)

        mine.save(wandb.run.id, episode=episode)

        if not args.train_set:
            hiddens, beliefs = generate_hiddens_and_beliefs(
                agent, environment, num_samples=args.mine_num_samples,
                epsilon=args.epsilon, approximate=args.approximate)

        mi = mine.estimate(hiddens, beliefs)

        if args.belief_part is None:
            mi_part_key = 'mine_estimate/mi'
        else:
            mi_part_key = f'mine_estimate/mi-{args.belief_part}'

        # Do not mix logging when epsilon
        if args.epsilon != 0.0:
            mi_part_key += f'-{args.epsilon}'

        estimate = {'train/episode': episode, mi_part_key: mi}
        wandb.log(estimate)
        estimates.append(estimate)

        print(f'Episode {episode}\n\tMI = {mi}')

    wandb.finish()

    wandb.init(project='belief-train', id=args.train_id, resume='must')
    for estimate in estimates:
        wandb.log(estimate)
    wandb.finish()


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Estimate MI for a certain training session',
    )
    parser.add_argument('name', type=str, nargs='?', default=None)

    # Run id
    parser.add_argument('train_id', type=str)

    # MINE estimator
    parser.add_argument('--mine-num-samples', type=int, default=10000)
    parser.add_argument('--mine-num-layers', type=int, default=2)
    parser.add_argument('--mine-hidden-size', type=int, default=256)
    parser.add_argument('--mine-alpha', type=float, default=0.01)
    parser.add_argument('--mine-num-epochs', type=int, default=100)
    parser.add_argument('--mine-batch-size', type=int, default=1024)
    parser.add_argument('--mine-learning-rate', type=float, default=1e-3)
    parser.add_argument('--mine-lambda', type=float, default=0.0)
    parser.add_argument('--mine-period', type=int, default=100)
    parser.add_argument('--representation-size', type=int, default=16)
    parser.add_argument('--belief-part', type=int, default=None)

    parser.add_argument('--valid-size', type=float, default=0.2)
    parser.add_argument('--train-set', action='store_true')
    parser.add_argument('--approximate', action='store_true')

    parser.add_argument('--epsilon', type=float, default=0.0)

    # Parse command line arguments
    args = parser.parse_args()
    print('\n'.join(f'\033[90m{k}=\033[0m{v}' for k, v in vars(args).items()))

    main(args)
