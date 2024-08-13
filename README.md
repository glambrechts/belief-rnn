# Recurrent Networks, Hidden States and Beliefs in Partially Observable Environments

Official implementation of the paper "Recurrent Networks, Hidden States and Beliefs in Partially Observable Environments".

If you find this code useful, please reference in your paper:
```bibtex
@article{
    lambrechts2022recurrent,
    title={Recurrent Networks, Hidden States and Beliefs in Partially Observable Environments},
    author={Gaspard Lambrechts and Adrien Bolland and Damien Ernst},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2022},
    url={https://openreview.net/forum?id=dkHfV3wB2l},
}
```

To learn mode:

- [Research paper](https://arxiv.org/abs/2208.03520)
- [Poster summary](https://people.montefiore.uliege.be/lambrechts/pdf/belief-rnn-poster.pdf)

## Usage

Launch DRQN trainings with:
```bash
python train.py --num-episodes 5000 tmaze
python train.py --num-episodes 5000 --irrelevant 2 tmaze
python train.py --num-episodes 5000 --irrelevant 2 tmaze --length 50 --stochasticity 0.2

python train.py --num-episodes 5000 hike
python train.py --num-episodes 5000 hike --variations rotations
```
and the logger (wandb) will display the training id (e.g., q3yszobj).

Afterwards, compute the mutual information between the environment belief and the agent state throughout training with:
```bash
python mi.py q3yszobj
```
