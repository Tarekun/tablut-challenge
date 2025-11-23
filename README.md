# Tablut Challenge 2021

Project for the Tablut Competition, course _Foundations of Artificial Intelligence – Module 1_.
The project involves the creation of an agent that plays the Tablut game, a medieval asymmetric board game.

## Requirements
To execute the main script and play a game connected to the server, install install the dependencies listed in [requirements.txt]():

```bash
pip install -r requirements.txt
```

To execute the training phase and generate/train a model, install PyTorch in addition to the requirements:

```bash
pip install torch
```

## Usage
To run the player you can execute:

```bash
python main.py <WHITE|BLACK> <server_address>
```

## Architecture
The project contains several search algorithms that we tested and used to train the neural network. These include implementations of Minimax with Alpha-Beta pruning and Monte Carlo Tree Search.
These implementations differ in how they use the two heads of the neural network: __Policy__ and __Heuristic__.

You can find all of them in [profiles.py](), and the implementation of the actual algorithms in [search.py].

The neural network is trained using Reinforcement Learning with Self-Play, using the MCTS as a policy improvement operator, following the AlphaZero algorithm.
The neural network architecture is implemented in [model.py]()
The training pipeline is implemented in [train.py]()
To execute the training use:
```bash
python training_run.py
```

The final search algoritm the competed in the challenge is mcts_deep_model, which combines the policy head value with the PUCT score and uses the heuristic head value as a dropout for leaf states.  

