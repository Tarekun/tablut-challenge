# search.py

This file contains configuration functions for the implemented search algorithms.
Search algorithms are usually parametric in some of their sub components, such as search stopping criterions or action selection functions. Moreover one wants the rest of the code to be agnostic on the specific search procedure and only wants a function that given the current `GameState` return the next one picked by such algorithm.
As such, this takes a functional style approach where we have one function that takes parameter to the search algorithms and returns a function `GameState` -> `GameState` that is an instance of this search algorith with those fixed parameters.

## Alpha/Beta search

Implements minimax search with alpha-beta pruning. This algorithm explores the game tree to find the optimal move by evaluating all possible moves up to a certain depth, using alpha-beta pruning to eliminate branches that won't affect the final decision.

Parameters
----------
heuristic : GameState -> [-1, 1]
    State estimation function that evaluates a GameState and returns a
    float between -1 and 1, where positive values favor the maximizing player
    and negative values favor the minimizing player.
stop_criterion : [GameState, int] -> bool
    Boolean function that determines whether to stop search at a node.
    Takes current GameState and current search depth as input and returns
    True if search should stop (leaf node or depth limit reached).
move_sequence : GameState -> list[GameState]
    Policy function that generates and orders possible moves from a given
    GameState. Proper ordering is crucial for effective alpha-beta pruning.

Returns
-------
GameState -> GameState
    Function that takes current GameState as input and returns the
    optimal next GameState according to the minmax search with alpha-beta pruning.

## Monte Carlo Tree Search (MCTS)

WIP

