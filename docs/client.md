# client.py

This file contains the code that implements our client to the provided Java server. It includes functions to read/write to the socket (\_read_n_bytes, \_read_string_from_stream, \_write_string_to_stream), to subscribe to the server (`initialize_connection`), to parse a game state (`parse_state`), but users of this file will mainly use the simple function `play_game`: given the `player` we're playing as, the `name` used for our player, the server `ip`, a `search_algorithm`: `GameState` -> `GameState` it subscribes to a game and plays with the configured search function. Optionally one can set `track=True` to get back the tracking of the game, a list of (list\[GameState\], int) which contains the list of played states and the outcome of the game relative to the given player.

## API Reference
