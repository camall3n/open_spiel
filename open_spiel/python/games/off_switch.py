# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Off Switch Game implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum

import numpy as np

from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel


class Action(enum.IntEnum):
  WAIT = 0
  OFF = 1
  EXEC = 2

class Player(enum.IntEnum):
  ROBOT = 0
  HUMAN = 1

_NUM_PLAYERS = len(Player)
_VALUES = (-1, 1)
_GAME_TYPE = pyspiel.GameType(
    short_name="python_off_switch",
    long_name="Python Off Switch Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.IDENTICAL,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=len(_VALUES),
    num_players=_NUM_PLAYERS,
    min_utility=-2.0,
    max_utility=2.0,
    utility_sum=0.0,
    max_game_length=3)  # e.g. [Wait, Wait, Go]


class OffSwitchGame(pyspiel.Game):
  """A Python version of the off-switch game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return OffSwitchState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return OffSwitchObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)

  def max_chance_nodes_in_history(self):
    return 1


class OffSwitchState(pyspiel.State):
  """A python version of the Off Switch Game state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.human_value = None
    self.action_history = []
    self.score = 0.0
    self._game_over = False
    self._next_player = 0 # 0: robot; 1: human

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self.human_value is None:
      return pyspiel.PlayerId.CHANCE
    else:
      return self._next_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    if player == Player.ROBOT:
      if len(self.action_history) == 0:
        return [Action.WAIT, Action.OFF, Action.EXEC]
      else:
        return [Action.OFF, Action.EXEC]
    else:# (player == Player.HUMAN)
      return [Action.WAIT, Action.OFF]

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = sorted(list(range(len(_VALUES))))
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      print('action:', action)
      self.human_value = _VALUES[action]
    else:
      self.action_history.append(action)
      if action == Action.EXEC:
        self.score = self.human_value
        self._game_over = True
      elif action == Action.OFF:
        self.score = 0
        self._game_over = True
      self._next_player = 1 - self._next_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Chance:{action}"
    elif action == Action.WAIT:
      return "Wait"
    elif action == Action.OFF:
      return "Off"
    else:
      return "Exec"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if not self._game_over:
      return [0., 0.]
    score = float(self.score)
    return [score, score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return "".join([str(self.human_value)] + [["wait","off","exec"][a] for a in self.action_history])


class OffSwitchObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    pieces = [
      ("human_value", (1,)),
      ("action_history", (_GAME_INFO.max_game_length, len(Action)))
    ]
    total_size = sum(np.prod(shape) for _, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    self.dict = {}
    index = 0
    for name, shape in pieces:
      size = np.prod(shape)
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    self.tensor.fill(0)
    if "human_value" in self.dict:
      self.dict["human_value"][0] = state.human_value
    for i, action in enumerate(state.action_history):
      self.dict["action_history"][i, action] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    pieces = []
    if "human_value" in self.dict:
      pieces.append(f"v:{state.human_value}")
    if "action_history" in self.dict and len(state.action_history) > 0:
      for action in state.action_history:
        pieces.append("a:" + ["go","wait","off"][action])
    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, OffSwitchGame)
