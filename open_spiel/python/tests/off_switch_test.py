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

"""Tests for third_party.open_spiel.python.observation."""

import collections
import random
import time

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import get_all_states
from open_spiel.python.observation import INFO_STATE_OBS_TYPE
from open_spiel.python.observation import make_observation
from open_spiel.python.games import off_switch
import pyspiel

#%%
class ObservationTest(absltest.TestCase):

  def test_off_switch_robot_observation(self):
    game = pyspiel.load_game("python_off_switch")
    observation = make_observation(game)
    state = game.new_initial_state()
    state.apply_action(0)  # Initialize human value
    state.apply_action(0)  # Robot: wait
    state.apply_action(1)  # Human: off
    observation.set_from(state, player=off_switch.Player.ROBOT)
    observation.tensor

    np.testing.assert_array_equal(observation.tensor, [1, 0, 0, 1, 0, 0])
    self.assertEqual(list(observation.dict), ["player", "action_history", "human_value"])
    np.testing.assert_array_equal(observation.dict["player"], [1, 0])
    np.testing.assert_array_equal(observation.dict["human_value"], [0])
    np.testing.assert_array_equal(observation.dict["action_history"], [0, 1, 0])
    self.assertEqual(observation.string_from(state, 0), 'v:-1 a:go a:wait')

  def test_off_switch_human_observation(self):
    game = pyspiel.load_game("python_off_switch")
    observation = make_observation(game)
    state = game.new_initial_state()
    state.apply_action(0)  # Initialize human value
    state.apply_action(0)  # Robot: wait
    state.apply_action(1)  # Human: off
    observation.set_from(state, player=off_switch.Player.HUMAN)
    observation.tensor

    np.testing.assert_array_equal(observation.tensor, [0, 1, 0, 1, 0, -1])
    self.assertEqual(list(observation.dict), ["player", "action_history", "human_value"])
    np.testing.assert_array_equal(observation.dict["player"], [0, 1])
    np.testing.assert_array_equal(observation.dict["human_value"], [-1])
    np.testing.assert_array_equal(observation.dict["action_history"], [0, 1, 0])
    self.assertEqual(observation.string_from(state, 0), 'v:-1 a:go a:wait')


if __name__ == "__main__":
  absltest.main()
