// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace reverse_tic_tac_toe {
namespace {

namespace testing = open_spiel::testing;

void BasicReverseTicTacToeTests() {
  testing::LoadGameTest("reverse_tic_tac_toe");
  testing::NoChanceOutcomesTest(*LoadGame("reverse_tic_tac_toe"));
  testing::RandomSimTest(*LoadGame("reverse_tic_tac_toe"), 100);
}

}  // namespace
}  // namespace reverse_tic_tac_toe
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::reverse_tic_tac_toe::BasicReverseTicTacToeTests();
}
