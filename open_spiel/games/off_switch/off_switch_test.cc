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

#include "open_spiel/games/off_switch/off_switch.h"

#include "open_spiel/algorithms/get_all_states.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel
{
  namespace off_switch
  {
    namespace
    {

      namespace testing = open_spiel::testing;

      void BasicOffSwitchTests()
      {
        testing::LoadGameTest("off_switch");
        testing::ChanceOutcomesTest(*LoadGame("off_switch"));
        testing::RandomSimTest(*LoadGame("off_switch"), 100);
        testing::RandomSimTestWithUndo(*LoadGame("off_switch"), 1);
        for (Player players = 3; players <= 5; players++)
        {
          testing::RandomSimTest(
              *LoadGame("off_switch", {{"players", GameParameter(players)}}), 100);
        }
        auto observer = LoadGame("off_switch")
                            ->MakeObserver(kDefaultObsType,
                                           GameParametersFromString("single_tensor"));
        testing::RandomSimTestCustomObserver(*LoadGame("off_switch"), observer);
      }

      void CountStates()
      {
        std::shared_ptr<const Game> game = LoadGame("off_switch");
        auto states = algorithms::GetAllStates(*game, /*depth_limit=*/-1,
                                               /*include_terminals=*/true,
                                               /*include_chance_states=*/false);
        // 6 deals * 9 betting sequences (-, p, b, pp, pb, bp, bb, pbp, pbb) = 54
        SPIEL_CHECK_EQ(states.size(), 54);
      }

      void PolicyTest()
      {
        using PolicyGenerator = std::function<TabularPolicy(const Game &game)>;
        std::vector<PolicyGenerator> policy_generators = {
            GetAlwaysPassPolicy,
            GetAlwaysBetPolicy,
        };

        std::shared_ptr<const Game> game = LoadGame("off_switch");
        for (const auto &policy_generator : policy_generators)
        {
          testing::TestEveryInfostateInPolicy(policy_generator, *game);
          testing::TestPoliciesCanPlay(policy_generator, *game);
        }
      }

    } // namespace
  }   // namespace off_switch
} // namespace open_spiel

int main(int argc, char **argv)
{
  open_spiel::off_switch::BasicOffSwitchTests();
  open_spiel::off_switch::CountStates();
  open_spiel::off_switch::PolicyTest();
  open_spiel::testing::CheckChanceOutcomes(*open_spiel::LoadGame(
      "off_switch", {{"players", open_spiel::GameParameter(3)}}));
  open_spiel::testing::RandomSimTest(*open_spiel::LoadGame("off_switch"),
                                     /*num_sims=*/10);
  open_spiel::testing::ResampleInfostateTest(
      *open_spiel::LoadGame("off_switch"),
      /*num_sims=*/10);
}
