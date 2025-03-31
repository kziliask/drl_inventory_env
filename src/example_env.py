import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional

kwargs = {
    "n_locations": 10,
    "n_timesteps": 7,
    "starting_inventory": [100.0 for _ in range(10)],
    "lead_times": [[2 for _ in range(10)] for _ in range(10)],
    "costs": {"unit": 1.0, "holding": 1.0, "shipment": 10.0},
    "unit_cost_param": 1.0,
    "mission_params": None,
    "random_demand_params": [
        {"missions": (2, 6), "distance": (500.0, 1000.0), "aircraft": (2, 4)}
        for _ in range(10)
    ],
    "random_demand_fixed_param": 1000.0,
    "random_demand_dist_params": {"p": (0.01, 0.03)},
    "seed": None,
    "scenario": str,
    "variance": float,
}


class MultiInvEnv(gym.Env):
    """
    A simple multi-location inventory allocation environment.

    Args:
        n_locations: The number of locations in the environment.
        n_timesteps: The number of timesteps in the environment.
        starting_inventory: The starting inventory at each location.
        lead_times: The lead times from each location j to each other location i.
        costs: A dictionary containing the unit, holding, and shipment costs.
        unit_cost_param: The weight of the unit costs in the reward function.
        mission_params: If available, the missions to be completed at each location at each timestep.\
            If not available, the missions will be generated from the random_demand_params.
        random_demand_params: The range of missions, distances, and aircraft per location.
        random_demand_fixed_param: The denominator of the fixed demand.
        random_demand_dist_params: The parameters for the random demand distribution.
        seed: The seed for the random number generator. \
            If not provided, no seed will be used.
        scenario: The scenario to use. Currently implemented: ["base", "variable", "depot"]
        variance: The variance to use for the "variable" scenario.
        can_truncate: Whether to truncate the episode at a random time. \
            If True, the episode will be truncated with probability 1/365.
        precomputed_demands: The precomputed demands for each location at each timestep, optional.
    Raises:
        ValueError: If neither mission_params nor random_demand_params are provided.
    """

    def __init__(
        self,
        n_locations: int,
        n_timesteps: int,
        starting_inventory: List[float],
        lead_times: List[List[int]],  # from j to i
        costs: Dict[str, float],
        unit_cost_param: float = 1.0,
        mission_params: Optional[List[List[List[Dict[str, float]]]]] = None,
        random_demand_params: Optional[List[Dict[str, Tuple[float, float]]]] = None,
        random_demand_fixed_param: float = 1000.0,
        random_demand_dist_params: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
        scenario: str = "base",  # new -- select scenario
        variance: float = 1.0,  # control the perturbation of the "variable" scenario
        can_truncate: bool = False,
        precomputed_demands: Optional[List[List[List[float]]]] = None,
    ) -> None:
        super().__init__()
        self.render_mode = None
        self.n_locations = n_locations
        self.starting_inventory = starting_inventory
        self.T = n_timesteps
        self.current_T = 0
        self.lead_times = lead_times
        self.unit_costs = costs["unit"]
        self.holding_costs = costs["holding"]
        self.shipment_costs = costs["shipment"]
        self.total_costs = 0.0
        self.lam_param = unit_cost_param
        self.variance = variance
        self.scenario = scenario
        self.shortages: List[Dict[str, int]] = []
        self.mission_params: Optional[List[List[List[Dict[str, float]]]]] = None
        self.can_truncate = can_truncate
        self.depot_idx = -1
        self.total_demand_count = 0

        self.precomputed_demands = precomputed_demands
        # save all demands per period and location for analysis
        self.demands: List[List[List[float]]] = []
        # save the original random_demand_params so we can perturb each reset
        self.original_random_demand_params = random_demand_params

        if random_demand_dist_params:
            self.demand_dist_params = random_demand_dist_params
        else:
            self.demand_dist_params = {"p": (0.01, 0.03)}
        if seed:
            rng_gen = np.random.default_rng(seed)
            self.rng_gen = rng_gen
        else:
            self.rng_gen = np.random.default_rng()
        if mission_params is not None:
            self.mission_params = mission_params
        elif random_demand_params is not None:
            if scenario != "variable":
                self.mission_params = self._generate_missions(random_demand_params)
            else:
                # For 'variable' scenario, we will generate new mission_params in reset()
                self.mission_params = None
        else:
            exception_str = (
                "Either mission_params or random_demand_params must be provided."
            )
            raise ValueError(exception_str)
        self.fixed_param = random_demand_fixed_param
        self.pipeline_inventory_tuples: Optional[List[List[int]]] = []
        # Environment parameters
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=100.0,
            shape=(self.n_locations, self.n_locations),
            dtype=np.float32,
        )
        if scenario == "depot":
            self.depot_idx = self.rng_gen.integers(0, self.n_locations)
            # give one random location a very large inventory
            self.starting_inventory = [x for x in self.starting_inventory]
            self.starting_inventory[self.depot_idx] = 100_000.0
        # The observation space is the current inventory levels at each location,
        # the expected demand at each location, and the pipeline inventory at each location.
        self.observation_space = gym.spaces.Dict(
            {
                "inventory": gym.spaces.Box(
                    low=0.0, high=np.inf, shape=(self.n_locations,), dtype=np.float32
                ),
                "expected_demand": gym.spaces.Box(
                    low=0.0, high=np.inf, shape=(self.n_locations,), dtype=np.float32
                ),
                "expected_demand_ratio": gym.spaces.Box(
                    low=0.0, high=np.inf, shape=(self.n_locations,), dtype=np.float32
                ),
                "pipeline_inventory": gym.spaces.Box(
                    low=0.0, high=np.inf, shape=(self.n_locations,), dtype=np.float32
                ),
            }
        )
        self.inventory = starting_inventory
        self.pipeline_inventory = [0 for _ in range(self.n_locations)]
        self.current_T = 0
        return None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # If variable scenario, perturb or re-sample random_demand_params each reset,
        # then generate fresh mission_params
        if self.scenario == "variable":
            if self.original_random_demand_params is None:
                raise ValueError(
                    "variable scenario requires original_random_demand_params"
                )
            perturbed_params = self._perturb_random_demand_params(
                self.original_random_demand_params
            )
            self.mission_params = self._generate_missions(perturbed_params)

        self.current_T = 0
        self.inventory = [x for x in self.starting_inventory]
        self.pipeline_inventory = [0 for _ in range(self.n_locations)]

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        reward = np.float32(0.0)
        total_shortages = 0.0
        # Clip the action to the action space.
        action = self._clip_actions(action, self.inventory)
        # Step 1: Reduce inventory levels by actual demand.
        actual_demand = self._generate_demand()
        self.demands.append(actual_demand)
        for location in range(self.n_locations):
            for mission_demand in actual_demand[location]:
                if location != self.depot_idx:
                    self.total_demand_count += 1
                self.inventory[location] -= mission_demand
                if self.inventory[location] < 0.0:
                    self.inventory[location] = 0.0
                    total_shortages += 1
                    self.shortages.append(
                        {"location": location, "time": self.current_T}
                    )
                    reward -= 1
                else:
                    reward += 1

        # Step 2: Update pipeline inventory levels.
        # First, deliver inventory from the pipeline.
        # Iterate over a copy (slice) or build a new list
        new_pipeline = []
        for pipeline_tuple in self.pipeline_inventory_tuples:
            pipeline_tuple[0] -= 1
            if pipeline_tuple[0] == 0:
                # Deliver inventory
                self.inventory[pipeline_tuple[2]] += pipeline_tuple[1]
                self.pipeline_inventory[pipeline_tuple[2]] -= pipeline_tuple[1]
                # Don’t append to new_pipeline because it's fully delivered
            else:
                # Keep it in the pipeline
                new_pipeline.append(pipeline_tuple)

        self.pipeline_inventory_tuples = new_pipeline

        # Next, update pipeline inventory levels based on the action.
        for location in range(self.n_locations):
            for other_location in range(self.n_locations):
                if other_location == location:
                    continue
                self.pipeline_inventory[location] += min(
                    action[location][other_location], self.inventory[other_location]
                )
                # Increment costs
                self.total_costs += self.unit_costs * action[location][other_location]
                if (
                    min(
                        action[location][other_location], self.inventory[other_location]
                    )
                    > 1
                ):
                    self.total_costs += self.shipment_costs
                self.pipeline_inventory_tuples.append(
                    [
                        max(0, self.lead_times[location][other_location]),
                        min(
                            action[location][other_location],
                            self.inventory[other_location],
                        ),
                        location,
                    ]
                )

                # Step 3: Update inventory levels based on the action.
                self.inventory[other_location] -= min(
                    action[location][other_location], self.inventory[other_location]
                )
                # Calculate the holding costs
                if other_location != self.depot_idx:
                    self.total_costs += (
                        self.holding_costs * self.inventory[other_location]
                    )
                if self.inventory[other_location] < 0:
                    self.inventory[other_location] = 0
                for row in action:
                    for val in row:
                        if val >= 1.0:
                            reward -= 0.00
                # reward -= (
                #         self.lam_param
                #         * action[location][other_location]
                #         * self.c[location][other_location]
                #         * 0.0 #just for now
                #     )
        # Step 4: Update the timestep.
        self.current_T += 1
        # Check for end of episode
        terminated = False
        if self.can_truncate:
            if self.rng_gen.random() < 1 / 365:
                truncated = True
        if self.current_T >= self.T - 1:
            # print("Mean final inventory: ",np.mean(self.inventory), "\nMean final pipeline: ",np.mean(self.pipeline_inventory))
            terminated = True
        truncated = False
        # Step 5: Generate the next observation.
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _perturb_random_demand_params(
        self, base_params: List[Dict[str, Tuple[float, float]]]
    ) -> List[Dict[str, Tuple[float, float]]]:
        """
        Perturb each location's (missions, distance, aircraft) by some small random offset,
        controlled by self.variance. You can get more/less fancy as desired.
        """
        new_params = []
        for location_params in base_params:
            # Example: add integer offset in [-variance, variance]
            # Missions
            m_low, m_high = location_params["missions"]
            # random integer offset in [-variance, variance]
            m_offset_low = self.rng_gen.integers(
                int(-self.variance), int(self.variance) + 1
            )
            m_offset_high = self.rng_gen.integers(
                int(-self.variance), int(self.variance) + 1
            )
            m_new_low = max(1, m_low + m_offset_low)
            m_new_high = max(m_new_low + 1, m_high + m_offset_high)

            # Distances
            d_low, d_high = location_params["distance"]
            # random float offset
            d_offset_low = self.rng_gen.uniform(
                -self.variance * 100, self.variance * 100
            )
            d_offset_high = self.rng_gen.uniform(
                -self.variance * 100, self.variance * 100
            )
            d_new_low = max(100.0, d_low + d_offset_low)
            d_new_high = max(d_new_low + 1e5, d_high + d_offset_high)

            # Aircraft
            a_low, a_high = location_params["aircraft"]
            a_offset_low = self.rng_gen.integers(
                int(-self.variance), int(self.variance) + 1
            )
            a_offset_high = self.rng_gen.integers(
                int(-self.variance), int(self.variance) + 1
            )
            a_new_low = max(1, a_low + a_offset_low)
            a_new_high = max(a_new_low + 1, a_high + a_offset_high)

            new_params.append(
                {
                    "missions": (m_new_low, m_new_high),
                    "distance": (d_new_low, d_new_high),
                    "aircraft": (a_new_low, a_new_high),
                }
            )
        return new_params

    def _clip_actions(self, action, current_inventory):
        # For each source location (columns), ensure total shipped does not exceed its available inventory.
        for j in range(self.n_locations):
            total_shipped = np.sum(action[:, j])
            if total_shipped > current_inventory[j]:
                scaling_factor = current_inventory[j] / total_shipped
                action[:, j] *= scaling_factor
        return action

    def _get_observation(self) -> Dict[str, np.ndarray]:
        # Return the current observation.
        return {
            "inventory": np.clip(np.array(self.inventory, dtype=np.float32), 0, np.inf),
            "expected_demand": np.clip(
                np.array(self._generate_expected_demand(), dtype=np.float32), 0, np.inf
            ),
            "expected_demand_ratio": np.clip(
                np.array(self.inventory, dtype=np.float32)
                / np.array(self._generate_expected_demand(), dtype=np.float32),
                0,
                np.inf,
            ),
            "pipeline_inventory": np.clip(
                np.array(self.pipeline_inventory, dtype=np.float32), 0, np.inf
            ),
        }

    def _get_info(self) -> Dict[str, np.ndarray]:
        return {
            "lead_times": np.array(self.lead_times),
            "current_T": np.array(self.current_T),
        }

    def _generate_missions(
        self,
        random_demand_params: List[Dict[str, Tuple[float, float]]],
    ) -> List[List[List[Dict[str, float]]]]:
        # random_demand_params = per base dict, {missions: (min, max), distance: (min, max), aircraft: (min, max)}
        # return [location][day][mission]{distance, aircraft}
        env_missions = []
        for location in range(self.n_locations):
            daily_missions = []
            for _ in range(self.T):
                num_missions = self.rng_gen.integers(
                    int(random_demand_params[location]["missions"][0]),
                    int(random_demand_params[location]["missions"][1]),
                )
                missions = []
                for _ in range(num_missions):
                    distance = self.rng_gen.uniform(
                        random_demand_params[location]["distance"][0],
                        random_demand_params[location]["distance"][1],
                    )
                    aircraft = self.rng_gen.integers(
                        int(random_demand_params[location]["aircraft"][0]),
                        int(random_demand_params[location]["aircraft"][1]),
                    )
                    missions.append({"distance": distance, "aircraft": aircraft})
                daily_missions.append(missions)
            env_missions.append(daily_missions)
        return env_missions

    def _generate_demand(self) -> List[List[float]]:
        # For the current timestep, generate the demand for each location.
        # If the user provides precomputed demand, use it. Otherwise continue as usual.
        if self.precomputed_demands is not None:
            daily_demand = self.precomputed_demands[self.current_T]
            return daily_demand
        daily_demand = []  # [location][mission_demand]
        total_demand = 0.0
        # total_inventory = sum(self.inventory)
        for location in range(self.n_locations):
            loc_demand = []
            if self.mission_params is not None:
                for mission in self.mission_params[location][self.current_T]:
                    fixed_demand = (
                        mission["distance"] * mission["aircraft"] / self.fixed_param
                    )
                    random_demand = self.rng_gen.binomial(
                        int(mission["aircraft"]),
                        self.rng_gen.uniform(*self.demand_dist_params["p"]),
                    )
                    mission_demand = fixed_demand + random_demand
                    loc_demand.append(mission_demand)
                daily_demand.append(loc_demand)

        for loc_demand in daily_demand:
            total_demand += sum(loc_demand)

        return daily_demand

    def _generate_expected_demand(self) -> List[float]:
        # For the current timestep, generate the expected demand for each location.
        expected_demand = []  # [location]
        for location in range(self.n_locations):
            mission_params = {}
            mission_params["distance"] = 0.0
            mission_params["aircraft"] = 0
            if self.mission_params is not None:
                for mission in self.mission_params[location][self.current_T]:
                    mission_params["distance"] += mission["distance"]
                    mission_params["aircraft"] += mission["aircraft"]
            fixed_demand = (
                mission_params["distance"]
                * mission_params["aircraft"]
                / self.fixed_param
            )
            demand = (
                fixed_demand
                + mission_params["aircraft"]
                * (self.demand_dist_params["p"][0] + self.demand_dist_params["p"][1])
                / 2
            )
            expected_demand.append(demand)
        return expected_demand


if __name__ == "__main__":
    gym.register(
        id="MultiInv-v0",
        entry_point="example_env:MultiInvEnv",
    )
    print(gym.pprint_registry())
