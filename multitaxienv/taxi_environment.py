# -*- coding: utf-8 -*-

# TODO - Update Notebook

import sys

import gym
from contextlib import closing
from io import StringIO
from gym import utils
from gym.utils import seeding
import numpy as np
import random
from .config import taxi_env_rewards, base_available_actions, all_action_names

MAP = [
    "+---------+",
    "|X: |F: :X|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|X| :G|X: |",
    "+---------+",
]


class TaxiEnv(gym.Env):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue).
    When the episode starts, the taxi starts off at a random square and the passenger is at a random location.
    The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination
    (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off,
    the episode ends.

    Observations:
    A list (taxis, fuels, pass_start, destinations, pass_locs):
        taxis:                  a list of coordinates of each taxi
        fuels:                  a list of fuels for each taxi
        pass_start:             a list of starting coordinates for each passenger (current position or last available)
        destinations:           a list of destination coordinates for each passenger
        passengers_locations:   a list of locations of each passenger.
                                -1 means delivered
                                0 means not picked up
                                positive number means the passenger is in the corresponding taxi number

    Passenger start: coordinates of each of these
    - -1: In a taxi
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Passenger location:
    - -1: delivered
    - 0: not in taxi
    - x: in taxi x (x is integer)

    Destinations: coordinates of each of these
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Fuel:
     - 0 to np.inf: default with 10

    Actions:
    Actions are given as a list, each element referring to one taxi's action. Each taxi has 7 actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger
    - 6: turn engine on
    - 7: turn engine off
    - 8: standby
    - 9: refuel fuel tank


    Rewards:
    - Those are specified in the config file.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    Main class to be characterized with hyper-parameters.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, num_taxis: int = 2, num_passengers: int = 2, max_fuel: list = None,
                 domain_map: list = None, taxis_capacity: list = None, collision_sensitive_domain: bool = False,
                 fuel_type_list: list = None, option_to_stand_by: bool = True):
        """
        TODO -  later version make number of passengers dynamic, even in runtime
        Args:
            num_taxis: number of taxis in the domain
            num_passengers: number of passengers occupying the domain
            max_fuel: list of max and starting fuel, we use np.inf as default for fuel free taxi.
            domain_map: map of the domain
            taxis_capacity: max capacity of passengers in each taxi (list)
            collision_sensitive_domain: is the domain show and react (true) to collisions or not (false)
            fuel_type_list: list of fuel types of each taxi
            option_to_stand_by: can taxis simply stand in place
        """

        # Initializing default value
        if max_fuel is None:
            self.max_fuel = [np.inf] * num_passengers
        else:
            self.max_fuel = max_fuel

        if domain_map is None:
            self.desc = np.asarray(MAP, dtype='c')
        else:
            self.desc = np.asarray(domain_map, dtype='c')

        if taxis_capacity is None:
            self.taxis_capacity = [1] * num_passengers
        else:
            self.taxis_capacity = taxis_capacity

        if fuel_type_list is None:
            self.fuel_type_list = ['F'] * num_passengers
        else:
            self.fuel_type_list = fuel_type_list

        # Relevant features for map boundaries, notice that we can only drive between the columns (':')
        self.num_rows = num_rows = len(self.desc) - 2
        self.num_columns = num_columns = len(self.desc[0][1:-1:2])

        # Set locations of passengers and fuel stations according to the map.
        self.passengers_locations = []
        self.fuel_station1 = None
        self.fuel_station2 = None
        self.fuel_stations = []

        for i, row in enumerate(self.desc[1:-1]):
            for j, char in enumerate(row[1:-1:2]):
                loc = [i, j]
                if char == b'X':
                    self.passengers_locations.append(loc)
                elif char == b'F':
                    self.fuel_station1 = loc
                    self.fuel_stations.append(loc)
                elif char == b'G':
                    self.fuel_station2 = loc
                    self.fuel_stations.append(loc)

        self.coordinates = [[i, j] for i in range(num_rows) for j in range(num_columns)]

        self.num_taxis = num_taxis

        self.collision_sensitive_domain = collision_sensitive_domain

        # Indicator list of 1's (collided) and 0's (not-collided) of all taxis
        self.collided = np.zeros(num_taxis)

        self.option_to_standby = option_to_stand_by

        # A list to indicate whether the engine of taxi i is on (1) or off (0), all taxis start as on.
        self.engine_status_list = list(np.ones(num_taxis).astype(bool))
        self.num_passengers = num_passengers

        # Available actions in relation to all actions based on environment parameters.
        self.available_actions_indexes, self.index_action_dictionary, self.action_index_dictionary \
            = self.set_available_actions_dictionary()
        self.num_actions = len(self.available_actions_indexes)
        self.action_space = gym.spaces.MultiDiscrete([self.num_actions for _ in range(self.num_taxis)])
        self.last_action = None

        self.seed()
        self.state = None
        self.dones = []

        self.np_random = None

    def seed(self, seed=None) -> list:
        """
        Setting a seed for the random sample state generation.
        Args:
            seed: seed to use

        Returns: list[seed]

        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> list:
        """
        Reset the environment's state:
            - taxis coordinates.
            - refuel all taxis
            - random get destinations.
            - random locate passengers.
            - preserve other definitions of the environment (collision, capacity...)
            - all engines turn on.
        Args:

        Returns: The reset state.

        """
        taxis_locations = random.sample(self.coordinates, self.num_taxis)
        fuels = [self.max_fuel[i] for i in range(self.num_taxis)]

        passengers_start_location = [start for start in
                                     random.choices(self.passengers_locations, k=self.num_passengers)]
        passengers_destinations = [random.choice([x for x in self.passengers_locations if x != start])
                                   for start in passengers_start_location]

        # Status of each passenger: delivered (-1), in_taxi (positive number), waiting (0)
        passengers_status = [0 for _ in range(self.num_passengers)]
        self.state = [taxis_locations, fuels, passengers_start_location, passengers_destinations, passengers_status]

        self.last_action = None
        # Turning all engines on
        self.engine_status_list = list(np.ones(self.num_taxis))

        return self.state

    def set_available_actions_dictionary(self) -> (list, dict, dict):
        """

        TODO: Later versions - maybe return an action-dictionary for each taxi individually.

        Generates list of all available actions in the parametrized domain, index->action dictionary to decode.
        Generation is based on the hyper-parameters passed to __init__ + parameters defined in config.py

        Returns: list of available actions, index->action dictionary for all actions and the reversed dictionary.

        """

        action_names = all_action_names  # From config.py
        base_dictionary = {}  # Total dictionary{index -> action_name}
        for index, action in enumerate(action_names):
            base_dictionary[index] = action

        available_action_list = base_available_actions  # From config.py

        if self.option_to_standby:
            available_action_list += ['turn_engine_on', 'turn_engine_off', 'standby']

        # TODO - when we return dictionary per taxi we can't longer assume that on np.inf fuel
        #  means no limited fuel for all the taxis
        if not self.max_fuel[0] == np.inf:
            available_action_list.append('refuel')

        action_index_dictionary = dict((value, key) for key, value in base_dictionary.items())  # {action -> index} all
        available_actions_indexes = [action_index_dictionary[action] for action in available_action_list]
        index_action_dictionary = dict((key, value) for key, value in base_dictionary.items())

        return list(set(available_actions_indexes)), index_action_dictionary, action_index_dictionary

    def get_available_actions_dictionary(self) -> (list, dict):
        """
        Returns: list of available actions and index->action dictionary for all actions.

        """
        return self.available_actions_indexes, self.index_action_dictionary

    def is_place_on_taxi(self, passengers_locations: np.array, taxi_index: int) -> bool:
        """
        Checks if there is room for another passenger on taxi number 'taxi_index'.
        Args:
            passengers_locations: list of all passengers locations
            taxi_index: index of the desired taxi

        Returns: Whether there is a place (True) or not (False)

        """
        return (len([location for location in passengers_locations if location == (taxi_index + 1)]) <
                self.taxis_capacity[taxi_index])

    def map_at_location(self, location: list) -> str:
        """
        Returns the map character on the specified coordinates of the grid.
        Args:
            location: location to check [row, col]

        Returns: character on specific location on the map

        """
        domain_map = self.desc.copy().tolist()
        row, col = location[0], location[1]
        return domain_map[row + 1][2 * col + 1].decode(encoding='UTF-8')

    def at_valid_fuel_station(self, taxi: int, taxis_locations: list) -> bool:
        """
        Checks if the taxi's location is a suitable fuel station or not.
        Args:
            taxi: the desirable taxi
            taxis_locations: list of taxis coordinates
        Returns: whether the taxi is at a suitable fuel station (true) or not (false)

        """
        return (taxis_locations[taxi] in self.fuel_stations and
                self.map_at_location(taxis_locations[taxi]) == self.fuel_type_list[taxi])

    def step(self, actions: list) -> (list, list, bool):
        """
        TODO - add an option to choose whether to execute in joint/serialized manner.
        Executing a list of actions (action for each taxi) at the domain current state.
        Supports not-joined actions, just pass 1 element instead of list.

        actions[i] is the action of taxi i.
        Args:
            actions: list[int] - list of actions to take.

        Returns: list of next_state, reward_collected, is_done
        """
        # Boundaries to check if "hit a wall" occurred and calculate movement
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1

        rewards = []

        # Main of the function, for each taxi-i act on action[i]
        for taxi, action in enumerate(actions):
            reward = taxi_env_rewards['step']  # Default reward
            moved = False  # Indicator variable for later use
            # If the taxi collided, it can't perform a step
            if self.collided[taxi] == 1:
                continue

            taxis_locations, fuels, passengers_start_locations, destinations, passengers_status = self.state

            # If the taxi is out of fuel, it can't perform a step
            if fuels[taxi] == 0 and not self.at_valid_fuel_station(taxi, taxis_locations):
                continue

            taxi_location = taxis_locations[taxi]
            row, col = taxi_location
            fuel = fuels[taxi]
            is_taxi_engine_on = self.engine_status_list[taxi]
            _, index_action_dictionary = self.get_available_actions_dictionary()

            if not is_taxi_engine_on:  # Engine is off
                if index_action_dictionary[action] == 'standby':  # standby while engine is off
                    reward = taxi_env_rewards['standby_engine_off']
                elif index_action_dictionary[action] == 'turn_engine_on':  # turn engine on
                    reward = taxi_env_rewards['turn_engine_on']
                    self.engine_status_list[taxi] = 1

            elif is_taxi_engine_on:  # Engine is on
                # Movement
                if index_action_dictionary[action] == 'south':  # south
                    if row != max_row:
                        moved = True
                    row = min(row + 1, max_row)
                elif index_action_dictionary[action] == 'north':  # north
                    if row != 0:
                        moved = True
                    row = max(row - 1, 0)
                if index_action_dictionary[action] == 'east' and self.desc[1 + row, 2 * col + 2] == b":":  # east
                    if col != max_col:
                        moved = True
                    col = min(col + 1, max_col)
                elif index_action_dictionary[action] == 'west' and self.desc[1 + row, 2 * col] == b":":  # west
                    if col != 0:
                        moved = True
                    col = max(col - 1, 0)

                # Check for collisions
                if self.collision_sensitive_domain and moved:
                    if self.collided[taxi] == 0:
                        # Check if the number of taxis on the destination location is greater than 1
                        if len([i for i in range(self.num_taxis) if taxis_locations[i] == [row, col]]) > 0:
                            if self.option_to_standby:
                                moved = False
                                action = self.action_index_dictionary['standby']
                            else:
                                self.collided[[i for i in range(len(taxis_locations)) if taxis_locations[i] ==
                                               [row, col]]] = 1
                                self.collided[taxi] = 1
                                reward = taxi_env_rewards['collision']
                if self.collision_sensitive_domain and self.collided[taxi] == 1:  # Taxi is already collided
                    pass

                # Pickup
                elif index_action_dictionary[action] == 'pickup':
                    successful_pickup = False
                    for i, location in enumerate(passengers_status):
                        # Check if we can take this passenger
                        if location == 0 and taxi_location == passengers_start_locations[i] and self.is_place_on_taxi(
                                passengers_status, taxi):
                            passengers_status[i] = taxi + 1
                            successful_pickup = True
                            reward = taxi_env_rewards['pickup']
                    if not successful_pickup:  # passenger not at location
                        reward = taxi_env_rewards['bad_pickup']

                # Dropoff
                elif index_action_dictionary[action] == 'dropoff':
                    successful_dropoff = False
                    for i, location in enumerate(passengers_status):  # at destination
                        # Check if we have the passenger and we are at his destination
                        if location == taxi + 1 and taxi_location == destinations[i]:
                            passengers_status[i] = -1
                            reward = taxi_env_rewards['final_dropoff']
                            successful_dropoff = True
                        elif location == taxi + 1:  # drops off passenger not at destination
                            passengers_status[i] = 0
                            successful_dropoff = True
                            reward = taxi_env_rewards['intermediate_dropoff']
                            passengers_start_locations[i] = taxi_location
                    if not successful_dropoff:  # not carrying a passenger
                        reward = taxi_env_rewards['bad_dropoff']

                # Turning engine off
                elif index_action_dictionary[action] == 'turn_engine_off':
                    reward = taxi_env_rewards['turn_engine_off']
                    self.engine_status_list[taxi] = 0

                # Standby with engine on
                elif index_action_dictionary[action] == 'standby':
                    reward = taxi_env_rewards['standby_engine_on']

            # Here we have finished checking for action for taxi-i
            # Fuel consumption
            if moved:
                if fuel == 0:
                    reward = taxi_env_rewards['no_fuel']
                else:
                    fuel = max(0, fuel - 1)
                    taxis_locations[taxi] = [row, col]
                    fuels[taxi] = fuel

            if (not moved) and action in [self.action_index_dictionary[direction] for
                                          direction in ['north', 'south', 'west', 'east']]:
                reward = taxi_env_rewards['hit_wall']

            # taxi refuel
            if index_action_dictionary[action] == 'refuel':
                if self.at_valid_fuel_station(taxi, taxis_locations):
                    fuels[taxi] = self.max_fuel[taxi]
                else:
                    reward = taxi_env_rewards['bad_refuel']

            # TODO - add feature to describe the 'done' cause
            # check if all the passengers are at their destinations
            done = all(loc == -1 for loc in passengers_status)
            self.dones.append(done)

            # check if all taxis collided
            done = all(self.collided == 1)
            self.dones.append(done)

            # check if all taxis are out of fuel
            done = all(np.array(fuels) == 0)
            self.dones.append(done)

            rewards.append(reward)
            self.state = [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
            self.last_action = actions

        return self.state, rewards, any(self.dones)

    def render(self, mode: str = 'human') -> str:
        """
        Renders the domain map at the current state
        Args:
            mode: Demand mode (file or human watching).

        Returns: Value string of writing the output

        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # Copy map to work on
        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]

        taxis, fuels, passengers_start_coordinates, destinations, passengers_locations = self.state

        colors = ['yellow', 'red', 'white', 'green', 'cyan', 'crimson', 'gray', 'magenta'] * 5
        colored = [False] * self.num_taxis

        def ul(x):
            """returns underline instead of spaces when called"""
            return "_" if x == " " else x

        for i, location in enumerate(passengers_locations):
            if location > 0:  # Passenger is on a taxi
                taxi_row, taxi_col = taxis[location - 1]

                # Coloring taxi's coordinate on the map
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    out[1 + taxi_row][2 * taxi_col + 1], colors[location - 1], highlight=True, bold=True)
                colored[location - 1] = True
            else:  # Passenger isn't in a taxi
                # Coloring passenger's coordinates on the map
                pi, pj = passengers_start_coordinates[i]
                out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)

        for i, taxi in enumerate(taxis):
            if self.collided[i] == 0:  # Taxi isn't collided
                taxi_row, taxi_col = taxi
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), colors[i], highlight=True)
            else:  # Collided!
                taxi_row, taxi_col = taxi
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), 'gray', highlight=True)

        for dest in destinations:
            di, dj = dest
            out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        if self.last_action is not None:
            moves = all_action_names
            output = [moves[i] for i in self.last_action]
            outfile.write("  ({})\n".format(' ,'.join(output)))
        for i, taxi in enumerate(taxis):
            outfile.write("Taxi{}-{}: Fuel: {}, Location: ({},{}), Collided: {}\n".format(i + 1, colors[i].upper(),
                                                                                          fuels[i], taxi[0], taxi[1],
                                                                                          self.collided[i] == 1))
        for i, location in enumerate(passengers_locations):
            start = tuple(passengers_start_coordinates[i])
            end = tuple(destinations[i])
            if location < 0:
                outfile.write("Passenger{}: Location: Arrived!, Destination: {}\n".format(i + 1, end))
            if location == 0:
                outfile.write("Passenger{}: Location: {}, Destination: {}\n".format(i + 1, start, end))
            else:
                outfile.write("Passenger{}: Location: Taxi{}, Destination: {}\n".format(i + 1, location, end))

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    @staticmethod
    def partial_observations(state: list) -> list:
        """
        Get partial observation of state.
        Args:
            state: state of the domain (taxis, fuels, passengers_start_coordinates, destinations, passengers_locations)

        Returns: list of observations s.t each taxi sees only itself

        """

        def flatten(x):
            return [item for sub in x for item in sub]

        observations = []
        taxis, fuels, passengers_start_locations, passengers_destinations, passengers_locations = state
        pass_info = flatten(passengers_start_locations) + flatten(passengers_destinations) + passengers_locations

        for i in range(len(taxis)):
            obs = taxis[i] + [fuels[i]] + pass_info
            obs = np.reshape(obs, [1, len(obs)])
            observations.append(obs)
        return observations

    @staticmethod
    def get_observation(state: list, agent_index: int) -> np.array:
        """
        Takes only the observation of the specified agent.
        Args:
            state: state of the domain (taxis, fuels, passengers_start_coordinates, destinations, passengers_locations)
            agent_index: 

        Returns: observation of the specified agent (state wise)

        """

        def flatten(x):
            return [item for sub in x for item in sub]

        taxis, fuels, passengers_start_locations, passengers_destinations, passengers_locations = state
        passengers_information = flatten(passengers_start_locations) + flatten(
            passengers_destinations) + passengers_locations

        observations = taxis[agent_index] + [fuels[agent_index]] + passengers_information
        observations = np.reshape(observations, [1, len(observations)])

        return observations
