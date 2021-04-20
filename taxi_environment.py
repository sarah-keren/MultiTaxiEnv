# -*- coding: utf-8 -*-

import gym
from gym.utils import seeding
import numpy as np
from config import TAXI_ENVIRONMENT_REWARDS, BASE_AVAILABLE_ACTIONS, ALL_ACTIONS_NAMES
from gym.spaces import Box, Tuple, MultiDiscrete
import random
import sys
from contextlib import closing
from io import StringIO
from gym import utils

orig_MAP = [
    "+---------+",
    "|X: |F: :X|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|X| :G|X: |",
    "+---------+",
]

MAP2 = [
    "+-------+",
    "|X: |F:X|",
    "| : | : |",
    "| : : : |",
    "|X| :G|X|",
    "+-------+",
]

MAP = [
    "+-----------------------+",
    "|X: |F: | : | : | : |F:X|",
    "| : | : : : | : | : | : |",
    "| : : : : : : : : : : : |",
    "| : : : : : | : : : : : |",
    "| : : : : : | : : : : : |",
    "| : : : : : : : : : : : |",
    "|X| :G| | | :G| | | : |X|",
    "+-----------------------+",
]

simple_MAP = [
    "+---------+",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "| : : : : |",
    "+---------+",
]


class TaxiEnv:
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
        pass_start:             a list of starting coordinates for taeach passenger (current position or last available)
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

    def __init__(self, _=0, num_taxis: int = 2, num_passengers: int = 1, max_fuel: list = None,
                 domain_map: list = None, taxis_capacity: list = None, collision_sensitive_domain: bool = True,
                 fuel_type_list: list = None, option_to_stand_by: bool = False):
        """
        TODO -  later version make number of passengers dynamic, even in runtime
        Args:
            num_taxis: number of taxis in the domain
            num_passengers: number of passengers occupying the domain at initiailaization
            max_fuel: list of max (start) fuel, we use np.inf as default for fuel free taxi.
            domain_map: 2D - map of the domain
            taxis_capacity: max capacity of passengers in each taxi (list)
            collision_sensitive_domain: is the domain show and react (true) to collisions or not (false)
            fuel_type_list: list of fuel types of each taxi
            option_to_stand_by: can taxis simply stand in place
        """
        # Initializing default values
        self.num_taxis = num_taxis
        if max_fuel is None:
            self.max_fuel = [100] * num_taxis  # TODO - needs to figure out how to insert np.inf into discrete obs.space
        else:
            self.max_fuel = max_fuel

        if domain_map is None:
            self.desc = np.asarray(orig_MAP, dtype='c')
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

        # Relevant features for map orientation, notice that we can only drive between the columns (':')
        self.num_rows = num_rows = len(self.desc) - 2
        self.num_columns = num_columns = len(self.desc[0][1:-1:2])

        # Set locations of passengers and fuel stations according to the map.
        self.passengers_locations = []
        self.fuel_station1 = None
        self.fuel_station2 = None
        self.fuel_stations = []

        # initializing map with passengers and fuel stations
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

        # self.num_taxis = num_taxis
        self.taxis_names = ["taxi_" + str(index + 1) for index in range(self.num_taxis)]

        self.collision_sensitive_domain = collision_sensitive_domain

        # Indicator list of 1's (collided) and 0's (not-collided) of all taxis
        self.collided = np.zeros(self.num_taxis)

        self.option_to_standby = option_to_stand_by

        # A list to indicate whether the engine of taxi i is on (1) or off (0), all taxis start as on.
        self.engine_status_list = list(np.ones(self.num_taxis).astype(bool))

        self.num_passengers = num_passengers

        # Available actions in relation to all actions based on environment parameters.
        self.available_actions_indexes, self.index_action_dictionary, self.action_index_dictionary \
            = self._set_available_actions_dictionary()
        self.num_actions = len(self.available_actions_indexes)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.MultiDiscrete(self._get_observation_space_list())
        self.bounded = False

        self.last_action = None
        self.num_states = self._get_num_states()

        self._seed()
        self.state = None
        self.dones = {taxi_name: False for taxi_name in self.taxis_names}
        self.dones['__all__'] = False

        self.np_random = None
        self.reset()

    def _get_num_states(self):
        map_dim = (self.num_rows * self.num_columns)
        passengers_loc_dim = 1
        for i in range(self.num_passengers):
            passengers_loc_dim *= len(self.passengers_locations) + self.num_taxis - i
        passengers_dest_dim = 1
        for i in range(self.num_passengers):
            passengers_dest_dim *= len(self.passengers_locations) - i
        num_states = map_dim * passengers_loc_dim * passengers_dest_dim
        return num_states

    def _get_observation_space_list(self) -> list:
        """
        Returns a list that emebed the observation space size in each dimension.
        An observation is a list of the form:
        [
            taxi_row, taxi_col, taxi_fuel,
            passenger1_row, passenger1_col,
            ...
            passenger_n_row, passenger_n_col,
            passenger1_dest_row, passenger1_dest_col,
            ...
            passenger_n_dest_row, passenger_n_dest_col,
            passenger1_status,
            ...
            passenger_n_status
        ]
        Returns: a list with all the dimensions sizes of the above.

        """
        locations_sizes = [self.num_rows, self.num_columns]
        fuel_size = [max(self.max_fuel) + 1]
        passengers_status_size = [self.num_taxis + 3]
        dimensions_sizes = []

        for _ in range(self.num_taxis):
            dimensions_sizes += locations_sizes
        for _ in range(self.num_taxis):
            dimensions_sizes += fuel_size

        for _ in range(self.num_passengers):
            dimensions_sizes += 2 * locations_sizes
        for _ in range(self.num_passengers):
            dimensions_sizes += passengers_status_size

        return [dimensions_sizes]

    def _seed(self, seed=None) -> list:
        """
        Setting a seed for the random sample state generation.
        Args:
            seed: seed to use

        Returns: list[seed]

        """
        self.np_random, self.seed_id = seeding.np_random(seed)
        return np.array([self.seed_id])

    def reset(self) -> dict:
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
        # reset taxis locations
        taxis_locations = random.sample(self.coordinates, self.num_taxis)
        self.collided = np.zeros(self.num_taxis)
        self.bounded = False
        self.window_size = 5
        self.counter = 0

        # refuel everybody
        fuels = [self.max_fuel[i] for i in range(self.num_taxis)]

        # reset passengers
        passengers_start_location = [start for start in
                                     random.choices(self.passengers_locations, k=self.num_passengers)]
        passengers_destinations = [random.choice([x for x in self.passengers_locations if x != start])
                                   for start in passengers_start_location]

        # Status of each passenger: delivered (1), in_taxi (positive number>2), waiting (2)
        passengers_status = [2 for _ in range(self.num_passengers)]
        self.state = [taxis_locations, fuels, passengers_start_location, passengers_destinations, passengers_status]

        self.last_action = None
        # Turning all engines on
        self.engine_status_list = list(np.ones(self.num_taxis))

        # resetting dones
        self.dones = {taxi_id: False for taxi_id in self.taxis_names}
        self.dones['__all__'] = False
        obs = {}
        for taxi_id in self.taxis_names:
            obs[taxi_id] = self.get_observation(self.state, taxi_id)

        return obs

    def _set_available_actions_dictionary(self) -> (list, dict, dict):
        """

        TODO: Later versions - maybe return an action-dictionary for each taxi individually.

        Generates list of all available actions in the parametrized domain, index->action dictionary to decode.
        Generation is based on the hyper-parameters passed to __init__ + parameters defined in config.py

        Returns: list of available actions, index->action dictionary for all actions and the reversed dictionary
        (action -> index).

        """

        action_names = ALL_ACTIONS_NAMES  # From config.py
        base_dictionary = {}  # Total dictionary{index -> action_name}
        for index, action in enumerate(action_names):
            base_dictionary[index] = action

        available_action_list = BASE_AVAILABLE_ACTIONS  # From config.py

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

    def _is_there_place_on_taxi(self, passengers_locations: np.array, taxi_index: int) -> bool:
        """
        Checks if there is room for another passenger on taxi number 'taxi_index'.
        Args:
            passengers_locations: list of all passengers locations
            taxi_index: index of the desired taxi

        Returns: Whether there is a place (True) or not (False)

        """
        # Remember that passengers "location" is: 1 - delivered, 2 - waits for a taxi, >2 - on a taxi with index
        # location+2

        return (len([location for location in passengers_locations if location == (taxi_index + 3)]) <
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
            taxi: the index of the desired taxi
            taxis_locations: list of taxis coordinates [row, col]
        Returns: whether the taxi is at a suitable fuel station (true) or not (false)

        """
        return (taxis_locations[taxi] in self.fuel_stations and
                self.map_at_location(taxis_locations[taxi]) == self.fuel_type_list[taxi])

    def _get_action_list(self, action_list) -> list:
        """
        Return a list in the correct format for the step function that should
        always get a list even if it's a single action.
        Args:
            action_list:

        Returns: list(action_list)

        """
        if type(action_list) == int:
            return [action_list]
        elif type(action_list) == np.int64:
            return [action_list]

        return action_list

    def _engine_is_off_actions(self, action: str, taxi: int) -> int:
        """
        Returns the reward according to the requested action given that the engine's is currently off.
        Also turns engine on if requested.
        Args:
            action: requested action
            taxi: index of the taxi specified, relevant for turning engine on
        Returns: correct reward

        """
        reward = self.partial_closest_path_reward('unrelated_action')
        if action == 'standby':  # standby while engine is off
            reward = self.partial_closest_path_reward('standby_engine_off')
        elif action == 'turn_engine_on':  # turn engine on
            reward = self.partial_closest_path_reward('turn_engine_on')
            self.engine_status_list[taxi] = 1

        return reward

    def _take_movement(self, action: str, row: int, col: int) -> (bool, int, int):
        """
        Takes a movement with regard to a apecific location of a taxi,
        Args:
            action: direction to move
            row: current row
            col: current col

        Returns: if moved (false if there is a wall), new row, new col

        """
        moved = False
        new_row, new_col = row, col
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1
        if action == 'south':  # south
            if row != max_row:
                moved = True
            new_row = min(row + 1, max_row)
        elif action == 'north':  # north
            if row != 0:
                moved = True
            new_row = max(row - 1, 0)
        if action == 'east' and self.desc[1 + row, 2 * col + 2] == b":":  # east
            if col != max_col:
                moved = True
            new_col = min(col + 1, max_col)
        elif action == 'west' and self.desc[1 + row, 2 * col] == b":":  # west
            if col != 0:
                moved = True
            new_col = max(col - 1, 0)

        return moved, new_row, new_col

    def _check_action_for_collision(self, taxi_index: int, taxis_locations: list, current_row: int, current_col: int,
                                    moved: bool, current_action: int, current_reward: int) -> (int, bool, int, list):
        """
        Takes a desired location for a taxi and update it with regard to collision check.
        Args:
            taxi_index: index of the taxi
            taxis_locations: locations of all other taxis.
            current_row: of the taxi
            current_col: of the taxi
            moved: indicator variable
            current_action: the current action requested
            current_reward: the current reward (left unchanged if there is no collision)

        Returns: new_reward, new_moved, new_action_index

        """
        reward = current_reward
        row, col = current_row, current_col
        moved = moved
        action = current_action
        taxi = taxi_index
        # Check if the number of taxis on the destination location is greater than 0
        if len([i for i in range(self.num_taxis) if taxis_locations[i] == [row, col]]) > 0:
            if self.option_to_standby:
                moved = False
                action = self.action_index_dictionary['standby']
            else:
                self.collided[[i for i in range(len(taxis_locations)) if taxis_locations[i] == [row, col]]] = 1
                self.collided[taxi] = 1
                reward = self.partial_closest_path_reward('collision')
                taxis_locations[taxi] = [row, col]

        return reward, moved, action, taxis_locations

    def _make_pickup(self, taxi: int, passengers_start_locations: list, passengers_status: list,
                     taxi_location: list, reward: int) -> (list, int):
        """
        Make a pickup (successful or fail) for a given taxi.
        Args:
            taxi: index of the taxi
            passengers_start_locations: current locations of the passengers
            passengers_status: list of passengers statuses (1, 2, greater..)
            taxi_location: location of the taxi
            reward: current reward

        Returns: updates passengers status list, updates reward

        """
        passengers_status = passengers_status
        reward = reward
        successful_pickup = False
        for i, location in enumerate(passengers_status):
            # Check if we can take this passenger
            if location == 2 and taxi_location == passengers_start_locations[i] and \
                    self._is_there_place_on_taxi(passengers_status, taxi):
                passengers_status[i] = taxi + 3
                successful_pickup = True
                reward = self.partial_closest_path_reward('pickup')
        if not successful_pickup:  # passenger not at location
            reward = self.partial_closest_path_reward('bad_pickup')

        return passengers_status, reward

    def _make_dropoff(self, taxi: int, current_passengers_start_locations: list, current_passengers_status: list,
                      destinations: list, taxi_location: list, reward: int) -> (list, list, int):
        """
        Make a dropoff (successful or fail) for a given taxi.
        Args:
            taxi: index of the taxi
            current_passengers_start_locations: current locations of the passengers
            current_passengers_status: list of passengers statuses (1, 2, greater..)
            destinations: list of passengers destinations
            taxi_location: location of the taxi
            reward: current reward

        Returns: updates passengers status list, updated passengers start location, updates reward

        """
        reward = reward
        passengers_start_locations = current_passengers_start_locations.copy()
        passengers_status = current_passengers_status.copy()
        successful_dropoff = False
        for i, location in enumerate(passengers_status):  # at destination
            location = passengers_status[i]
            # Check if we have the passenger and we are at his destination
            if location == (taxi + 3) and taxi_location == destinations[i]:
                passengers_status[i] = 1
                reward = self.partial_closest_path_reward('final_dropoff', taxi)
                passengers_start_locations[i] = taxi_location
                successful_dropoff = True
                break
            elif location == (taxi + 3):  # drops off passenger not at destination
                passengers_status[i] = 2
                successful_dropoff = True
                reward = self.partial_closest_path_reward('intermediate_dropoff', taxi)
                passengers_start_locations[i] = taxi_location
                break
        if not successful_dropoff:  # not carrying a passenger
            reward = self.partial_closest_path_reward('bad_dropoff')

        return passengers_status, passengers_start_locations, reward

    def _update_movement_wrt_fuel(self, taxi: int, taxis_locations: list, wanted_row: int, wanted_col: int,
                                  reward: int, fuel: int) -> (int, int, list):
        """
        Given that a taxi would like to move - check the fuel accordingly and update reward and location.
        Args:
            taxi: index of the taxi
            taxis_locations: list of current locations (prior to movement)
            wanted_row: row after movement
            wanted_col: col after movement
            reward: current reward
            fuel: current fuel

        Returns: updated_reward, updated fuel, updared_taxis_locations

        """
        reward = reward
        fuel = fuel
        taxis_locations = taxis_locations
        if fuel == 0:
            reward = ('no_fuel')
        else:
            fuel = max(0, fuel - 1)

            taxis_locations[taxi] = [wanted_row, wanted_col]

        return reward, fuel, taxis_locations

    def _refuel_taxi(self, current_fuel: int, current_reward: int, taxi: int, taxis_locations: list) -> (int, int):
        """
        Try to refuel a taxi, if successful - updates fuel tank, if not - updates the reward.
        Args:
            current_fuel: current fuel of the taxi
            current_reward: current reward for the taxi.
            taxi: taxi index
            taxis_locations: list of current taxis locations

        Returns: updated reward, updated fuel

        """
        fuel = current_fuel
        reward = current_reward
        if self.at_valid_fuel_station(taxi, taxis_locations) and fuel != self.max_fuel[taxi]:
            fuel = self.max_fuel[taxi]
        else:
            reward = self.partial_closest_path_reward('bad_refuel')

        return reward, fuel

    def step(self, action_dict: dict) -> (dict, dict, dict, dict):
        """
        Executing a list of actions (action for each taxi) at the domain current state.
        Supports not-joined actions, just pass 1 element instead of list.

        Args:
            action_dict: {taxi_name: action} - action of specific taxis to take on the step

        Returns: - dict{taxi_id: observation}, dict{taxi_id: reward}, dict{taxi_id: done}, _
        """

        rewards = {}
        self.counter += 1
        if self.counter >= 90:
            self.window_size = 3

        # Main of the function, for each taxi-i act on action[i]
        for taxi_name, action_list in action_dict.items():
            # meta operations on the type of the action
            action_list = self._get_action_list(action_list)

            for action in action_list:
                taxi = self.taxis_names.index(taxi_name)
                reward = self.partial_closest_path_reward('step')  # Default reward
                moved = False  # Indicator variable for later use

                # taxi locations: [i, j]
                # fuels: int
                # passengers_start_locations and destinations: [[i, j] ... [i, j]]
                # passengers_status: [[1, 2, taxi_index+2] ... [1, 2, taxi_index+2]], 1 - delivered
                taxis_locations, fuels, passengers_start_locations, destinations, passengers_status = self.state

                if all(list(self.dones.values())):
                    rewards[taxi_name] = reward
                    continue

                # If taxi is collided, it can't perform a step
                if self.collided[taxi] == 1:
                    rewards[taxi_name] = self.partial_closest_path_reward('collided')
                    self.dones[taxi_name] = True
                    continue

                # If the taxi is out of fuel, it can't perform a step
                if fuels[taxi] == 0 and not self.at_valid_fuel_station(taxi, taxis_locations):
                    rewards[taxi_name] = self.partial_closest_path_reward('bad_fuel')
                    self.dones[taxi_name] = True
                    continue

                taxi_location = taxis_locations[taxi]
                row, col = taxi_location

                fuel = fuels[taxi]
                is_taxi_engine_on = self.engine_status_list[taxi]
                _, index_action_dictionary = self.get_available_actions_dictionary()

                if not is_taxi_engine_on:  # Engine is off
                    # update reward according to standby/ turn-on/ unrelated + turn engine on if requsted
                    reward = self._engine_is_off_actions(index_action_dictionary[action], taxi)


                else:  # Engine is on
                    # Binding
                    if index_action_dictionary[action] == 'bind':
                        self.bounded = False
                        reward = self.partial_closest_path_reward('bind')

                    # Movement
                    if index_action_dictionary[action] in ['south', 'north', 'east', 'west']:
                        moved, row, col = self._take_movement(index_action_dictionary[action], row, col)

                    # Check for collisions
                    if self.collision_sensitive_domain and moved:
                        if self.collided[taxi] == 0:
                            reward, moved, action, taxis_locations = self._check_action_for_collision(taxi,
                                                                                                      taxis_locations,
                                                                                                      row, col, moved,
                                                                                                      action, reward)

                    # Pickup
                    elif index_action_dictionary[action] == 'pickup':
                        passengers_status, reward = self._make_pickup(taxi, passengers_start_locations,
                                                                      passengers_status, taxi_location, reward)

                    # Dropoff
                    elif index_action_dictionary[action] == 'dropoff':
                        passengers_status, passengers_start_locations, reward = self._make_dropoff(taxi,
                                                                                                   passengers_start_locations,
                                                                                                   passengers_status,
                                                                                                   destinations,
                                                                                                   taxi_location,
                                                                                                   reward)

                    # Turning engine off
                    elif index_action_dictionary[action] == 'turn_engine_off':
                        reward = self.partial_closest_path_reward('turn_engine_off')
                        self.engine_status_list[taxi] = 0

                    # Standby with engine on
                    elif index_action_dictionary[action] == 'standby':
                        reward = self.partial_closest_path_reward('standby_engine_on')

                # Here we have finished checking for action for taxi-i
                # Fuel consumption
                if moved:
                    reward, fuels[taxi], taxis_locations = self._update_movement_wrt_fuel(taxi, taxis_locations,
                                                                                          row, col, reward, fuel)

                if (not moved) and action in [self.action_index_dictionary[direction] for
                                              direction in ['north', 'south', 'west', 'east']]:
                    reward = TAXI_ENVIRONMENT_REWARDS['hit_wall']

                # taxi refuel
                if index_action_dictionary[action] == 'refuel':
                    reward, fuels[taxi] = self._refuel_taxi(fuel, reward, taxi, taxis_locations)

                # check if all the passengers are at their destinations
                done = all(loc == 1 for loc in passengers_status)
                self.dones[taxi_name] = done

                # check if all taxis collided
                done = all(self.collided == 1)
                self.dones[taxi_name] = self.dones[taxi_name] or done

                # check if all taxis are out of fuel
                done = fuels[taxi] == 0
                self.dones[taxi_name] = self.dones[taxi_name] or done

                rewards[taxi_name] = reward
                self.state = [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
                self.last_action = action_dict

        self.dones['__all__'] = True
        self.dones['__all__'] = all(list(self.dones.values()))

        if self.bounded:
            total_reward = 0
            for taxi_id in action_dict.keys():
                total_reward += rewards[taxi_id]
            total_reward /= len(action_dict.keys())
            for taxi_id in action_dict.keys():
                rewards[taxi_id] = total_reward

        obs = {}
        for taxi_id in action_dict.keys():
            obs[taxi_id] = self.get_observation(self.state, taxi_id)

        return obs, {taxi_id: rewards[taxi_id] for taxi_id in action_dict.keys()}, self.dones, {}

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
            if location > 2:  # Passenger is on a taxi
                taxi_row, taxi_col = taxis[location - 3]

                # Coloring taxi's coordinate on the map
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    out[1 + taxi_row][2 * taxi_col + 1], colors[location - 3], highlight=True, bold=True)
                colored[location - 3] = True
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
            moves = ALL_ACTIONS_NAMES
            output = [moves[i] for i in np.array(list(self.last_action.values())).reshape(-1)]
            outfile.write("  ({})\n".format(' ,'.join(output)))
        for i, taxi in enumerate(taxis):
            outfile.write("Taxi{}-{}: Fuel: {}, Location: ({},{}), Collided: {}\n".format(i + 1, colors[i].upper(),
                                                                                          fuels[i], taxi[0], taxi[1],
                                                                                          self.collided[i] == 1))
        for i, location in enumerate(passengers_locations):
            start = tuple(passengers_start_coordinates[i])
            end = tuple(destinations[i])
            if location == 1:
                outfile.write("Passenger{}: Location: Arrived!, Destination: {}\n".format(i + 1, end))
            if location == 2:
                outfile.write("Passenger{}: Location: {}, Destination: {}\n".format(i + 1, start, end))
            else:
                outfile.write("Passenger{}: Location: Taxi{}, Destination: {}\n".format(i + 1, location - 2, end))
        outfile.write("Done: {}, {}\n".format(all(self.dones.values()), self.dones))
        outfile.write("Passengers Status's: {}\n".format(self.state[-1]))

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

    def get_l1_distance(self, location1, location2):
        """
        Return the minimal travel length between 2 locations on the grid world.
        Args:
            location1: [i1, j1]
            location2: [i2, j2]

        Returns: np.abs(i1 - i2) + np.abs(j1 - j2)

        """
        return np.abs(location1[0] - location2[0]) + np.abs(location1[1] - location2[1])

    def get_observation(self, state: list, agent_name: str) -> np.array:
        """
        Takes only the observation of the specified agent.
        Args:
            state: state of the domain (taxis, fuels, passengers_start_coordinates, destinations, passengers_locations)
            agent_name: observer name
            window_size: the size that the agent can see in the map (around it) in terms of other txis

        Returns: observation of the specified agent (state wise)

        """

        def flatten(x):
            return [item for sub in list(x) for item in list(sub)]

        agent_index = self.taxis_names.index(agent_name)

        taxis, fuels, passengers_start_locations, passengers_destinations, passengers_locations = state.copy()
        passengers_information = flatten(passengers_start_locations) + flatten(
            passengers_destinations) + passengers_locations

        closest_taxis_indices = []
        for i in range(self.num_taxis):
            if self.get_l1_distance(taxis[agent_index], taxis[i]) <= self.window_size and i != agent_index:
                closest_taxis_indices.append(i)

        observations = taxis[agent_index].copy()
        for i in closest_taxis_indices:
            observations += taxis[i]
        observations += [0, 0] * (self.num_taxis - 1 - len(closest_taxis_indices)) + [fuels[agent_index]] + \
                        [0] * (self.num_taxis - 1) + passengers_information
        observations = np.reshape(observations, (1, len(observations)))

        return observations

    def passenger_destination_l1_distance(self, passenger_index, current_row: int, current_col: int) -> int:
        """
        Returns the manhattan distance between passenger current defined "start location" and it's destination.
        Args:
            passenger_index: index of the passenger.
            current_row: current row to calculate distance from destination
            current_col: current col to calculate distance from destination

        Returns: manhattan distance

        """
        current_state = self.state
        destination_row, destination_col = current_state[3][passenger_index]
        return int(np.abs(current_col - destination_col) + np.abs(current_row - destination_row))

    def partial_closest_path_reward(self, basic_reward_str: str, taxi_index: int = None) -> int:
        """
        Computes the reward for a taxi and it's defined by:
        dropoff[s] - gets the reward equal to the closest path multiply by 15, if the drive got a passenger further
        away - negative.
        other actions - basic reward from config table
        Args:
            basic_reward_str: the reward we would like to give
            taxi_index: index of the specific taxi

        Returns: updated reward

        """
        if basic_reward_str not in ['intermediate_dropoff', 'final_dropoff'] or taxi_index is None:
            return TAXI_ENVIRONMENT_REWARDS[basic_reward_str]

        # [taxis_locations, fuels, passengers_start_locations, destinations, passengers_status]
        current_state = self.state
        passengers_start_locations = current_state[2]

        taxis_locations = current_state[0]

        passengers_status = current_state[-1]
        passenger_index = passengers_status.index(taxi_index + 3)
        passenger_start_row, passenger_start_col = passengers_start_locations[passenger_index]
        taxi_current_row, taxi_current_col = taxis_locations[taxi_index]

        return 15 * (self.passenger_destination_l1_distance(passenger_index, passenger_start_row, passenger_start_col) -
                     self.passenger_destination_l1_distance(passenger_index, taxi_current_row, taxi_current_col))


env = TaxiEnv()
env.render()
