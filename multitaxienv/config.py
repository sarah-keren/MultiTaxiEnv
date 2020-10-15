taxi_env_rewards = dict(
    step=-1,
    no_fuel=-20,
    bad_pickup=-15,
    bad_dropoff=-15,
    bad_refuel=-10,
    pickup=-1,
    standby_engine_off=-1,
    turn_engine_on=-1,
    turn_engine_off=-1,
    standby_engine_on=-1,
    intermediate_dropoff=-10,
    final_dropoff=100,
    hit_wall=-20,
    collision=-30,
)

all_action_names = ['south', 'north', 'east', 'west',
                    'pickup', 'dropoff',
                    'turn_engine_on', 'turn_engine_off',
                    'standby',
                    'refuel']

base_available_actions = ['south', 'north', 'east', 'west',
                          'pickup', 'dropoff']
