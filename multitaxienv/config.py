taxifuel_rewards = dict(
    step = -1,
    no_fuel = -10,
    bad_pickup = -10,
    bad_dropoff = -10,
    bad_refuel = -10,
    pickup = -1,
    dropoff = 20,
    turn_engine_off = 0,
    standby_engine_off = 0,
    turn_engine_on = -1,
    standby_engine_on = -1,
)

multitaxi_rewards = dict(
    step = -1,
    bad_pickup = -10,
    bad_dropoff = -10,
    pickup = -1,
    intermediate_dropoff = -1,
    final_dropoff = 100,
    standby_engine_off = 0,
    turn_engine_on = -1,
    turn_engine_off = 0,
    standby_engine_on = -1,
    bad_refuel = -10,
)

multitaxifuel_rewards = dict(
    step = -1,
    no_fuel = -10,
    bad_pickup = -10,
    bad_dropoff = -10,
    bad_refuel = -10,
    pickup = -1,
    standby_engine_off = 0,
    turn_engine_on = -1,
    turn_engine_off = 0,
    standby_engine_on = -1,
    intermediate_dropoff = -1,
    final_dropoff = 100,
    hit_wall = -20,
)