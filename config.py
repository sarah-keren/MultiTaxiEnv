taxifuel_rewards = dict(
    step = -1,
    no_fuel = -10,
    bad_pickup = -10,
    bad_dropoff = -10,
    bad_refuel = -10,
    pickup = -1,
    dropoff = 20,
)

twotaxi_rewards = dict(
    step = -1,
    bad_pickup = -10,
    bad_dropoff = -10,
    pickup = -1,
    dropoff = 20,
)

twotaxifuel_rewards = dict(
    step = -1,
    no_fuel = -10,
    bad_pickup = -10,
    bad_dropoff = -10,
    bad_refuel = -10,
    pickup = -1,
    dropoff = 20,
    hit_wall = -20,
)

multitaxifuel_rewards = dict(
    step = -1,
    no_fuel = -10,
    bad_pickup = -10,
    bad_dropoff = -10,
    bad_refuel = -10,
    pickup = -1,
    intermediate_dropoff = -1,
    final_dropoff = 100,
    hit_wall = -20,
)