import itertools

dx_values = [-1, 0, 1]
dy_values = [-1, 0, 1]
dz_values = [-1, 0, 1]
press_values = [0, 1]

action_list = list(itertools.product(dx_values, dy_values, dz_values, press_values))

def id_to_action(action_id):
    return action_list[action_id]