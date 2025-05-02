import itertools
import numpy as np

dx_values = [-1, 0, 1]
dy_values = [-1, 0, 1]
dz_values = [-1, 0, 1]
press_values = [0, 1]

action_list = list(itertools.product(dx_values, dy_values, dz_values, press_values))

def id_to_action(action_id):
    return action_list[action_id]

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)