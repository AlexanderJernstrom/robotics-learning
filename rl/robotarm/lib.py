from jax import numpy as jnp
from jax import grad as grad
import numpy as np

def create3d_rotation_matrix(q) -> jnp.ndarray:
    rotation_vector = jnp.array([jnp.cos(q), -jnp.sin(q), 0, jnp.sin(q), jnp.cos(q), 0, 0, 0, 1])
    matrix = jnp.reshape(rotation_vector, (-1, 3)) 
    return matrix

def create_position_vector(r, axis) -> jnp.ndarray:
    vector = jnp.zeros(3)
    vector = vector.at[axis].set(r)
    return vector

def build_transformation_matrix(rotation, position) -> jnp.ndarray:
    col1 = jnp.concatenate((rotation.at[0].get(), jnp.array([position[0]])))
    col2 = jnp.concatenate((rotation.at[1].get(), jnp.array([position[1]]))) 
    col3 = jnp.concatenate((rotation.at[2].get(), jnp.array([position[2]])))
    col4 = jnp.array([ 0, 0, 0, 1])
    transformation_vector = jnp.concatenate(arrays=(col1, col2, col3, col4)) 
    transformation_matrix = jnp.reshape(transformation_vector, (-1, 4))
    return transformation_matrix
# f: R^2 -> R^4x4
# q => x (where x is the joint position)
# a lil harcoded and adapted to this robot configuration
def calculate_fk(q: jnp.ndarray, r: float) -> jnp.ndarray:
    # rotation matrices not really working for me at the moment, more hardcoded solution now
    # joint 1 
    """ rotation1 = create3d_rotation_matrix(q.at[0].get())
    pos1 = create_position_vector(r=r, axis=0)
    transformation_1 = build_transformation_matrix(rotation=rotation1, position=pos1)
    # join 2
    rotation2 = create3d_rotation_matrix(q.at[1].get())
    pos2 = create_position_vector(r=q.at[1].get(), axis=2)
    transformation_2 = build_transformation_matrix(rotation=rotation2, position=pos2)

    transformation = jnp.matmul(transformation_1, transformation_2)
    position = transformation[:, -1] """
    return jnp.array([0, (r*jnp.cos(q.at[0].get())) + (r*jnp.cos(q.sum())), (r*jnp.sin(q.at[0].get())) + (r*jnp.sin(q.sum()))]) 

def loss(q: float, r: float, desired_pose: jnp.ndarray):
    fk = calculate_fk(q, r)
    return desired_pose - fk

# Desired pose, pose_d:
# [R_d p_d]
# [0_1x3 1]
# Current pose, pose_k:
# [R_k p_k]
# [0_1x3 1]


def calculate_ik_step(q: jnp.ndarray, length: float, pose_d: jnp.ndarray, pose_k: jnp.ndarray, alpha = 1e-3) -> jnp.ndarray:
    # TODO: calculate df/dq
    q1 = q.at[0].get()
    q2 = q.at[1].get()
    # Only for demonstration purposes, all exists in d_fk
    dq_1 = jnp.array([
                    0, 
                    -length*jnp.sin(q1) - length*jnp.sin(q1+q2),
                    length*jnp.cos(q1) + length*jnp.cos(q1+q2),
    ])
    dq_2 = jnp.array([
        0,
        -length*jnp.sin(q1+q2),
        length*jnp.cos(q1+q2)
    ])

    d_fk = jnp.array([
                    [ 
                    -length*jnp.sin(q1) - length*jnp.sin(q1+q2),
                    length*jnp.cos(q1) + length*jnp.cos(q1+q2)], 
                    [
                    -length*jnp.sin(q1+q2),
                    length*jnp.cos(q1+q2)]])
    d_fk_inv = jnp.linalg.pinv(a=d_fk) 
    position_error = pose_d - pose_k
    return [q - alpha * jnp.matmul(d_fk_inv, position_error[-2:]), position_error]

def calculate_ik(initial_q: jnp.ndarray, length: float, desired_position: jnp.ndarray, alpha = 1e-3):
    q = initial_q
    current_error = desired_position - (calculate_fk(q, r))
    errors = []
    iterations = 0
    while current_error.sum() > 0.07 and iterations < 10000:
        curr_pos = calculate_fk(q, length)
        q, error = calculate_ik_step(q=q, length=length, alpha=alpha, pose_k=curr_pos, pose_d=desired_position)
        current_error = error
        errors.append(current_error.sum())
        print(current_error.sum())
        iterations += 1

    
    return q

q = jnp.array([jnp.pi/2, jnp.pi/4])
r = 0.2 

desired_position = jnp.array([0.3, 0, 0.2])
r = 0.2
desired_position = jnp.array([0.3, 0.1, 0.2])
initial_q = np.random.uniform(-1, 1, size=(2,))
""" ik = calculate_ik(initial_q=jnp.array(initial_q), length=r, desired_position=desired_position, alpha=1e-4)

import matplotlib.pyplot as plt
plt.plot(errors)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error vs. Iteration')
plt.show()
print(q) """
""" 
errors = []
for i in range(1000):
    fk = calculate_fk(q, r)
    current_position = fk

    print(f'Previous joint configs: {q}')
    ik = calculate_ik_step(q, r, desired_position, current_position)
    print(f'New joint configs: {ik}')
    q = ik

    error = desired_position - fk
    errors.append(error)

    print(f'Error: {error}') """