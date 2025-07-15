from jax import numpy as jnp
import jax
from scipy.optimize import minimize
import numpy as np
import math


# Inverse kinematics for a particular 3-DOF arm
# See mujoco file for specifics
# Position only solver uisng an optimization based approach
class IKSolver:
    def __init__(self, desired_pose: jnp.ndarray, initial_q: jnp.ndarray, learning_rate: float, joint_lengths: jnp.ndarray) -> None:
        self.desired_position = desired_pose
        self.q = initial_q
        self.learning_rate = learning_rate
        self.joint_lengths = joint_lengths


    def forward_kinematics(self, q):
        # in this example all arms are of same length, could make it a bit more nicer but yeah
        L = 0.5
        q1, q2, q3 = q[0], q[1], q[2]

        # we only care about positions so yeah, this is very simplified
        position = jnp.array([
            jnp.cos(q1) * (L * jnp.sin(q2) + L * jnp.sin(q2 + q3)),
            jnp.sin(q1) * (L * jnp.sin(q2) + L * jnp.sin(q2 + q3)),
            L * (1 + jnp.cos(q2) + jnp.cos(q3 + q2)), 
        ])
        return position

    def loss(self, q, q0):
        return jnp.sum((q-q0)**2)

    def iterative_solver(self, q0):
        q = q0
        error = self.desired_position - self.forward_kinematics(q)

        jacobian_function = jax.jacobian(self.forward_kinematics)
        while jnp.linalg.norm(error) >= 0.01:
            jacobian = jacobian_function(q) 
            gradient = jnp.linalg.pinv(jacobian).dot(error)
            q += 0.01 * gradient 
            error = self.desired_position - self.forward_kinematics(q) 
        return q