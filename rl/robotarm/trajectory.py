import jax
import jax.numpy as jnp
# Trajectory generation using 5th degree polynomial
class Trajectory:
    def __init__(self, constants: jnp.ndarray, horizon: float):
        self.constants = constants 
        self.horizon = horizon

    def _create_coeff_matrix(self):
        T = self.horizon
        A = jnp.array([
            [1, 0, 0,      0,        0,         0      ],
            [0, 1, 0,      0,        0,         0      ],
            [0, 0, 2,      0,        0,         0      ],
            [1, T, T**2,   T**3,     T**4,      T**5   ],
            [0, 1, 2*T,    3*T**2,   4*T**3,    5*T**4 ],
            [0, 0, 2,      6*T,     12*T**2,   20*T**3 ],
        ])
        return A 

    def generate(self) -> jnp.ndarray:
        A = self._create_coeff_matrix()
        polynomial_coefficients = jnp.linalg.solve(A, self.constants)
        return polynomial_coefficients

