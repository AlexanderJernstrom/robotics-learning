import jax
from jax import numpy as jnp

# state = [x, y, z, roll, pitch, yaw]
# control inputs = [u1, u2, u3, u4]
class LQR:
    def __init__(self, reference) -> None:
        self.u = jnp.zeros(4) 
        self.state = jnp.zeros(6)
        self.performance_weight = jnp.zeros((6, 6))
        self.cost_weight = jnp.zeros((4, 4))
        self.A = jnp.zeros((6, 6))
        self.B = jnp.zeros((4, 6))
        self.reference = reference
        self.gain = jnp.zeros((6, 4))

    def cost(self):

        pass
        
    def set_state(self, state: jnp.ndarray):
        self.state = state

    def state_step(self):
        x_dot = (self.A - self.B * self.gain) * self.state
        
        pass

    def optimal_control(self):
        # derivera loss med avseende på K
        # iterera ner i "dalen"

        return jnp.dot(-self.gain, self.state)

    
    def control_loop(self):
        x_dot = self.A * self.state + self.B * self.u
        # need to solve the diff equation to obtian the new 
        x = x_dot # do some operation to solve this

        self.state = x
        u = self.optimal_control()
        return u