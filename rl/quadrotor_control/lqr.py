import jax
from jax import numpy as jnp
from scipy.linalg import solve_continuous_are, solve_discrete_are
import cvxpy as cp

# state = [x, y, z, roll, pitch, yaw]
# control inputs = [u1, u2, u3, u4]

# Continous time infinite horizon 
class CTIHLQR:
    def __init__(self, reference, A: jnp.ndarray, B: jnp.ndarray, R: jnp.ndarray, Q: jnp.ndarray, initial_state) -> None:
        self.performance_weight = Q 
        self.cost_weight = R 
        self.A = A 
        self.B = B 
        self.reference = reference
        self.gain = None
        self.prev_state = initial_state 
        self.losses = []

    def calculate_gain(self):
        S = solve_continuous_are(self.A, self.B, self.performance_weight, self.cost_weight)
        K = jnp.linalg.pinv(self.cost_weight) @ self.B.transpose() @ S
        self.gain = K

    def cost(self, state: jnp.ndarray, inputs: jnp.ndarray):
        return ((state.transpose() - self.reference.transpose()) @  self.performance_weight @ (state - self.reference)) + (inputs.transpose() @ self.cost_weight @ inputs) 

    def control(self, state: jnp.ndarray):
        self.prev_state = state
        inputs = -self.gain @ (state - self.reference)
        self.losses.append(self.cost(state, inputs))
        return inputs



# Finite horizon
class LQR:
    def __init__(self, reference, A: jnp.ndarray, B: jnp.ndarray, R: jnp.ndarray, Q: jnp.ndarray, initial_state: jnp.ndarray, time_steps: int) -> None:
        self.u = jnp.zeros((time_steps, 4)) 
        self.time_steps = time_steps
        self.state = initial_state 
        self.performance_weight = Q 
        self.cost_weight = R 
        self.A = A 
        self.B = B 
        self.reference = reference
        self.gain = None
        self.outputs = jnp.zeros((1000, 6)) 
        self.D = jnp.zeros((6, 4))
        self.C = create_C_matrix() 

    def calculate_gain(self):
        S = solve_continuous_are(self.A, self.B, self.performance_weight, self.cost_weight)
        K = jnp.linalg.pinv(self.cost_weight) @ self.B.transpose() @ S
        self.gain = K

    def state_step(self, dt: float, time_step):
        x_dot = self.A @ self.state + self.B @ self.u.at[time_step].get() 
        self.state = self.state + dt * x_dot 

    def loss(self):
        return (self.state.transpose() @ self.performance_weight @ self.state) + (self.u.transpose() @ self.cost_weight @ self.u)

    def output(self):
        return self.C @ self.state + self.D @ self.u.at[self.time_steps].get()

    def control(self, time_step: int, dt: float) -> jnp.ndarray:
        if self.gain == None:
            pass 
        inputs = -self.gain @ (self.state) 
        self.u = self.u.at[time_step].set(inputs)
        self.state_step(dt, time_step)
        self.outputs = self.outputs.at[time_step].set(self.output()) 
        return inputs 

    def control_trajectory(self, dt = 0.01):
        if self.gain == None:
            pass
        for i in range(self.time_steps):
            self.control(i, dt)
        return self.u 
        

def control_to_motor_forces(U: jnp.ndarray, d):
    """
    Convert control vector U [U1, U2, U3, U4] to individual motor forces
    
    Parameters:
    U (array): Control vector [U1, U2, U3, U4] where:
        U1 = F1 + F2 + F3 + F4 (total thrust)
        U2 = l(F2 - F4) (roll torque)
        U3 = l(F1 - F3) (pitch torque)
        U4 = c(F1 - F2 + F3 - F4) (yaw torque)
    arm_length (float): Length of quad arm (l)
    drag_coefficient (float): Drag coefficient (c)
    
    Returns:
    array: Individual motor forces [F1, F2, F3, F4]
    """
    mixer = jnp.array([
    [ 0.25,  -d,     -d,    -0.1],  # FR
    [ 0.25,  -d,      d,     0.1],  # FL
    [ 0.25,   d,     -d,     0.1],  # BR
    [ 0.25,   d,      d,    -0.1]   # BL
    ])

    motor_controls = mixer @ U


    other_controls = jnp.array([
        jnp.sum(U), # front right
         U[0] - U[3] - U[1] + U[2], # back right
         U[0] + U[3] - U[1] - U[2], # front left
        U[0] - U[3] + U[1] - U[2] # back left
    ])
    #other_controls = jnp.absolute(other_controls)
    print(f'Another mixing algo: {other_controls}')
    
    return other_controls 
       
# basically want this syntax
""" controller = LQR(0.2, jnp.array((6, 6)), jnp.array((4, 6)), jnp.array((6, 6)), jnp.array((4, 4)))
controller.calculate_gain()

u = controller.control()

ctrl = u """

def create_C_matrix():
    m = jnp.zeros((6, 12))
    m = m.at[0, 0].set(1)
    m = m.at[1, 1].set(1)
    m = m.at[2, 2].set(1)
    m = m.at[3, 6].set(1)
    m = m.at[4, 7].set(1)
    m = m.at[5, 8].set(1)
    return m


def create_state_matrix(m, Ixx, Iyy, Izz, g):
    """
    Creates the state matrix A for the quadrotor linearized dynamics.

    Parameters:
    - m: Mass of the quadrotor
    - Ixx: Moment of inertia about the x-axis
    - Iyy: Moment of inertia about the y-axis
    - Izz: Moment of inertia about the z-axis
    - g: Gravitational acceleration

    Returns:
    - A: The state matrix (12x12 numpy array)
    """
    A = jnp.zeros((12, 12))

    # Position derivatives
    A = A.at[0, 3].set(1)  # dx/dt = vx
    A = A.at[1, 4].set(1)  # dy/dt = vy
    A = A.at[2, 5].set(1)  # dz/dt = vz

    # Velocity derivatives (linearized accelerations)
    A = A.at[3, 7].set(-g)  # ddx/dt = -g * theta
    A = A.at[4, 6].set(g)   # ddy/dt = g * phi

    # Orientation derivatives
    A = A.at[6, 9].set(1)   # d(phi)/dt = p
    A = A.at[7, 10].set(1)  # d(theta)/dt = q
    A = A.at[8, 11].set(1)  # d(psi)/dt = r

    return A

def create_input_matrix(m, Ixx, Iyy, Izz):
    """
    Creates the input matrix B for the quadrotor linearized dynamics.

    Parameters:
    - m: Mass of the quadrotor
    - Ixx: Moment of inertia about the x-axis
    - Iyy: Moment of inertia about the y-axis
    - Izz: Moment of inertia about the z-axis

    Returns:
    - B: The input matrix (12x4 numpy array)
    """
    B = jnp.zeros((12, 4))

    # Vertical acceleration due to total thrust deviation
    B = B.at[5, 0].set(1 / m)  # ddz/dt = -u1 / m

    # Angular accelerations due to torques
    B = B.at[9, 1].set(1 / Ixx)   # dp/dt = u2 / Ixx
    B = B.at[10, 2].set(1 / Iyy)  # dq/dt = u3 / Iyy
    B = B.at[11, 3].set(1 / Izz)  # dr/dt = u4 / Izz

    return B

