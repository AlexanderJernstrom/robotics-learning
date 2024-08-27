# Robotics learning

This is a quite messy repo with a bunch of code snippets and notebooks related to tracking my progress in robotics and RL.

# Projects

Currently contains two small little projects outlined below

## "Robot" arm

This is a quite messy repo with a bunch of code snippets and notebooks related to tracking my progress in robotics and RL.

# Projects

Currently contains two small little projects outlined below

## "Robot" arm

An exetremely simple two joint robot arm simulated in Mujoco. Uses a custom built inverse kinematics solver (uses optimization) to move the arm to a certain target. Implemented in Jax and MJX. Kind of works but still a bit janky and could be made much more dynamic. E.g relies on hand calculating Jacobian which could be done using rotation matrices (on the roadmap). Can be found in `/rl/robotarm` with the following files:

- `/rl/robotarm/mjx.ipynb` - Notebook responsible for running the simulation in Mujoco
- `/rl/robotarm/robot_arm.xml`- Mujoco XML file describing the robot arm
- `/rl/robotarm/lib.py` - Various functions related to the inverse kinematics of the arm

TODO:

- Simple description of the math involved and how it was done

## Multiarm bandit

A very simple simulation of the multiarm bandit where a greedy and an \(\epsilon\)-greedy method were compared. This exercise is taken from the book "Reinforcement Learning: An Introduction" by Richard Sutton and Andrew Barto from the section on Multi-arm bandits. My first real Reinforcement learning project, something is wrong since \(\epsilon\) method should perform much better, also on the roadmap to fix. Most of the code lies in `/rl/multiarmbandit.ipynb`.
