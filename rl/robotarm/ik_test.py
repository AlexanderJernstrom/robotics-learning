import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
import numpy as np
from ik import IKSolver
from PIL import Image


xml = """
<mujoco model="3dof_robot_arm">
    <compiler angle="radian" coordinate="local"/>
    <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>

    <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" pos="0 0 -0.1" size="1 1 0.1" type="plane" rgba="0.8 0.9 0.8 1"/>
         
        <!-- Base of the robot arm -->
        <body name="base" pos="0 0 0">
            <!-- Joint 1: Base rotation -->
            <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-180 180"/>
            <!-- Link 1 -->
            <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.5" size="0.05" mass="1"/>
            <!-- Link 2 -->
            <body name="link2" pos="0 0 0.5">
                <!-- Joint 2: Shoulder pitch (up/down) -->
                <joint name="joint2" type="hinge" axis="0 1 0" limited="true" range="-90 90"/>
                <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 0.5" size="0.05" mass="0.8"/>
                <!-- Link 3 -->
                <body name="link3" pos="0 0 0.5">
                    <!-- Joint 3: Elbow pitch (up/down) -->
                    <joint name="joint3" type="hinge" axis="0 1 0" limited="true" range="-90 90"/>
                    <geom name="link3_geom" type="capsule" fromto="0 0 0 0 0 0.5" size="0.05" mass="0.6"/>
                    <!-- End-effector -->
                    <body name="end_effector" pos="0 0 0.5">
                        <geom type="sphere" size="0.05" rgba="1 0 0 1" mass="0.2"/>
                        <site name="end_eff" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
                    </body>
                </body>
            </body>
        </body>
<body name="target" pos="0.3 0.3 1">
            <geom name="target" type="sphere" size="0.02" rgba="0 0 1 0.5"/>
        </body>
    </worldbody>

    <actuator>
        <motor joint="joint1" ctrlrange="-1 1" ctrllimited="true"/>
        <motor joint="joint2" ctrlrange="-1 1" ctrllimited="true" gear="10"/>
        <motor joint="joint3" ctrlrange="-1 1" ctrllimited="true" gear="10"/>
    </actuator>
   <sensor>
    <jointpos name="joint1_pos" joint="joint1"/>
    <jointpos name="joint2_pos" joint="joint2"/>
    <jointpos name="joint3_pos" joint="joint3"/>
    <framepos name="end_effector_pos" objtype="body" objname="end_effector"/>
</sensor> 
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(mj_model, camera)
camera.distance = 2
target = jnp.array([0.3, 0.3, 1], dtype="float64")
target_point = np.array([0.3, 0.3, 1], dtype="float64")

q0 = jnp.array([0, -jnp.pi/2, 0])

solver = IKSolver(desired_pose=target, initial_q=q0, learning_rate=0.001, joint_lengths=jnp.array([]))
#q = solver.solve(q0=q0)
q = solver.iterative_solver(q0=q0)
print("iterative q", q)


mj_data.qpos = q0
mujoco.mj_forward(mj_model, mj_data)
renderer.update_scene(mj_data, camera)
target_plot = renderer.render()

mj_data.qpos = q 
mujoco.mj_forward(mj_model, mj_data)
result_point = mj_data.body('end_effector').xpos
renderer.update_scene(mj_data, camera)
result_plot = renderer.render()

target_img = Image.fromarray(target_plot)
target_img.save("initial.png")
result_img = Image.fromarray(result_plot)
result_img.save("result.png")
print(result_point, "result")
print(target, "target")
print(jnp.abs(target - result_point))