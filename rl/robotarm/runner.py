import mujoco
import mujoco.viewer
import jax.numpy as jnp
from ik import IKSolver

xml = """
<mujoco model="3dof_robot_arm">
    <compiler angle="degree" coordinate="local"/>
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
                        <site name="end_eff" pos="0.2 0 0" size="0.01" rgba="1 0 0 1"/>
                    </body>
                </body>
            </body>
        </body>
<body name="target" pos="0.3 0.3 1">
            <geom name="target" type="sphere" size="0.02" rgba="2 0 0 0.5"/>
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

target = jnp.array([0.3, 0.3, 1])

solver = IKSolver(desired_pose=target, initial_q=jnp.array([0, -jnp.pi/2, jnp.pi/2]), learning_rate=0.001, joint_lengths=jnp.array([]))


with mujoco.viewer.launch_passive(
    model=mj_model, data=mj_data, show_left_ui=False
) as viewer:
    mujoco.mjv_defaultFreeCamera(mj_model, viewer.cam)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    viewer.sync()

    mujoco.mj_forward( mj_model, mj_data)
    renderer.update_scene(mj_data, viewer.cam)
    result_plot = renderer.render()
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        mujoco.mj_camlight(mj_model, mj_data)
        viewer.sync()
        