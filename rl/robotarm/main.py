import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
import mediapy as media

xml = """
<mujoco model="two_joint_arm">
    <compiler angle="degree" inertiafromgeom="true"/>
    <option timestep="0.01" gravity="0 0 -9.81"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" pos="0 0 -0.1" size="1 1 0.1" type="plane" rgba="0.8 0.9 0.8 1"/>
        <body name="base" pos="0 0 0">
            <geom name="base" type="cylinder" size="0.05 0.02" rgba="0.2 0.2 0.2 1"/>
            <body name="upper_arm" pos="0 0 0.02">
                <joint name="shoulder" type="hinge" axis="0 0 1" range="-90 90"/>
                <geom name="upper_arm" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" rgba="0.7 0.7 0 1"/>
                <body name="lower_arm" pos="0.2 0 0">
                    <joint name="elbow" type="hinge" axis="0 1 0" range="-90 90"/>
                    <geom name="lower_arm" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" rgba="0 0.7 0.7 1"/>
                    <site name="end_effector" pos="0.2 0 0" size="0.01" rgba="1 0 0 1"/>
                </body>
            </body>
        </body>
        <body name="target" pos="0.3 0 0.2">
            <geom name="target" type="sphere" size="0.02" rgba="1 0 0 0.5"/>
        </body>
    </worldbody>
    <actuator>
        <motor joint="shoulder" ctrlrange="-1 1" ctrllimited="true"/>
        <motor joint="elbow" ctrlrange="-1 1" ctrllimited="true"/>
    </actuator>
    <sensor>
        <touch name="touch_sensor" site="end_effector"/>
    </sensor>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
while mj_data.time < duration:
  mujoco.mj_step(mj_model, mj_data)
  if len(frames) < mj_data.time * framerate:
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)

# Simulate and display video.
media.show_video(frames, fps=framerate)