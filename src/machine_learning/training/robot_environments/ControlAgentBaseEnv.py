import math
import mujoco
import numpy as np
import pathlib

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 1.25,
    "elevation": -25,
    "azimuth": 45,
}

PITCH_MAX = 0.25
PITCH_DOT_MAX = 1
WHEEL_SPEED_MAX = 170.0
WHEEL_SPEED_DELTA_MAX = 4.0

"""
Most of the Env code will be common across different scenarios as the robot
doesn't change. The base class includes all this common code.
"""
class ControlAgentBaseEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 200,
    }

    def __init__(self, env_filename: str, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

    # From Original Repo
        # order of observation space
        # pitch
        # pitch dot
        # left wheel speed (actual)
        # right wheel speed (actual)
        # difference in target wheel speed to actual wheel speed

    # From ZSB Team
        # difference in target yaw to actual yaw

        observation_space = Box(
            np.array([-math.pi * 2, -math.pi * 2,  -1.0, -1.0, -1.0, -math.pi]),
            np.array([math.pi * 2, math.pi * 2, 1.0, 1.0, 1.0, math.pi]),
            dtype=np.float32
        )

        MujocoEnv.__init__(
            self,
            str(pathlib.Path(__file__).parent.joinpath(env_filename)),
            250,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=800,
            height=800,
            **kwargs,
        )

        self.loop_count = 0
        self.last_time = None
        self.last_pitch = None

        self.target_wheel_speed = 0.0
        self.target_yaw = 0.0

    def _set_action_space(self):
        # called by init in parent class
        # normalize the action space to better support quantization later

        # order of action space
        # left wheel speed, right wheel speed
        self.action_space = Box(
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            dtype=np.float32
        )
        return self.action_space

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qpos[2] = 0

        # face a random direction
        x_rot = (np.random.random() - 0.5) * 2 * math.pi
        # rotate and pitch slightly
        y_rot = (np.random.random() - 0.5) * 0.4
        z_rot = (np.random.random() - 0.5) * 0.4
        euler_angles = [x_rot, y_rot, z_rot]
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        qpos[3:7] = rotation.as_quat()

        qvel = self.init_qvel

        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        reward = self._get_reward()

        vel_l = self.data.joint('torso_l_wheel').qvel[0] + a[0] * WHEEL_SPEED_DELTA_MAX
        vel_r = self.data.joint('torso_r_wheel').qvel[0] + a[1] * WHEEL_SPEED_DELTA_MAX

        self.data.actuator('motor_l_wheel').ctrl = [vel_l]
        self.data.actuator('motor_r_wheel').ctrl = [vel_r]
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)

        self._update_camera_follow()

        # terminate if pitch is greater than 50deg
        terminated = np.abs(self.get_pitch()) > (50 * math.pi / 180)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, False, {}

    def _update_camera_follow(self):
        # get robot position
        pos = self.data.body("robot_body").xpos
        if self.unwrapped.mujoco_renderer.viewer is not None:
            # viewer is None first time render is called!
            v = self.unwrapped.mujoco_renderer.viewer
            # Adjust the camera to follow the robot
            v.cam.lookat[:] = pos
            v.cam.distance = 1.25

    def render(self):
        if self.mujoco_renderer.viewer is not None and self.render_mode == 'human':
            # in this case it's a Gymnasium Mujoco Viewer
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Pitch",
                text2="{:.2f}".format(self.get_pitch() * 180 / math.pi)
            )
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Speed",
                text2="{:.2f}".format(self.get_wheel_speed())
            )
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Target",
                text2="{:.2f}".format(self.target_wheel_speed)
            )
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Yaw",
                text2="{:.2f}".format(self.get_yaw())
            )
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="Target Yaw",
                text2="{:.2f}".format(self.target_yaw)
            )

        frame = super().render()

        if self.render_mode == 'rgb_array' and frame is not None:
            frame = self._annotate_frame(frame)

        return frame

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        # burns target/actual speed and yaw into the top-left corner of a
        # rendered rgb_array frame, so they show up in recorded training videos
        # (the mujoco viewer overlay used in "human" mode above isn't captured
        # by RecordVideo, since that wrapper only ever sees the raw pixel array)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=20)

        lines = [
            "Target Speed: {:.2f}".format(self.target_wheel_speed),
            "Actual Speed: {:.2f}".format(self.get_wheel_speed()),
            "Target Yaw: {:.2f}".format(self.target_yaw),
            "Actual Yaw: {:.2f}".format(self.get_yaw()),
        ]

        y = 10
        for line in lines:
            draw.text((10, y), line, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            y += 24

        return np.array(image)

    def get_pitch(self) -> float:
        quat = self.data.body("robot_body").xquat
        if quat[0] == 0:
            return 0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Quaternion order is [x, y, z, w]
        angles = rotation.as_euler('xyz', degrees=False)
        # print(angles)
        return angles[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        # print(angular)
        return angular[0]

    def get_pitch_dot_alt(self) -> float:
        # alternate method of calculating pitch dot, this is how the real
        # robot does it
        pitch = self.get_pitch()
        ts = self.data.time

        pitch_dot = 0
        if self.last_time is not None and self.last_pitch is not None:
            dt = ts - self.last_time
            if dt > 0.0:
                pitch_dot = (pitch - self.last_pitch) / dt

        self.last_time = ts
        self.last_pitch = pitch

        return pitch_dot

    def get_wheel_velocities(self) -> tuple[float, float]:
        vel_m_0 = self.data.joint('torso_l_wheel').qvel[0]
        vel_m_1 = self.data.joint('torso_r_wheel').qvel[0]

        # both wheels spin "forward", but one is spinning in a negative
        # direction as it's rotated 180deg from the other
        return (vel_m_0, vel_m_1)

    def get_wheel_speed(self) -> float:
        vel_l, vel_r = self.get_wheel_velocities()
        wheel_speed = (vel_l + (-1 * vel_r)) / 2
        return wheel_speed

    def get_yaw(self) -> float:
        quat = self.data.body("robot_body").xquat
        if quat[0] == 0:
            return 0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Quaternion order is [x, y, z, w]
        angles = rotation.as_euler('xyz', degrees=False)
        return angles[2]

    def get_wheel_yaw(self) -> float:
            vel_l, vel_r = self.get_wheel_velocities()
            wheel_yaw = vel_l - (-1 * vel_r)
            return wheel_yaw

    def _get_obs(self):
        self.loop_count += 1

        pitch = self.get_pitch()
        pitch_dot = self.get_pitch_dot_alt()
        wheel_vel_l, wheel_vel_r = self.get_wheel_velocities()

        pitch_normalized = pitch / PITCH_MAX
        pitch_dot_normalized = pitch_dot / PITCH_DOT_MAX
        wheel_vel_l_normalized = wheel_vel_l / WHEEL_SPEED_MAX * 4
        wheel_vel_r_normalized = wheel_vel_r / WHEEL_SPEED_MAX * 4
        delta_wheel_speed_normalized = (self.target_wheel_speed - self.get_wheel_speed()) / WHEEL_SPEED_MAX * 4
        delta_wheel_yaw_normalized = (self.target_yaw - self.get_yaw() + np.pi) % (2*np.pi) - np.pi

        return np.array(
            [
                pitch_normalized,
                pitch_dot_normalized,
                wheel_vel_l_normalized,
                wheel_vel_r_normalized,
                delta_wheel_speed_normalized,
                delta_wheel_yaw_normalized
            ],
            dtype=np.float32
        ).ravel()

        # obs_clip = np.clip(obs, a_min = -1.0, a_max = 1.0)
        # print(obs_clip)
        # return obs_clip
