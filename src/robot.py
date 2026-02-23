"""
A simulated robotic agent with teleoperation and sensing capabilities.

The Robot class models the robotic agent that explores the world. The robot is remote-controlled by angular and linear velocity commands read from an external file. The robot can execute motor commands to move, and can sense both externally (GPS, landmarks, obstacles) and internally (odometry, IMU).
"""

import sensors
import pandas as pd
import random
import math
from utils import floating_mod_zero, NEAR_ZERO
from environment import Environment
from sensors import SensorInterface


class Robot:
    """
    A class that models a simulated robotic agent.

    Attributes:
        env: the environment this robot is operating in
        sensors: list of all robot sensors
    """

    def __init__(self, env: Environment):
        """
        Initialize an instance of the Robot class.

        Args:
            env: the environment this robot is operating in
        """
        self.env = env
        self.sensors = [
            sensors.WheelEncoder(self),
            sensors.LandmarkPinger(self),
            sensors.GPS(self),
        ]
        self.linear = False

        if self.linear:
            self.real_x_vel = 0
            self.real_y_vel = 0
            self.cmd_x_vel = 0
            self.cmd_y_vel = 0
        else:
            self.real_lin_vel = 0
            self.cmd_lin_vel = 0

        self.real_ang_vel = 0
        self.cmd_ang_vel = 0

    def robot_step_differential(self, lin_vel: float, ang_vel: float):
        """
        Differential-drive mode. Given forward linear and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            lin_vel: input linear velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """
        self.cmd_lin_vel = lin_vel
        self.cmd_ang_vel = ang_vel

        lin_vel = lin_vel * (
            1 + random.gauss(0, 0.03)
        )  # Note: Move this into a config file
        ang_vel = ang_vel * (1 + random.gauss(0, 0.05))

        self.real_lin_vel = lin_vel
        self.real_ang_vel = ang_vel

        # moving in straight line
        if abs(ang_vel) < NEAR_ZERO:  # Note: move to config file?
            dx = lin_vel * self.env.DT * math.cos(self.env.robot_pose.theta)
            dy = lin_vel * self.env.DT * math.sin(self.env.robot_pose.theta)
            dtheta = 0
        else:
            r = lin_vel / ang_vel
            dtheta = ang_vel * self.env.DT
            dx = r * (
                math.sin(self.env.robot_pose.theta + dtheta)
                - math.sin(self.env.robot_pose.theta)
            )
            dy = r * (
                math.cos(self.env.robot_pose.theta)
                - math.cos(self.env.robot_pose.theta + dtheta)
            )
        # step environment
        self.env.robot_step(dx, dy, dtheta)

    def robot_step_translational(self, x_vel: float, y_vel: float, ang_vel: float):
        """
        Swerve-drive mode. Given x, y, and angular velocities, determine the robot's change in x, y, and heading and apply those changes in the environment.

        Args:
            x_vel: input x velocity command
            y_vel: input y velocity command
            ang_vel: input angular velocity command

        Returns:
            dx: change in x position
            dy: change in y position
            d-theta: change in heading
        """

        self.cmd_x_vel = x_vel
        self.cmd_y_vel = y_vel
        self.cmd_ang_vel = ang_vel

        x_vel *= 1 + random.gauss(0, 0.03)
        y_vel *= 1 + random.gauss(0, 0.03)
        ang_vel *= 1 + random.gauss(0, 0.05)

        self.real_x_vel = x_vel
        self.real_y_vel = y_vel
        self.real_ang_vel = ang_vel

        dx = x_vel * self.env.DT
        dy = y_vel * self.env.DT
        dtheta = ang_vel * self.env.DT

        self.env.robot_step(dx, dy, dtheta)

    def take_sensor_measurements(self):
        """
        Return noisy sensor readings of the environment at this timestep, including data from all sensors, in a table format.
        """
        measurements = pd.DataFrame({"Time": [self.env.time]})
        for sensor in self.sensors:
            if floating_mod_zero(self.env.time, sensor.interval):
                measurements = pd.merge(
                    measurements, sensor.sample(), left_index=True, right_index=True
                )

        if self.linear:
            measurements["CMD_X_Velocity"] = [self.cmd_x_vel]
            measurements["CMD_Y_Velocity"] = [self.cmd_y_vel]
        else:
            measurements["CMD_LinearVelocity"] = [self.cmd_lin_vel]
        measurements["CMD_AngularVelocity"] = [self.cmd_ang_vel]

        return measurements

    def take_gt_snapshot(self) -> pd.DataFrame:
        """
        Return timestep-specific GT data for CSV logging.
        """
        # grab env data: time, robot pose, gt to landmarks
        env_data = self.env.take_state_snapshot()

        if self.linear:
            # add in actual, imperfect velocity commands
            env_data["Actual_X_Velocity"] = self.real_x_vel
            env_data["Actual_Y_Velocity"] = self.real_y_vel
            env_data["Actual_AngularVelocity"] = self.real_ang_vel
        else:
            # add in actual, imperfect velocity commands
            env_data["Actual_LinearVelocity"] = self.real_lin_vel
            env_data["Actual_AngularVelocity"] = self.real_ang_vel

        return env_data
