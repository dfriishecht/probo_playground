"""
An abstract base class that all sensor classes must inherit from. This structure guarantees that all sensors have certain traits, including a name, sampling interval, and sampling function.

In addition to basic features, all sensors should have noise constants. Different sensors may use different distributions to model noise, and may take in different parameters to shape that noise. For example, one sensor might have a constant noise mean, while another might have noise that grows proportionally with distance or time.

Exteroceptive sensors measure the robot's relationship to the world. This includes GPS, cameras, LiDAR, and anything else that takes a measurement that can relate the robot's state to things beyond the robot.

Proprioceptive sensors measure the robot's relationship to its past states. This includes IMUs, wheel encoders, and anything else that measures how the robot's state is relatively changing, without relating the robot to the world.
"""

from abc import ABC, abstractmethod
from math import pi
import math
import numpy as np
import pandas as pd
import random
from utils import BearingRange, Position
import sympy
from sympy.abc import x, y, k, j, theta
from sympy import symbols, Matrix, Symbol, pprint


class SensorInterface(ABC):
    """
    A basic Interface to standardize all sensors.

    Attributes:
        name: string identifier
        robot: reference robot. required for observing the environment
        interval: period between measurements
        last_meas_t: time of last sensor measurement
    """

    def __init__(self, name: str, robot, interval: float):
        """
        Initialize a sensor class instace.

        Args:
            name: reference identifier
            robot: reference robot
            interval: period between measurements
        """
        self._name = name
        self.robot = robot
        self._interval = interval
        self.last_meas_t = robot.env.time

    @property
    def name(self) -> str:
        """
        Getter for the name property.
        """
        return self._name

    @property
    def interval(self) -> float:
        """
        Getter for the interval property.
        """
        return self._interval

    @property
    def last_meas_t(self) -> float:
        """
        Getter for the time of last measurement property.
        """
        return self._last_meas_t

    @last_meas_t.setter
    def last_meas_t(self, value: float):
        """
        Setter for the time of last measurement property.
        """
        self._last_meas_t = value

    @abstractmethod
    def sample(self):
        """
        Sample the environment and return the noisy measurement(s).
        """
        pass


class WheelEncoder(SensorInterface):
    """
    This class represents a wheel encoder set that measures the robot's motor speeds.
    Reports noisy estimates of linear and angular velocities.

    Attributes:
        name: string identifier
        robot: reference robot
        interval: period between measurements
        last_meas_t: time of last measurement
        LIN_NOISE: absolute noise for linear velocity stdev
        ANG_NOISE: absolute noise for angular velocity stdev
    """

    def __init__(
        self,
        robot,
        name="wheel_encoder",
        interval=0.1,
        lin_noise=0.05,
        ang_noise=0.03,
    ):
        """
        Initialize an instance of the WheelEncoder class.

        Args:
            robot: reference robot
            name: reference identifier
            interval: period between measurements
            linear_noise_ratio: proportional noise for linear velocity
            angular_noise_ratio: proportional noise for angular
        """
        super().__init__(name, robot, interval)
        self.LIN_NOISE = lin_noise  # m/s
        self.ANG_NOISE = ang_noise  # rad/s
        self.LIN_NOISE_PROPORTION = 0.01
        self.ANG_NOISE_PROPORTION = 0.1

    def sample(self):
        """
        Sample the robot's linear and angular velocity.
        """
        if self.robot.linear:
            last_x_vel = self.robot.real_x_vel
            last_y_vel = self.robot.real_y_vel
            last_ang_vel = self.robot.real_ang_vel

            noisy_x_vel = random.gauss(
                last_x_vel, self.LIN_NOISE + abs(last_x_vel) * self.LIN_NOISE_PROPORTION
            )
            noisy_y_vel = random.gauss(
                last_y_vel, self.LIN_NOISE + abs(last_y_vel) * self.LIN_NOISE_PROPORTION
            )
            noisy_ang_vel = random.gauss(
                last_ang_vel,
                self.ANG_NOISE + abs(last_ang_vel) * self.ANG_NOISE_PROPORTION,
            )

            return pd.DataFrame(
                {
                    f"{self.name}_XVelocity": [noisy_x_vel],
                    f"{self.name}_YVelocity": [noisy_y_vel],
                    f"{self.name}_AngularVelocity": [noisy_ang_vel],
                }
            )

        else:
            last_lin_vel = self.robot.real_lin_vel
            last_ang_vel = self.robot.real_ang_vel

            noisy_lin_vel = random.gauss(
                last_lin_vel,
                self.LIN_NOISE + abs(last_lin_vel) * self.LIN_NOISE_PROPORTION,
            )
            noisy_ang_vel = random.gauss(
                last_ang_vel,
                self.ANG_NOISE + abs(last_ang_vel) * self.ANG_NOISE_PROPORTION,
            )

            return pd.DataFrame(
                {
                    f"{self.name}_LinearVelocity": [noisy_lin_vel],
                    f"{self.name}_AngularVelocity": [noisy_ang_vel],
                }
            )


class LandmarkPinger(SensorInterface):
    """
    This class represents a sensor that measures the range and bearing between the robot and the floating-point landmarks on the map. In practice, this sensor could be a ToF sensor, a node in a network of beacons, or even a camera.

    Attributes:
        name: reference identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        MAX_RANGE (int): maximum distance from a beacon for it to be visible
        RANGE_NOISE (float): absolute noise for range stdev
        RANGE_NOISE_RATIO (float): porportional noise for range stdev
        BEARING_NOISE (float): absolute noise for bearing stdev
    """

    def __init__(
        self,
        robot,
        name="landmark_pinger",
        interval=1.0,
        range_noise=0.05,
        range_prop_noise=0.01,
        bearing_noise=pi / 6,
        max_range=10.0,
    ):
        """
        Initialize an instance of the LandmarkPinger class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
        """
        super().__init__(name, robot, interval)
        self.MAX_RANGE = max_range  # meters
        self.RANGE_NOISE = range_noise  # meters
        self.RANGE_PROP_NOISE = range_prop_noise
        self.BEARING_NOISE = bearing_noise  # radians

        # TODO: define the nonlinear measurement model symbolically
        self.h_x: Matrix = Matrix(
            [
                [sympy.sqrt((x - k) ** 2 + (y - j) ** 2)],  # calculation of r (range)
                [sympy.atan2(y - j, x - k)],  # calculation of phi (bearing)
            ]
        )

        # TODO: define the Jacobian of h(x) symbolically
        self.H: Matrix = self.h_x.jacobian([x, y, theta])

        self.subs: dict[Symbol, float] = {
            x: 0.0,
            y: 0.0,
            k: 0.0,
            j: 0.0,
        }

    def sample(self):
        """
        Reports noisy measurements of the bearing and range between the robot and all nearby landmarks.
        """
        true_distances = self.robot.env.get_proximity_to_landmarks()
        noisy_landmarks = pd.DataFrame()
        for id, landmark in true_distances.items():
            gt_bearing = landmark
            if gt_bearing.range <= self.MAX_RANGE:
                bearing_noisy = BearingRange(
                    id,
                    random.gauss(gt_bearing.bearing, self.BEARING_NOISE),
                    random.gauss(
                        gt_bearing.range,
                        self.RANGE_NOISE + self.RANGE_PROP_NOISE * gt_bearing.range,
                    ),
                )
            else:
                bearing_noisy = BearingRange(id, math.inf, math.inf)
            noisy_landmarks[f"{self.name}_{id}"] = [bearing_noisy]
        return noisy_landmarks

    def R(self, z):
        """
        Estimate variance of a given pinger measurement.

        Args:
            z (ndarray): pinger observation [[range 0], [0 bearing]]

        Returns:
            Sensor noise model for pinger measurement
        """
        bearing_stdev = self.BEARING_NOISE
        range_stdev = (
            self.RANGE_NOISE + float(np.array(z[0]).item()) * self.RANGE_PROP_NOISE
        )
        return np.diag([range_stdev, bearing_stdev]) ** 2

    def H_eval(self, x_state, lm_id):
        """
        Evaluate the Jacobian of h(x) at x, which reshapes a state vector to be in the observation space. This matrix is used to turn a state prediction into an observation prediction for a specific landmark.

        Args:
            x_state: the current state vector, to linearize with respect to
            lm_id: the ID of the landmark that we are predicting an observation of
        """
        lm_x = self.robot.env.LANDMARKS[lm_id].pos.x
        lm_y = self.robot.env.LANDMARKS[lm_id].pos.y

        self.subs[x] = float(np.array(x_state[0]).item())
        self.subs[y] = float(np.array(x_state[1]).item())
        self.subs[theta] = float(np.array(x_state[2]).item())
        self.subs[j] = lm_x  # note: we use j for landmark x position
        self.subs[k] = lm_y  # note: we use k for landmark y position

        H_eval = np.array(self.H.subs(self.subs)).astype(float)

        return H_eval

    def y(self, z, x_state, lm_id):
        """
        Calculate the residual between an observation x and a predicted observation derived from a predicted state. The predicted observation is in reference to a specified landmark.
        """
        lm_x = self.robot.env.LANDMARKS[lm_id].pos.x
        lm_y = self.robot.env.LANDMARKS[lm_id].pos.y

        self.subs[x] = float(np.array(x_state[0]).item())
        self.subs[y] = float(np.array(x_state[1]).item())
        self.subs[theta] = float(np.array(x_state[2]).item())
        self.subs[j] = lm_x  # note: we use j for landmark x position
        self.subs[k] = lm_y  # note: we use k for landmark y position

        hx_eval = np.array(self.h_x.subs(self.subs)).astype(float)

        residual = z - hx_eval

        return residual


class GPS(SensorInterface):
    """
    This class represents a GPS sensor that measures the position of the robot in 2D space.

    Attributes:
        name (str): string identifier
        robot (Robot): reference robot
        interval (float): period between measurements
        last_meas_t (float): time of last measurement
        X_NOISE (float): absolute noise for x stdev
        Y_NOISE (float): absolute noise for y stdev
    """

    def __init__(
        self,
        robot,
        name="gps",
        interval=1.0,
        x_noise=0.5,
        y_noise=0.5,
    ):
        """
        Initialize an instance of the GPS class.

        Args:
            name (str): reference identifier
            robot (Robot): reference robot
            interval (float): period between measurements
            x_noise (float): absolute noise for x stdev
            y_noise (float): absolute noise for y stdev
        """
        super().__init__(name, robot, interval)
        self.X_NOISE = x_noise
        self.Y_NOISE = y_noise

        # Convert x pos, y pos, theta
        self.H = np.array([[1.0, 0, 0], [0, 1, 0]])

        # For now, assume independent noise between GPS x and y measurement
        self.R = np.diag([self.X_NOISE**2, self.Y_NOISE**2])

    def sample(self):
        """
        Take a noisy GPS measurement of robot position.
        """
        gt_pose = self.robot.env.get_robot_pose()
        x_noisy = random.gauss(gt_pose.pos.x, self.X_NOISE)
        y_noisy = random.gauss(gt_pose.pos.y, self.Y_NOISE)

        return pd.DataFrame({self.name: [Position(x_noisy, y_noisy)]})
