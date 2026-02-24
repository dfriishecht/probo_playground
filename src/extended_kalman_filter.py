"""
Extended Kalman Filter implementation for the simulator. Tracks the following states:

x = [x, y, theta]

We expect the following control inputs:

u = [v, w]
"""

import numpy as np
import sympy
from sympy.abc import x, y, v, w, R, theta
from sympy import Matrix, Symbol
import random

from utils import wrap_angle


class ExtendedKalmanFilter:
    """
    This class implements the Extended Kalman Filter algorithm.
    """

    def __init__(self, dt: float, prior: np.ndarray):
        """
        Initialize an Extended Kalman Filter.

        A state vector includes the following:
            x position
            y position
            heading


        Args:
            dt: the length of each timestep, in seconds
            prior: the initial estimates for each state variable-
        """
        self.DT: float = dt
        self.x_state: np.ndarray = np.array(
            [[prior.pos.x, prior.pos.y, prior.theta]]
        ).flatten()
        self.P: np.ndarray = np.eye(3)

        self.f_xu: Matrix = Matrix(
            [
                [x + v * sympy.cos(theta) * self.DT],  # calculation of x
                [y + v * sympy.sin(theta) * self.DT],  # calculation of y
                [theta + w * self.DT],  # calculation of theta
            ]
        )

        self.F: Matrix = self.f_xu.jacobian([x, y, theta])

        # dictionary that maps Sympy symbols to numerical values.
        self.subs: dict[Symbol, float] = {
            x: self.x_state[0],
            y: self.x_state[1],
            theta: self.x_state[2],
            v: 0,
            w: 0,
        }

    def predict(self, u: np.ndarray):
        """
        Predicts the next state vector and its covariance matrix using the state transition matrix and an input control vector. The Kalman Filter uses the following predict equations:

        x_t+1 = f(x,u)
        P_t+1 = F * P * F.T + Q

        where F is the Jacobian of f(x,u)

        Args:
            u: the input control vector
        """
        self.subs[x] = float(np.array(self.x_state[0]).item())
        self.subs[y] = float(np.array(self.x_state[1]).item())
        self.subs[theta] = float(np.array(self.x_state[2]).item())
        self.subs[v] = float(np.array(u[0]).item())
        self.subs[w] = float(np.array(u[1]).item())

        fxu_eval = np.array(self.f_xu.subs(self.subs)).astype(float).flatten()

        F_eval = np.array(self.F.subs(self.subs)).astype(float)

        self.x_state = fxu_eval

        Q = self.get_Q()
        self.P = F_eval @ self.P @ F_eval.T + Q

        self.x_state[2] = wrap_angle(self.x_state[2])
        return self.x_state, self.P

    def update(
        self,
        H: np.ndarray,
        R: np.ndarray,
        z: np.ndarray | None,
        y: np.ndarray | None,
    ):
        """
        Updates the current state prediction using observations from the environment. The Extended Kalman Filter uses the following update equations:

        x = x + K * y
        P = P - K * H * P

        Where K and y are given by the following:
        y = z - h(x) (residual: error between observation and expected observation given estimated state vector)
        K = P * H.T * inv(S) (Kalman Gain: portion of total uncertainty that is from the prediction)
        S = H * P * H.T + R (total uncertainty in the system)

        where H is the Jacobian of h(x)

        Args:
            H: the Jacobian of the nonlinear measurement model, which relates the state space to the measurement space
            R: the measurement noise model (covariance)
            y: the residual, which is the error between the measured observation and the observation expected by the predicted state
        """
        S = H @ self.P @ H.T + R

        K = self.P @ H.T @ np.linalg.inv(S)

        if y is None:
            y = z.T - H @ self.x_state

        self.x_state = (self.x_state.reshape(3, 1) + K @ y.reshape(2, 1)).flatten()

        self.P = self.P - K @ H @ self.P

        self.x_state[2] = wrap_angle(self.x_state[2])

        return self.x_state, self.P

    def get_Q(self):
        """
        Generate white noise to apply to the process model after each prediction.
        """
        stdev = 0.01
        return np.array(
            [
                [
                    abs(random.gauss(0, stdev)),
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                ],
                [
                    random.gauss(0, stdev),
                    abs(random.gauss(0, stdev)),
                    random.gauss(0, stdev),
                ],
                [
                    random.gauss(0, stdev),
                    random.gauss(0, stdev),
                    abs(random.gauss(0, stdev)),
                ],
            ]
        )
