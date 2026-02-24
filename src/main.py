"""
Main file for running the simulator.
"""

from environment import Environment
from robot import Robot
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter
from utils import Position, Pose, Landmark, Bounds
from viz import Visualizer
import pandas as pd
import os
import csv
import pickle
import numpy as np
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulator.")
    parser.add_argument("--linear", action="store_true", help="Run with linear filter instead of extended Kalman filter.")
    args = parser.parse_args()
    LINEAR = args.linear

    # set up the environment
    dimensions = Bounds(0, 10, 0, 10)
    dt = 0.1
    obstacles = [
        Bounds(5, 7, 5, 7),
        Bounds(0, 4, 6, 8),
    ]
    landmarks = [
        Landmark(Position(2.0, 2.0), 0),
        Landmark(Position(5.0, 5.0), 1),
        Landmark(Position(8.0, 8.0), 2),
    ]

    initial_robot_pose = Pose(Position(0.2, 0.2), 0.6)

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(env, linear=LINEAR)

    # set up the (Extended) Kalman Filter
    if LINEAR:
        kf = KalmanFilter(
            dt,
            initial_robot_pose,
        )
    else:
        # set up the Extended Kalman Filter
        kf = ExtendedKalmanFilter(
            dt,
            initial_robot_pose,
        )

    # set up timekeeping
    total_seconds = 30
    total_timesteps = total_seconds / env.DT
    terminal = False

    # set up logging
    ground_truth_history = pd.DataFrame()
    sensor_data_history = pd.DataFrame()
    kalman_filter_history = pd.DataFrame()

    # set up input filepath and output filepaths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if LINEAR:
        input_commands_filepath = os.path.join(script_dir, "../input/translational_example.csv")
    else:
        input_commands_filepath = os.path.join(script_dir, "../input/diff_example.csv")
    output_ground_truth_filepath = os.path.join(
        script_dir, "../output/ground_truth.pkl"
    )
    output_sensor_data_filepath = os.path.join(script_dir, "../output/sensor_data.pkl")
    output_env_data_filepath = os.path.join(script_dir, "../output/env_data.pkl")
    output_kalman_filter_filepath = os.path.join(
        script_dir, "../output/kalman_data.pkl"
    )

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
        vel_cmds = csv.reader(cmd)
        next_cmd = next(vel_cmds)
        next_cmd = next(vel_cmds)
        if LINEAR:
            current_x_vel = float(next_cmd[1])
            current_y_vel = float(next_cmd[2])
            current_ang_vel = float(next_cmd[3])
        else:
            current_lin_vel = float(next_cmd[1])
            current_ang_vel = float(next_cmd[2])

        for step in range(int(total_timesteps) + 1):
            ground_truth_history = pd.concat(
                [
                    ground_truth_history,
                    robot.take_gt_snapshot(),
                ],
                ignore_index=True,
            )

            current_sensor_data = robot.take_sensor_measurements()
            sensor_data_history = pd.concat(
                [
                    sensor_data_history,
                    current_sensor_data,
                ],
                ignore_index=True,
            )
            if LINEAR:
                u = np.array(
                    [
                        current_sensor_data["wheel_encoder_XVelocity"],
                        current_sensor_data["wheel_encoder_YVelocity"],
                        current_sensor_data["wheel_encoder_AngularVelocity"],
                    ]
                )

                x, P = kf.predict(u)

                # TODO: call the Kalman Filter update step if new sensor data is available
                if "gps" in current_sensor_data:
                    z = np.array(
                        [
                            [
                                current_sensor_data["gps"][0].x,
                                current_sensor_data["gps"][0].y,
                            ]
                        ]
                    )
                    x, P = kf.update(z, robot.sensors[2].H, robot.sensors[2].R)

                kalman_filter_history = pd.concat(
                    [kalman_filter_history, pd.DataFrame([{"x": x, "P": P}])],
                    ignore_index=True,
                )
            else:
                u = np.array(
                    [
                        current_sensor_data["wheel_encoder_LinearVelocity"],
                        current_sensor_data["wheel_encoder_AngularVelocity"],
                    ]
                )

                x, P = kf.predict(u)

                if "gps" in current_sensor_data:
                    z = np.array(
                        [
                            current_sensor_data["gps"][0].x,
                            current_sensor_data["gps"][0].y,
                        ]
                    )
                    x, P = kf.update(
                        robot.sensors[2].H, robot.sensors[2].R, z=z, y=None
                    )

                for col in current_sensor_data.filter(like="landmark_pinger_").columns:
                    reading = current_sensor_data[col].iloc[0]
                    if reading.range == float("inf"):
                        continue

                    z = np.array(
                        [
                            [reading.range],
                            [reading.bearing],
                        ]
                    )
                    lm_id = int(col.split("_")[-1])
                    y = robot.sensors[1].y(z, x, lm_id)
                    H_eval = robot.sensors[1].H_eval(x, lm_id)
                    R_eval = robot.sensors[1].R(z)
                    x, P = kf.update(H=H_eval, R=R_eval, z=None, y=y)

                kalman_filter_history = pd.concat(
                    [kalman_filter_history, pd.DataFrame([{"x": x, "P": P}])],
                    ignore_index=True,
                )

            if round(float(next_cmd[0]), 3) <= env.DT * step and not terminal:
                if LINEAR:
                    current_x_vel = float(next_cmd[1])
                    current_y_vel = float(next_cmd[2])
                    current_ang_vel = float(next_cmd[3])
                else:
                    current_lin_vel = float(next_cmd[1])
                    current_ang_vel = float(next_cmd[2])
                try:
                    next_cmd = next(vel_cmds)
                except StopIteration:
                    terminal = True

            if LINEAR:
                robot.robot_step_translational(
                    current_x_vel, current_y_vel, current_ang_vel
                )
            else:
                robot.robot_step_differential(current_lin_vel, current_ang_vel)

    pickle.dump(ground_truth_history, open(output_ground_truth_filepath, "wb"))
    pickle.dump(sensor_data_history, open(output_sensor_data_filepath, "wb"))
    pickle.dump(env.get_environment_info(), open(output_env_data_filepath, "wb"))
    pickle.dump(kalman_filter_history, open(output_kalman_filter_filepath, "wb"))
    print("Done Running Simulation...")

    viz = Visualizer(Path(os.path.join(script_dir, "../output")), linear=LINEAR)
    viz.draw_all()

    ANIMATE = False
    if ANIMATE:
        viz.animate_trajectories()
