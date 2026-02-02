"""
Main file for running the simulator.
"""
import os
import pandas as pd
import csv, pickle
from environment import Environment
from robot import Robot
from utils import Position, Pose, Landmark, Bounds

if __name__ == "__main__":
    # set up the environment
    dimensions = Bounds(
        0,
        10,
        0,
        10
    )

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
    initial_robot_pose = Pose(Position(0,0), 0.6)

    env = Environment(
        dimensions,
        dt,
        obstacles,
        landmarks,
        initial_robot_pose,
    )

    # set up the robot
    robot = Robot(env)

    # set up timekeeping
    total_seconds = 20
    total_timesteps = total_seconds / env.DT
    terminal = False

    # set up logging
    ground_truth_history = pd.DataFrame()
    sensor_data_history = pd.DataFrame()

    # set up input filepath and output filepaths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_commands_filepath = os.path.join(script_dir, "../input/vel_cmd_example.csv")
    output_ground_truth_filepath = os.path.join(script_dir, "../output/ground_truth.pkl")
    output_sensor_data_filepath = os.path.join(script_dir, "../output/sensor_data.pkl")
    output_env_data_filepath = os.path.join(script_dir, "../output/env_data.pkl")

    # open up the instructions, pop the first
    with open(input_commands_filepath, "r") as cmd:
        vel_cmds = csv.reader(cmd)
        next_cmd = next(vel_cmds)
        next_cmd = next(vel_cmds)
        current_lin_vel = float(next_cmd[1])
        current_ang_vel = float(next_cmd[2])
        for step in range(int(total_timesteps) + 1):
            ground_truth_history = pd.concat(
                [
                    ground_truth_history,
                    robot.take_gt_snapshot(),
                ],
                ignore_index=True
            )
            # TODO: take sensor measurements and add it to the history
            sensor_data_history = pd.concat(
                [
                    sensor_data_history,
                    robot.take_sensor_measurements(),
                ],
                ignore_index=True
            )
            # TODO: retrieve the next motor command from the input file
            if round(float(next_cmd[0]), 3) <= env.DT * step and not terminal:
                current_lin_vel = float(next_cmd[1])
                current_ang_vel = float(next_cmd[2])
                try:
                    next_cmd = next(vel_cmds)
                except StopIteration:
                    terminal = True

            # TODO: execute the motor command
            robot.robot_step_differential(current_lin_vel, current_ang_vel)
        
        pickle.dump(
            ground_truth_history, open(output_ground_truth_filepath, "wb")
        )
        pickle.dump(sensor_data_history, open(output_sensor_data_filepath, "wb"))
        pickle.dump(env.get_environment_info(), open(output_env_data_filepath, "wb"))
        print("Done Running Simulation...")
