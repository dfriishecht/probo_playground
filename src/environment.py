"""
A simulation environment for a mobile robot operating in two dimensions.

The Environment class models the world that the robots navigate in. The world is continuous and two-dimensional. The world possesses an outer border, internal obstacles, and identifiable landmarks. The world also manages the passage of time and the motion of robotic agents within the world over time.

Critically, the environment tracks the robot's state. In this case, the robot's state is a vector that includes three state variables: x position, y position, and heading.
"""
import math
import csv
import pandas as pd
from utils import Position, Pose, Bounds, Landmark, BearingRange


class Environment:
    """
    A class that models the world simulation environment and the robot's state.

    Attributes:
        dimensions: the horizontal and vertical size of the world
        dt: the length of each timestep, in seconds
        obstacles: a list of obstacles
        landmarks: a list of landmarks
        robot_pose: the position and heading of the robot in the world
    """

    def __init__(
        self,
        dimensions: Bounds,
        dt: float,
        obstacles: list[Bounds],
        landmarks: list[Landmark],
        robot_starting_pose: Pose,
    ):
        """
        Initialize an instance of the Environment class.

        Args:
            dimensions: the horizontal and vertical size of the world
            dt: the length of each timestep, in seconds
            obstacles: a list of obstacles
            landmarks: a list of landmarks
            robot_starting_pose: the initial position and heading of the robot
        """
        self.DIMENSIONS = dimensions
        self.DT = dt
        self.time = 0
        self.OBSTACLES = obstacles
        self.LANDMARKS = landmarks
        self.robot_pose = robot_starting_pose

    def robot_step(self, dx: float, dy: float, dtheta: float):
        """
        Update the robot's position and heading in the world. The robot should not be able to pass through obstacles or outside of the world bounds.

        Args:
            dx: change in x position
            dy: change in y position
            dtheta: change in heading

        Returns:
            Nothing, but update the robot_pose property at the end
        """
        new_pos = self.is_valid_motion(dx, dy)

        new_theta = self.robot_pose.theta + dtheta
        new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi

        self.time += round(self.DT, 3)
        self.robot_pose = Pose(new_pos, new_theta)

    def is_valid_motion(self, dx: float, dy: float):
        """
        Given attempted x and y motion by the robot, determine what motion is physically possible (i.e. doesn't go through any obstacles or barriers). Return the actual motion that will be executed.

        Args:
            dx: attempted change in x position
            dy: attempted change in y position

        Returns:
            dx: change in x position that should be executed
            dy: change in y position that should be executed
        """
        if self.is_valid_position(Position(self.robot_pose.pos.x + dx, self.robot_pose.pos.y)):
            x_new = self.robot_pose.pos.x + dx
        else:
            x_new = self.robot_pose.pos.x

        if self.is_valid_position(Position(self.robot_pose.pos.x, self.robot_pose.pos.y + dy)):
            y_new = self.robot_pose.pos.y + dy
        else:
            y_new = self.robot_pose.pos.y

        return Position(x_new, y_new)

    def is_valid_position(self, position: Position):
        """
        Check if a given robot position is valid; i.e. not out-of-bounds or within an obstacle. Return a boolean representing whether or not this condition is true.

        Args:
            position: the robot position

        Returns:
            true if the position is valid and false otherwise
        """
        if self.DIMENSIONS.within_bounds(position):
            result = True
            for obstacle in self.OBSTACLES:
                result = result and not obstacle.within_bounds(position)
            return result
        return False

    def get_robot_pose(self):
        """
        Return the true robot pose.
        """
        return self.robot_pose

    def get_proximity_to_landmarks(self) -> dict:
        """
        Return a list of the robot's true range and bearing to all landmarks.
        """
        measurements = {}
        for landmark in self.LANDMARKS:
            x_diff = landmark.pos.x - self.robot_pose.pos.x
            y_diff = landmark.pos.y - self.robot_pose.pos.y
            distance = math.sqrt(x_diff**2+y_diff**2)
            bearing = math.atan2(y_diff, x_diff) - self.robot_pose.theta
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            measurements[landmark.id] = BearingRange(landmark_id=landmark.id,
                                                     bearing=bearing,range=distance)
        return measurements

    def take_state_snapshot(self):
        """
        Return true state information about this timestep, including time, robot position, and the robot's bearing/range to landmarks, in a table format.
        """
        instrinsic_df = pd.DataFrame(
            {
                "Time": [self.time],
                "RobotPose": [self.robot_pose],
            }
        )
        landmark_dist = self.get_proximity_to_landmarks()
        landmark_df = pd.DataFrame()
        for id, landmark in landmark_dist.items():
            landmark_df[id] = [landmark]
        
        return pd.merge(
            instrinsic_df,
            landmark_df,
            left_index=True,
            right_index=True,
        )

    def get_environment_info(self):
        """
        Return static information about the environment, including dimensions, timestep size, locations and dimensions of obstacles, and locations of landmarks.
        """
        info = {
            "Dimensions": self.DIMENSIONS.to_dict(),
            "Timestep Size": self.DT,
            "Obstacles": [obstacle.to_dict() for obstacle in self.OBSTACLES],
            "Landmarks": [landmark.to_dict() for landmark in self.LANDMARKS],
        }
        return info
