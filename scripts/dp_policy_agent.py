# -*- coding: utf-8 -*-

import sys
print(sys.version)

CURR_DIR = '/content/drive/My Drive/google_colab_work/detailed_probabilistic_robotics/'
sys.path.append(CURR_DIR)

from puddle_world import *
import itertools
import collections


class DpPolicyAgent(PuddleIgnoreAgent):
    def __init__(
        self, time_interval, estimator, goal, policy_file_path,
        widths=np.array([0.2, 0.2, math.radians(10)]), puddle_coef=100, lower_left=np.array([-4,-4]), upper_right=np.array([4,4])
    ):
        super().__init__(time_interval, estimator, goal, puddle_coef)
        self.pose_min = np.array([*lower_left, math.radians(0)]) # [m], [m], [rad]
        self.pose_max = np.array([*upper_right, math.radians(360)]) # [m], [m], [rad]
        self.widths = widths
        self.index_nums = ((self.pose_max - self.pose_min)/self.widths).astype(int)
        self.policy_data = self.init_policy_data(self.index_nums, policy_file_path)

    def init_policy_data(self, index_nums, policy_file_path):
        policy_data = np.zeros([*self.index_nums, 2])
        for line in open(policy_file_path, 'r'):
            data = line.split()
            policy_data[int(data[0]), int(data[1]), int(data[2])] = [float(data[3]), float(data[4])]
        return policy_data

    def pose2index(self, pose, pose_min, index_nums, widths):
        index = np.floor((pose - pose_min)/widths).astype(int)
        index[2] = (index[2] + index_nums[2]*1000)%index_nums[2]
        for i in [0, 1]:
            if index[i] < 0:
                index[i] = 0
            elif index_nums[i] <= index[i]:
                index[i] = index_nums[i] - 1
        return tuple(index)

    def policy(self, pose, goal=None):
        return self.policy_data[self.pose2index(pose, self.pose_min, self.index_nums, self.widths)]

def trial():
    time_span = 30
    time_interval = 0.1
    world = PuddleWorld(time_span, time_interval, debug=False)

    m = Map()
    for landmark in [(-4,2), (2,-3), (4,4), (-4,-4)]:
        m.append_landmark(Landmark(*landmark))
    world.append(m)

    goal = Goal(-3, -3)
    world.append(goal)

    world.append(Puddle(lower_left=(-2, 0), upper_right=(0, 2), depth=0.1))
    world.append(Puddle(lower_left=(-0.5, -2), upper_right=(2.5, 1), depth=0.1))

    policy_file_path = CURR_DIR + 'sensor_data/mdp_policy.txt'

    for pose in [[-3, 3, 0], [0.5, 1.5, 0], [3, 3, 0], [2, -1, 0]]:
        init_pose = np.array(pose)
        kf_estimator = KalmanFilter(m, init_pose)
        dp_policy_agent = DpPolicyAgent(time_interval, kf_estimator, goal, policy_file_path)

        robot = Robot(
            init_pose, agent=dp_policy_agent,
            sensor=Camera(m, distance_bias_rate_std=0, direction_bias_std=0),
            color='red', bias_rate_stds=(0,0)
        )
        world.append(robot)

    world.draw()
    return world

    
'''
if __name__ == '__main__':
    world = trial()
    world.ani
'''
