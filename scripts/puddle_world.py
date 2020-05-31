# -*- coding: utf-8 -*-

import sys
print(sys.version)

CURR_DIR = '/content/drive/My Drive/google_colab_work/detailed_probabilistic_robotics/'
sys.path.append(CURR_DIR)

from kf import *


class Goal:
    def __init__(self, x, y, radius=0.3, value=0.0):
        self.pos = np.array([x, y])
        self.radius = radius
        self.value = value

    def is_inside(self, pose):
        dx = pose[0] - self.pos[0]
        dy = pose[1] - self.pos[1]
        return math.hypot(dx, dy) < self.radius

    def draw(self, ax, elems):
        x, y = self.pos
        flag = ax.scatter(x+0.16, y+0.5, s=50, marker='>', label='landmarks', color='red')
        elems.append(flag)
        elems += ax.plot([x,x], [y,y+0.6], color='black')


class Puddle:
    def __init__(self, lower_left, upper_right, depth):
        self.lower_left = lower_left
        self.upper_right = upper_right
        self.depth = depth

    def is_inside(self, pose):
        return all([self.lower_left[i] < pose[i] < self.upper_right[i] for i in [0,1]])

    def draw(self, ax, elems):
        width = self.upper_right[0] - self.lower_left[0]
        height = self.upper_right[1] - self.lower_left[1]
        rect = patches.Rectangle(self.lower_left, width, height, color='blue', alpha=self.depth)
        elems.append(ax.add_patch(rect))


class PuddleWorld(World):
    def __init__(self, time_span, time_interval, debug=False):
        super().__init__(time_span, time_interval, debug)
        self.puddles = []
        self.robots = []
        self.goals = []

    def append(self, obj):
        self.objects.append(obj)
        if isinstance(obj, Puddle):
            self.puddles.append(obj)
        if isinstance(obj, Robot):
            self.robots.append(obj)
        if isinstance(obj, Goal):
            self.goals.append(obj)

    def puddle_depth(self, pose):
        return sum([puddle.is_inside(pose)*puddle.depth for puddle in self.puddles])

    def one_step(self, i, elems, ax):
        super().one_step(i, elems, ax)
        for robot in self.robots:
            robot.agent.puddle_depth = self.puddle_depth(robot.pose)
            for goal in self.goals:
                if goal.is_inside(robot.pose):
                    robot.agent.is_goal = True
                    robot.agent.final_value = goal.value


class PuddleIgnoreAgent(EstimationAgent):
    def __init__(self, time_interval, estimator, goal, puddle_coef=100):
        super().__init__(0.0, 0.0, time_interval, estimator)
        self.puddle_coef = puddle_coef
        self.puddle_depth = 0.0
        self.total_reward = 0.0
        self.is_goal = False
        self.final_value = 0.0
        self.goal = goal

    def reward_per_sec(self):
        return -1.0 - self.puddle_coef*self.puddle_depth

    @classmethod
    def policy(cls, pose, goal):
        x, y, theta = pose
        dx = goal.pos[0] - x
        dy = goal.pos[1] - y
        direction = int(math.degrees(math.atan2(dy,dx) - theta)) # [deg]
        while direction < -180:
            direction += 360
        while 180 <= direction:
            direction -= 360

        if 10 < direction:
            nu = 0.0 # [m/s]
            omega = 2.0 # [rad/s]
        elif direction < -10:
            nu = 0.0 # [m/s]
            omega = -2.0 # [rad/s]
        else:
            nu = 1.0 # [m/s]
            omega = 0.0 # [rad/s]
        return nu, omega

    def decision(self, obs=None):
        if self.is_goal:
            nu = 0.0 # [m/s]
            omega = 0.0 # [rad/s]
            return nu, omega

        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.estimator.observation_update(obs)

        self.total_reward += self.time_interval*self.reward_per_sec()
        self.prev_nu, self.prev_omega = self.policy(self.estimator.pose, self.goal)
        return self.prev_nu, self.prev_omega

    def draw(self, ax, elems):
        super().draw(ax, elems)
        x, y, theta = self.estimator.pose
        elems.append(ax.text(x+1.0, y-0.5, 'reward/sec: '+str(self.reward_per_sec()), fontsize=8))
        elems.append(ax.text(x+1.0, y-1.0, 'eval: {:.1f}'.format(self.total_reward + self.final_value), fontsize=8))


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

    init_pose = np.array([0, 0, math.radians(0.0)])
    kf_estimator = KalmanFilter(m, init_pose)
    straight_agent = PuddleIgnoreAgent(time_interval, kf_estimator, goal)

    robot = Robot(
        init_pose, agent=straight_agent,
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
