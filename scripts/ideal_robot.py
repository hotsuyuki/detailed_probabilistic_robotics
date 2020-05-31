# -*- coding: utf-8 -*-

# %matplotlib inline
import matplotlib
matplotlib.use('nbagg')
import matplotlib.animation as anm
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import math
import scipy.stats
import seaborn as sns


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.time_span = time_span
        self.time_interval = time_interval
        self.debug = debug

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(4,4))
        ax =fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)

        elems = []
        if self.debug:
            for i in range(500):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(
                fig, self.one_step, fargs=(elems, ax),
                frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000),
                repeat=False
            )
            rc('animation', html='jshtml')
            # plt.show()

    def one_step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        time_str = 't = %.2f [s]' % (i * self.time_interval)
        elems.append(ax.text(-5, 5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, 'one_step'):
                obj.one_step(self.time_interval)


class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color='black'):
        self.r = 0.2
        self.pose = pose
        self.poses = [pose]
        self.agent = agent
        self.sensor = sensor
        self.color = color

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x,xn], [y,yn], color=self.color) ### Draw the robot nose by plotting a line
        circle = patches.Circle(xy=(x,y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(circle))

        self.poses.append(self.pose) ### Draw the robot trajectory
        elems += ax.plot([pose[0] for pose in self.poses], [pose[1] for pose in self.poses], color='black', linewidth=0.5)

        if self.sensor and 1 < len(self.poses):
            self.sensor.draw(ax, elems, self.poses[-2], self.color)
        if self.agent and hasattr(self.agent, 'draw'):
            self.agent.draw(ax, elems)

    ### x_t = f(x_t-1, u_t)
    @classmethod
    def state_transition(cls, prev_pose, nu, omega, dt):
        prev_theta = prev_pose[2]
        if math.fabs(omega) < 1.0e-10:
            delta_pose = np.array([
                nu*math.cos(prev_theta)*dt,
                nu*math.sin(prev_theta)*dt,
                omega*dt
            ])
        else:
            delta_pose = np.array([
                nu/omega*(math.sin(prev_theta + omega*dt) - math.sin(prev_theta)),
                nu/omega*(-math.cos(prev_theta + omega*dt) + math.cos(prev_theta)),
                omega * dt
            ])

        curr_pose = prev_pose + delta_pose
        return curr_pose

    def one_step(self, time_interval):
        if not self.agent:
            return

        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(self.pose, nu, omega, time_interval)


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, obs=None):
        return self.nu, self.omega


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y])
        self.id =None

    def draw(self, ax, elems):
        star = ax.scatter(self.pos[0], self.pos[1], s=100, marker='*', label='landmarks', color='orange')
        elems.append(star)
        elems.append(ax.text(self.pos[0], self.pos[1], 'id:'+str(self.id), fontsize=10))


class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for landmark in self.landmarks:
            landmark.draw(ax, elems)


class IdealCamera:
    def __init__(self, map, distance_range=(0.5, 6.0), direction_range=(math.radians(-60.0), math.radians(60.0))):
        self.map = map
        self.lastdata = []
        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):
        if polarpos is None:
            return False

        r = polarpos[0]
        theta = polarpos[1]
        visible = (self.distance_range[0] <= r <= self.distance_range[1]) \
            and (self.direction_range[0] <= theta <= self.direction_range[1])
        return visible

    def data(self, cam_pose):
        observed = []
        for landmark in self.map.landmarks:
            z = self.observation_function(cam_pose, landmark.pos)
            if self.visible(z):
                obs = (z, landmark.id)
                observed.append(obs)

        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        dist = obj_pos - cam_pose[0:2]
        phi = math.atan2(dist[1], dist[0]) - cam_pose[2]
        while math.pi <= phi:
            phi -= 2*math.pi
        while phi < -math.pi:
            phi += 2*math.pi
        return np.array([np.hypot(*dist), phi])

    def draw(self, ax, elems, cam_pose, cam_color='pink'):
        for obs in self.lastdata:
            x, y, theta = cam_pose
            z = obs[0]
            distance, direction = z[0], z[1]
            xl = x + distance*math.cos(theta + direction)
            yl = y + distance*math.sin(theta + direction)
            elems += ax.plot([x,xl], [y,yl], color=cam_color)

            
if __name__ == '__main__':
    world = World(time_span=10, time_interval=0.1, debug=False)

    m = Map()
    m.append_landmark(Landmark(x=2, y=-2))
    m.append_landmark(Landmark(x=-1, y=-3))
    m.append_landmark(Landmark(x=3, y=3))
    world.append(m)

    cam_sensor1 = IdealCamera(m)
    robot1 = IdealRobot(pose1, straight_agent, cam_sensor1)
    cam_sensor2 = IdealCamera(m)
    robot2 = IdealRobot(pose2, circling_agent, cam_sensor2, color='red')
    world.append(robot1)
    world.append(robot2)

    world.draw()


# world.ani
