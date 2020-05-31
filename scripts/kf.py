# -*- coding: utf-8 -*-

import sys
print(sys.version)

CURR_DIR = '/content/drive/My Drive/google_colab_work/detailed_probabilistic_robotics/'
sys.path.append(CURR_DIR)

from matplotlib.patches import Ellipse
from mcl import *

def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov)
    first_eig_val, second_eig_val = eig_vals[0], eig_vals[1]
    first_eig_vec, second_eig_vec = eig_vec[:, 0], eig_vec[:, 1]

    w = 2*n*math.sqrt(first_eig_val)
    h = 2*n*math.sqrt(second_eig_val)
    ang = math.degrees(math.atan2(first_eig_vec[1], first_eig_vec[0]))

    return Ellipse(p, width=w, height=h, angle=ang, fill=False, color='blue', alpha=0.5)


def matM(nu, omega, time_interval, stds):
    return np.diag([
        stds['nn']**2*abs(nu)/time_interval + stds['no']**2*abs(omega)/time_interval,
        stds['on']**2*abs(nu)/time_interval + stds['oo']**2*abs(omega)/time_interval,
    ])


def matA(nu, omega, time_interval, theta):
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega*time_interval), math.cos(theta + omega*time_interval)
    return np.array([
        [(stw - st)/omega,  -nu/(omega**2)*(stw - st) + nu/omega*time_interval*ctw],
        [(-ctw + ct)/omega, -nu/(omega**2)*(-ctw + ct) + nu/omega*time_interval*stw],
        [0,                   time_interval]
    ])


def matF(nu, omega, time_interval, theta):
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = nu/omega*(math.cos(theta + omega*time_interval) - math.cos(theta))
    F[1, 2] = nu/omega*(math.sin(theta + omega*time_interval) - math.sin(theta))
    return F


def matH(pose, landmark_pos):
    mu_x, mu_y, mu_theta = pose
    m_x, m_y = landmark_pos
    l = (mu_x - m_x)**2 + (mu_y - m_y)**2
    return np.array([
        [(mu_x - m_x)/math.sqrt(l), (mu_y - m_y)/math.sqrt(l), 0.0],
        [(m_y - mu_y)/l, (mu_x - m_x)/l, -1.0]
    ])


def matQ(distance_std, direction_std):
    return np.diag([distance_std**2, direction_std**2])


class KalmanFilter:
    def __init__(
        self, map, init_pose,
        motion_noise_stds={'nn':0.19, 'no':0.001, 'on':0.13, 'oo':0.20},
        distance_std_rate=0.14, direction_std=0.05
    ):
        self.belief = scipy.stats.multivariate_normal(mean=init_pose, cov=np.diag([1.0e-10,1.0e-10,1.0e-10]))
        self.pose = self.belief.mean
        self.motion_noise_stds = motion_noise_stds
        self.map = map
        self.distance_std_rate = distance_std_rate
        self.direction_std = direction_std

    def motion_update(self, nu, omega, time_interval):
        if abs(omega) < 1.0e-5:
            omega = 1.0e-5
        theta = self.belief.mean[2]
        M = matM(nu, omega, time_interval, self.motion_noise_stds)
        A = matA(nu, omega, time_interval, theta)
        F = matF(nu, omega, time_interval, theta)
        self.belief.cov = F.dot(self.belief.cov).dot(F.T) + A.dot(M).dot(A.T)
        self.belief.mean = IdealRobot.state_transition(self.belief.mean, nu, omega, time_interval)
        self.pose = self.belief.mean

    def observation_update(self, obs):
        for z, obs_id in obs:
            obs_pos = self.map.landmarks[obs_id].pos
            H = matH(self.belief.mean, obs_pos)
            estimated_z = IdealCamera.observation_function(self.belief.mean, obs_pos)
            Q = matQ(estimated_z[0]*self.distance_std_rate, self.direction_std)
            K = self.belief.cov.dot(H.T).dot(np.linalg.inv(Q + H.dot(self.belief.cov).dot(H.T)))
            self.belief.cov = (np.eye(3) - K.dot(H)).dot(self.belief.cov)
            self.belief.mean += K.dot(z - estimated_z)
            self.pose = self.belief.mean

    def draw(self, ax, elems):
        ellipse = sigma_ellipse(p=self.belief.mean[0:2], cov=self.belief.cov[0:2,0:2], n=3)
        elems.append(ax.add_patch(ellipse))

        x, y, theta = self.belief.mean
        theta_sigma = math.sqrt(self.belief.cov[2,2])
        xs = [x+math.cos(theta - 3*theta_sigma), x, x+math.cos(theta + 3*theta_sigma)]
        ys = [y+math.sin(theta - 3*theta_sigma), y, y+math.sin(theta + 3*theta_sigma)]
        elems += ax.plot(xs, ys, color='blue', alpha=0.5)

def trial():
    time_span = 30
    time_interval = 0.1
    world = World(time_span, time_interval, debug=False)

    m = Map()
    for landmark_pos in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*landmark_pos))
    world.append(m)

    init_pose = np.array([0.0, 0.0, math.radians(0.0)])

    nu = 0.2
    omega = math.radians(10.0)
    kf_estimator = KalmanFilter(m, init_pose)
    agent = EstimationAgent(nu, omega, time_interval, estimator=kf_estimator)
    robot = Robot(init_pose, agent=agent, sensor=Camera(m), color='red')
    world.append(robot)

    nu = 0.1
    omega = math.radians(0.0)
    kf_estimator = KalmanFilter(m, init_pose)
    agent = EstimationAgent(nu, omega, time_interval, estimator=kf_estimator)
    robot = Robot(init_pose, agent=agent, sensor=Camera(m), color='red')
    world.append(robot)

    nu = 0.1
    omega = math.radians(-3.0)
    kf_estimator = KalmanFilter(m, init_pose)
    agent = EstimationAgent(nu, omega, time_interval, estimator=kf_estimator)
    robot = Robot(init_pose, agent=agent, sensor=Camera(m), color='red')
    world.append(robot)

    world.draw()
    return world


'''
if __name__ == '__main__':
    world = trial()
    world.ani
'''