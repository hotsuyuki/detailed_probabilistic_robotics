# -*- coding: utf-8 -*-

import sys
print(sys.version)

CURR_DIR = '/content/drive/My Drive/google_colab_work/detailed_probabilistic_robotics/'
sys.path.append(CURR_DIR)

import scipy.stats 
from ideal_robot import *


class Robot(IdealRobot):
    def __init__(
        self, pose, agent=None, sensor=None, color='black',
        noise_per_meter=5, noise_std=math.radians(3.0),
        bias_rate_stds=(0.1, 0.1),
        expected_stuck_time=1e+100, expected_escape_time=1e-100,
        expected_kidnap_time=1e+100, kidnap_range_x=(-5.0,5.0), kidnap_range_y=(-5.0,5.0)
    ):
        super().__init__(pose, agent, sensor, color)

        expected_dist_until_noise = 1.0 / (noise_per_meter + sys.float_info.epsilon) # [m/noise]
        self.noise_dist_pdf = scipy.stats.expon(scale=expected_dist_until_noise)
        self.dist_until_noise = self.noise_dist_pdf.rvs() # [m/noise]
        self.theta_noise_pdf = scipy.stats.norm(loc=0.0, scale=noise_std)

        self.bias_rate_nu = scipy.stats.norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = scipy.stats.norm.rvs(loc=1.0, scale=bias_rate_stds[1])

        self.stuck_pdf = scipy.stats.expon(scale=expected_stuck_time)
        self.escape_pdf = scipy.stats.expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs() # [s/stuck]
        self.time_until_escape = self.escape_pdf.rvs() # [s/escape]
        self.is_stuck = False

        self.kidnap_pdf = scipy.stats.expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs() # [s/kidnap]
        self.kidnap_dist = scipy.stats.uniform(
            loc=(kidnap_range_x[0], kidnap_range_y[0], 0.0),
            scale=(kidnap_range_x[1]-kidnap_range_x[0], kidnap_range_y[1]-kidnap_range_y[0], 2*math.pi)
        )

    def noise(self, pose, nu, omega, time_interval):
        self.dist_until_noise -= (abs(nu) + self.r*abs(omega))*time_interval
        if self.dist_until_noise <= 0.0:
            self.dist_until_noise += self.noise_dist_pdf.rvs() # [m/noise]
            theta_noise = self.theta_noise_pdf.rvs() # [rad]
        else:
            theta_noise = 0
        return np.array([0, 0, theta_noise])

    def bias(self, nu, omega):
        return self.bias_rate_nu*nu, self.bias_rate_omega*omega

    def stuck(self, nu, omega, time_interval):
        if not self.is_stuck:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs() # [s/stuck]
                self.is_stuck = True
        else:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs() # [s/escape]
                self.is_stuck = False
        return nu*(not self.is_stuck), omega*(not self.is_stuck)

    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs() # [s/kidnap]
            return np.array(self.kidnap_dist.rvs()) # [m], [m], [rad]
        else:
            return pose

    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None

        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)

        self.pose = self.state_transition(self.pose, nu, omega, time_interval)
        self.pose += self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose, time_interval)


class Camera(IdealCamera):
    def __init__(
        self, map, distance_range=(0.5, 6.0), direction_range=(math.radians(-60.0), math.radians(60.0)),
        distance_noise_rate=0.1, direction_noise=math.radians(2.0),
        distance_bias_rate_std=0.1, direction_bias_std=math.radians(2.0),
        phantom_prob=0.0, phantom_range_x=(-5.0,5.0), phantom_range_y=(-5.0,5.0),
        oversight_prob=0.1, occlusion_prob=0.0
    ):
        super().__init__(map, distance_range, direction_range)

        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

        self.distance_bias_rate = scipy.stats.norm.rvs(loc=0.0, scale=distance_bias_rate_std) # [m]
        self.direction_bias = scipy.stats.norm.rvs(loc=0.0, scale=direction_bias_std) # [rad]

        self.phantom_prob = phantom_prob
        self.phantom_dist = scipy.stats.norm(
            loc=(phantom_range_x[0], phantom_range_y[0]),
            scale=(phantom_range_x[1]-phantom_range_x[0], phantom_range_y[1]-phantom_range_y[0])
        )

        self.oversight_prob = oversight_prob
        self.occlusion_prob = occlusion_prob

    def noise(self, z):
        dist, phi = z[0], z[1]
        dist_noise = scipy.stats.norm.rvs(loc=0.0, scale=self.distance_noise_rate*dist) # [m]
        phi_noise = scipy.stats.norm.rvs(loc=0.0, scale=self.direction_noise) # [rad]
        return np.array([dist_noise, phi_noise])

    def bias(self, z):
        dist, phi = z[0], z[1]
        dist_bias = self.distance_bias_rate*dist # [m]
        phi_bias = self.direction_bias # [rad]
        return np.array([dist_bias, phi_bias])

    def phantom(self, cam_pose, z):
        if scipy.stats.uniform.rvs(loc=0.0, scale=1.0) < self.phantom_prob:
            phantom_pos = np.array(self.phantom_dist.rvs()) # [m], [m]
            return self.observation_function(cam_pose, phantom_pos)
        else:
            return z

    def occlusion(self, z):
        dist, phi = z[0], z[1]
        if scipy.stats.uniform.rvs(loc=0.0, scale=1.0) < self.occlusion_prob:
            occluded_dist_rate = scipy.stats.uniform.rvs(loc=0.0, scale=1.0) 
            occluded_dist = dist + occluded_dist_rate*(self.distance_range[1] - dist)
            return np.array([occluded_dist, phi])
        else:
            return z

    def oversight(self, z):
        if scipy.stats.uniform.rvs(loc=0.0, scale=1.0) < self.oversight_prob:
            return None
        else:
            return z

    def data(self, cam_pose):
        observed = []
        for landmark in self.map.landmarks:
            z = self.observation_function(cam_pose, landmark.pos)
            z = self.phantom(cam_pose, z)
            z = self.occlusion(z)
            z = self.oversight(z)
            if self.visible(z):
                obs = (z+self.bias(z)+self.noise(z), landmark.id)
                observed.append(obs)
                
        self.lastdata = observed
        return observed


if __name__ == '__main__':
    world = World(time_span=20, time_interval=0.1, debug=False)
    
    m = Map()
    m.append_landmark(Landmark(x=-4, y=2))
    m.append_landmark(Landmark(x=2, y=-3))
    m.append_landmark(Landmark(x=3, y=3))
    world.append(m)
    
    pose = np.array([0, 0, math.radians(0.0)])
    circling_agent = Agent(nu=0.2, omega=math.radians(10.0))
    cam_sensor = Camera(m)
    robot = Robot(
        pose, agent=circling_agent, sensor=cam_sensor,
        noise_per_meter=0,
        bias_rate_stds=(0, 0),
        expected_stuck_time=sys.maxsize, expected_escape_time=0,
        expected_kidnap_time=sys.maxsize
    )
    world.append(robot)
    
    # ideal_cam_sensor = IdealCamera(m)
    # ideal_robot = IdealRobot(pose, agent=circling_agent, sensor=ideal_cam_sensor, color='red')
    # world.append(ideal_robot)
    
    world.draw()


#world.ani

