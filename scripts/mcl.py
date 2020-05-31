# -*- coding: utf-8 -*-

import sys
print(sys.version)

CURR_DIR = '/content/drive/My Drive/google_colab_work/detailed_probabilistic_robotics/'
sys.path.append(CURR_DIR)

import random
import copy
from robot import *


class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight

    def motion_update(self, nu, omega, time_interval, motion_noise_rate_pdf):
        deltas = motion_noise_rate_pdf.rvs()
        delta_nn = deltas[0] # [m/sqrt(m)]
        delta_no = deltas[1] # [m/sqrt(rad)]
        delta_on = deltas[2] # [rad/sqrt(m)]
        delta_oo = deltas[3] # [rad/sqrt(rad)]

        noise_nu = delta_nn*math.sqrt(abs(nu)/time_interval) + delta_no*math.sqrt(abs(omega)/time_interval)
        noise_omega = delta_on*math.sqrt(abs(nu)/time_interval) + delta_oo*math.sqrt(abs(omega)/time_interval)
        self.pose = IdealRobot.state_transition(self.pose, nu+noise_nu, omega+noise_omega, time_interval)

    def observation_update(self, obs, map, distance_std_rate, direction_std):
        for z, obs_id in obs:
            pos_on_map = map.landmarks[obs_id].pos
            estimated_z = IdealCamera.observation_function(self.pose, pos_on_map)

            c = np.diag([
                (estimated_z[0]*distance_std_rate)**2,
                direction_std**2,
            ])
            particle_pose_likelihood_dist = scipy.stats.multivariate_normal(mean=estimated_z, cov=c)
            self.weight *= particle_pose_likelihood_dist.pdf(z)


class Mcl:
    def __init__(
        self, map, init_pose, num,
        motion_noise_stds={'nn':0.19, 'no':0.001, 'on':0.13, 'oo':0.20},
        distance_std_rate=0.14, direction_std=0.05
    ):
        self.map = map
        weight = 1.0 / num
        self.particles = [Particle(init_pose, weight) for i in range(num)]
        c = np.diag([
            motion_noise_stds['nn']**2, # [m^2/m]
            motion_noise_stds['no']**2, # [m^2/rad]
            motion_noise_stds['on']**2, # [rad^2/m]
            motion_noise_stds['oo']**2 # [rad^2/rad]
        ])
        self.motion_noise_rate_pdf = scipy.stats.multivariate_normal(cov=c)
        self.distance_std_rate = distance_std_rate
        self.direction_std = direction_std
        self.max_likeli = self.particles[0]
        self.pose = self.max_likeli.pose

    def set_max_likeli(self):
        idx = np.argmax([particle.weight for particle in self.particles])
        self.max_likeli = self.particles[idx]
        self.pose = self.max_likeli.pose

    def motion_update(self, nu, omega, time_interval):
        for particle in self.particles:
            particle.motion_update(nu, omega, time_interval, self.motion_noise_rate_pdf)

    def observation_update(self, obs):
        for particle in self.particles:
            particle.observation_update(obs, self.map, self.distance_std_rate, self.direction_std)
        self.set_max_likeli()
        self.resampling()

    '''
    ### Random sampling: O(N*log(N))
    def resampling(self):
        weights = [particle.weight for particle in self.particles]
        if sum(weights) < 1.0e-100:
            weights = [weight + 1.0e-100 for weight in weights]
        resampled_particles = random.choices(self.particles, weights=weights, k=len(self.particles))
        self.particles = [copy.deepcopy(resampled_particle) for resampled_particle in resampled_particles]
        for particle in self.particles:
            particle.weight = 1.0 / len(self.particles)
    '''

    ### Systematic sampling: O(N)
    def resampling(self):
        cumsum_weights = np.cumsum([particle.weight for particle in self.particles])
        if cumsum_weights[-1] < 1.0e-100:
            cumsum_weights = [weight + 1.0e-100 for weight in cumsum_weights]

        step = cumsum_weights[-1] / len(self.particles)
        r = np.random.uniform(0.0, step)
        curr_idx = 0
        resampled_particles = []
        while(len(resampled_particles) < len(self.particles)):
            if r < cumsum_weights[curr_idx]:
                resampled_particles.append(self.particles[curr_idx])
                r += step
            else:
                curr_idx += 1

        self.particles = [copy.deepcopy(resampled_particle) for resampled_particle in resampled_particles]
        for particle in self.particles:
            particle.weight = 1.0 / len(self.particles)

    def draw(self, ax, elems):
        xs = [particle.pose[0] for particle in self.particles]
        ys = [particle.pose[1] for particle in self.particles]
        vxs = [particle.weight*len(self.particles)*math.cos(particle.pose[2]) for particle in self.particles]
        vys = [particle.weight*len(self.particles)*math.sin(particle.pose[2]) for particle in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, angles='xy', scale_units='xy', scale=1.5, color='blue', alpha=0.5))
        

class EstimationAgent(Agent):
    def __init__(self, nu, omega, time_interval, estimator):
        super().__init__(nu, omega)
        self.prev_nu = 0.0
        self.prev_omega = 0.0
        self.time_interval = time_interval
        self.estimator = estimator

    def decision(self, obs=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu = self.nu
        self.prev_omega = self.omega
        self.estimator.observation_update(obs)
        return self.nu, self.omega

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        x, y, theta = self.estimator.pose
        text = '(x={:.2f}, y={:.2f}, {}[deg])'.format(x, y, int(math.degrees(theta))%360)
        elems.append(ax.text(x, y+0.1, text, fontsize=8))


def trial():
    time_span = 20
    time_interval = 0.1
    world = World(time_span, time_interval, debug=False)
    # world = World(time_span, time_interval, debug=True)

    m = Map()
    for landmark in [(-4,3), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*landmark))
    world.append(m)

    init_pose = np.array([0, 0, math.radians(0.0)])
    num = 100
    mcl_estimator = Mcl(m, init_pose, num)

    nu = 0.2
    omega = math.radians(10.0)
    circling_agent = EstimationAgent(nu, omega, time_interval, estimator=mcl_estimator)

    robot = Robot(
        init_pose, agent=circling_agent, sensor=Camera(m), color='red',
        noise_per_meter=0,
        bias_rate_stds=(0, 0),
        expected_stuck_time=sys.maxsize, expected_escape_time=0,
        expected_kidnap_time=sys.maxsize
    )
    world.append(robot)

    world.draw()
    return world


'''
if __name__ == '__main__':
    world = trial()
    world.ani
'''
