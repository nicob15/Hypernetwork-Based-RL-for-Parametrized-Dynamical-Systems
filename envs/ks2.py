import gymnasium as gym
from gymnasium import spaces
import os
import numpy as np
try:
    from pylab import *
    mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
except ImportError:
    pass
import pandas as pd

class KuramotoSivashinskyEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, Nx=64, Lx=22, dt=0.1, T=200, frameskip=1, max_rl_steps=100, parametric=False, random_target=False, action_scale=1.0,
                 nr_actuators=16, nr_sensors=1, oversampling=15, alpha=0.0, mu=0.0, mu_target=0.0, sigma=0.4, offset=8,
                 control_start=100, seed=1, eval=False, random_init=True, nu=1.0):
        super(KuramotoSivashinskyEnv, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(nr_actuators,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(nr_sensors,), dtype=np.float32)

        self.Nx = Nx
        self.Lx = Lx
        self.T = T
        self.dt = dt
        self.t = 0
        self.frameskip = frameskip
        self.timestep = 0
        self.parametric = parametric
        self.random_target = random_target
        self.mu = mu
        self.mu_target = mu_target
        self.mu = float(self.mu)
        self.mu_target = float(self.mu_target)
        self.mu_ks = np.array([self.mu, self.mu_target], dtype=np.float32).reshape(1, -1)
        self.action_scale = action_scale
        self.max_steps = max_rl_steps
        self.oversampling = oversampling
        self.nr_actuators = nr_actuators
        self.nr_sensors = nr_sensors
        self.param_dim = 2
        self.nu = nu


        self.x = np.linspace(start=0, stop=Lx, num=Nx)
        self.dx = Lx / Nx
        self.Nt = int(T / dt)
        self.t = np.linspace(start=0, stop=T, num=self.Nt)

        self.u0 = np.array([0.5 if 4 <= i <= Nx-21 else 0.0 for i in range(0, Nx)])

        self.episode_num = 0

        self.da = int(Nx / nr_actuators)
        self.ds = int(Nx / nr_sensors)
        self.offset = offset
        self.actuator_loc = []
        self.phi = np.zeros((nr_actuators, Nx), dtype=np.float32)
        self.I = np.zeros((Nx,), dtype=np.float32)
        for i in range(nr_actuators):
            self.actuator_loc.append(self.offset + i * self.da)
        self.distr_act = np.zeros((self.Nt, Nx), dtype=np.float32)
        self.control_start = int(control_start / dt)
        self.I[self.offset::self.da] = 1.0
        if self.da == self.ds:
            self.J = np.zeros((Nx,), dtype=np.float32)
            self.J[self.offset::self.da] = 1.0
            self.sensor_loc = np.copy(np.array(self.actuator_loc))
        if nr_sensors == Nx:
            self.sensor_loc = np.arange(0, Nx)

        self.u_sol = []
        self.count_eval = 0
        self.alpha = alpha
        self.sigma = sigma

        self.eval = eval

        self.seed = seed

        self.random_init = random_init

    def gaussian_kernel(self, action, phi, x_a, Nx):
        for i in range(action.shape[0]):
            for x in range(Nx):
                phi[i][x] += action[i] * np.exp(-(1/2)*np.square((x - x_a[i])/self.sigma))
        phi_sum = phi.sum(axis=0)
        return phi_sum

    def ksintegrate_step(self, u, Lx, dt, p, mu):
        kx = np.concatenate((np.arange(0, self.Nx / 2), np.array([0]), np.arange(-self.Nx / 2 + 1, 0)))
        alpha = 2 * np.pi * kx / Lx
        D = 1j * alpha
        L = pow(alpha, 2) - self.nu * pow(alpha, 4)
        G = -0.5 * D

        dt_oversample = dt / self.oversampling
        dt2 = dt_oversample / 2
        dt32 = 3 * dt_oversample / 2
        B = np.ones(self.Nx) + dt2 * L
        A_inv = 1.0 / (np.ones(self.Nx) - dt2 * L)

        u = (1 + 0j) * u
        Nn = G * np.fft.fft(u * u)
        u = np.fft.fft(u)

        for n in range(self.oversampling):
            Nn1 = np.copy(Nn)
            Nn = u
            Nn = np.fft.ifft(Nn)
            Nn = Nn * Nn
            Nn = np.fft.fft(Nn)
            Nn = G * Nn
            u = A_inv * (B * u + dt32 * Nn - dt2 * Nn1 + dt_oversample * np.fft.fft(p) + dt_oversample * np.fft.fft(mu * np.cos((2*np.pi * self.x/(Lx/2)))))
        u = np.real(np.fft.ifft(u))
        return u

    def step(self, a_t):
        #self.u_sol = []
        a_t = self.scale_actions(a_t)
        u = self.u_sol[-1]
        for j in range(self.frameskip):
            self.phi = np.zeros((self.nr_actuators, self.Nx), dtype=np.float32)
            phi = self.gaussian_kernel(a_t, self.phi, self.actuator_loc, self.Nx)
            p = np.copy(phi)
            u = self.ksintegrate_step(u, self.Lx, self.dt, p, self.mu)
            self.u_sol.append(u)
        if self.frameskip > 1:
            d = self.distr_act[self.control_start + self.timestep + self.timestep*self.frameskip: self.control_start + self.timestep + (self.timestep+1)*self.frameskip].shape[0]
            if d == self.frameskip:
                self.distr_act[self.control_start + self.timestep + self.timestep*self.frameskip:self.control_start + self.timestep + (self.timestep+1)*self.frameskip] = np.copy(np.repeat(phi.reshape(1, -1), self.frameskip, axis=0))
            else:
                if d > 0:
                    self.distr_act[self.control_start + self.timestep + self.timestep * self.frameskip: self.control_start + self.timestep + (self.timestep + 1) * self.frameskip] = np.copy(np.repeat(phi.reshape(1, -1), d, axis=0))
                else:
                    pass
        else:
            self.distr_act[self.control_start + self.timestep, :] = np.copy(phi)

        self.history_phi.append(np.copy(self.distr_act))
        self.history_a.append(np.copy(a_t))

        x_t, r_t, d_t, c_t = self.get_step_return(a_t, np.array(self.u_sol).astype('float32'))
        self.history_obs.append(x_t.copy())

        if r_t >= 0:
            r_t = 0.

        self.timestep += 1

        self.history_r.append(r_t)

        return np.concatenate([x_t, self.mu_ks.reshape(1,-1)], axis=1), r_t, d_t, {'state_cost': c_t[0], 'action_cost': c_t[1], 'costs': c_t}

    def get_step_return(self, a_t, u_sol):
        obs_list = []
        if self.nr_sensors == self.Nx:
            obs_list = u_sol[self.control_start + self.timestep*self.frameskip + self.frameskip, :]
        else:
            for i in range(self.nr_sensors):
                o = u_sol[self.control_start + self.timestep*self.frameskip + self.frameskip, self.sensor_loc[i]]
                obs_list.append(o)
            obs_list = np.array(obs_list)
        if self.timestep < int(int(self.max_steps)) - 1:
            d_t = 0
        else:
            d_t = 1
        obs = obs_list.reshape(1, -1)
        u_now = u_sol[self.control_start + self.timestep*self.frameskip]
        c1 = -np.square(u_now - self.target_solution).mean()
        c2 = -np.square(self.action_scale*a_t).mean()
        rew = (c1 + self.alpha * c2).astype('float32')
        return obs, rew, d_t, [-c1, -c2]

    def get_obs(self, u_sol):
        obs_list = []
        if self.nr_sensors == self.Nx:
            obs_list = u_sol[self.control_start + self.timestep, :]
        else:
            for i in range(self.nr_sensors):
                o = u_sol[self.control_start + self.timestep, self.sensor_loc[i]]
                obs_list.append(o)
            obs_list = np.array(obs_list)
        return obs_list
    def scale_actions(self, action):
        return self.action_scale * action

    def generate_random_init(self):
        number_sin = 8
        a_i = np.random.uniform(-1, 1, number_sin)
        a_i /= np.linalg.norm(a_i)
        y0 = np.zeros(self.Nx)
        x_axis = np.arange(self.dx, self.Lx + self.dx, self.dx)
        for i in range(1, number_sin + 1):
            y0 += a_i[i - 1] * np.sin(i * x_axis / (2 * np.pi))
        y0 = y0 * 30 / np.linalg.norm(y0)
        return y0

    def reset(self, i=0):
        assert self.mu_ks.shape == (1, 2)

        self.episode_num += 1

        if self.parametric:
            if self.eval:
                self.mu = 0.104  #np.random.uniform(low=-0.25, high=0.25)
            else:
                self.mu = np.random.choice(
                    [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, -0.025, -0.05, -0.075, -0.1, -0.125, -0.15, -0.175, -0.2, -0.225], 1)[0]
        else:
            pass
        
        self.mu = np.round(self.mu, decimals=3)
        self.mu_target = 0.0
        if self.random_target:
            self.mu_target = np.random.uniform(low=-3.0, high=3.0)
        
        self.mu_target = np.round(float(self.mu_target), 1)

        self.nu = np.round(np.random.uniform(low=1.0, high=2.0), decimals=3)

        self.target_solution = self.mu_target * np.ones(self.Nx)
        self.mu_ks = np.array([self.mu, self.mu_target], dtype=np.float32).reshape(1, -1)

        

        self.history_phi = []
        self.history_a = []
        self.history_r = []
        self.history_sol = []
        self.u_sol = []
        self.history_obs = []

        self.timestep = 0

        # =========================
        # Stato iniziale FISSO
        # =========================
        u = np.array([
            0.071988, 0.097773, 0.073406, 0.08321 , 0.515983, 0.547977, 0.591293, 0.589327,
            0.512011, 0.538355, 0.505644, 0.520204, 0.589091, 0.506163, 0.521614, 0.539198,
            0.515872, 0.590325, 0.510815, 0.500121, 0.512881, 0.500232, 0.592766, 0.534985,
            0.502021, 0.527994, 0.525412, 0.545301, 0.593558, 0.592779, 0.553895, 0.52887,
            0.584011, 0.57372 , 0.508316, 0.59875 , 0.542189, 0.504772, 0.58545 , 0.521634,
            0.58115 , 0.574798, 0.514467, 0.581121, 0.086448, 0.028765, 0.014   , 0.002818,
            0.004026, 0.019971, 0.027136, 0.090142, 0.057282, 0.019871, 0.005811, 0.054028,
            0.047167, 0.031189, 0.078679, 0.011111, 0.013395, 0.09517 , 0.065392, 0.074113
        ], dtype=np.float32)
        for i in range(self.control_start + 1):
            p = 0.0 * np.ones(self.Nx)
            u = self.ksintegrate_step(u, self.Lx, self.dt, p, self.mu)
            self.u_sol.append(u)

        x0 = self.get_obs(np.array(self.u_sol).astype('float32'))

        self.I = np.zeros((self.Nx, ), dtype=np.float32)
        self.phi = np.zeros((self.nr_actuators, self.Nx), dtype=np.float32)
        self.distr_act = np.zeros((self.Nt, self.Nx), dtype=np.float32)

        return np.concatenate([x0.reshape(1, -1), self.mu_ks], axis=1)

    def render(self, name='testing', idx=0, best_policy=False, agent_type='td3', i=0, cost=(0.0, 0.0)):
        try:
            plt
        except NameError:
            return

        save_dir = "figures/kuramotosivashinsky/" + agent_type + "/testing_seed_" + str(self.seed) + "miss"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # =====================================================
        # FIGURA 1: Stato KS u(x,t)  (stessa dimensione/titoli)
        # =====================================================
        fig, ax1 = plt.subplots(1, figsize=(10, 8))
        xx, tt = np.meshgrid(self.x, self.t[:len(self.u_sol)])
        cs = ax1.contourf(tt, xx, np.asarray(self.u_sol), cmap=cm.bwr, levels=750)
        fig.colorbar(cs)
        ax1.set_ylabel("x")
        ax1.set_xlabel("t")

        if best_policy:
            plt.savefig(save_dir + '/' + name + 'state_best' + str(idx) + '_' + str(i) + ".png")
        else:
            plt.savefig(save_dir + '/' + name + 'state' + str(idx) + '_' + str(i) + ".png")
        plt.close()

        # =====================================================
        # FIGURA 2: Controllo distribuito p(x,t) (stessa dimensione/titoli)
        # =====================================================
        fig, ax2 = plt.subplots(1, figsize=(10, 8))
        xx, tt = np.meshgrid(self.x, self.t)
        cs = ax2.contourf(tt, xx, np.asarray(self.distr_act), cmap=cm.bwr, levels=750)  # cm.jet
        fig.colorbar(cs)
        ax2.set_ylabel("x")
        ax2.set_xlabel("t")

        if best_policy:
            plt.savefig(save_dir + '/' + name + 'control_best' + str(idx) + '_' + str(i) + ".png")
        else:
            plt.savefig(save_dir + '/' + name + 'control' + str(idx) + '_' + str(i) + ".png")
        plt.close()

        data = np.array([self.mu, cost[0], cost[1]])
        df1 = pd.DataFrame(np.array(self.u_sol))
        df2 = pd.DataFrame(np.array(self.history_a)[:2999, :])
        df3 = pd.DataFrame(data)

        if best_policy:
            plt.savefig(save_dir + '/' + name + 'opt_policy_best' + str(idx) + '_' + str(i) + ".png")
            if self.episode_num > 50:
                df1.to_csv(save_dir + '/' + name + 'solution_best' + str(idx) + '_' + str(i) + ".csv", index=False)
                df2.to_csv(save_dir + '/' + name + 'actions_best' + str(idx) + '_' + str(i) + ".csv", index=False)
                df3.to_csv(save_dir + '/' + name + 'parameters_best' + str(idx) + '_' + str(i) + ".csv", index=False)
        else:
            if self.episode_num > 50:
                plt.savefig(save_dir + '/' + name + 'opt_policy' + str(idx) + '_' + str(i) + ".png")
                df1.to_csv(save_dir + '/' + name + 'solution' + str(idx) + '_' + str(i) + ".csv", index=False)
                df2.to_csv(save_dir + '/' + name + 'actions' + str(idx) + '_' + str(i) + ".csv", index=False)
                df3.to_csv(save_dir + '/' + name + 'parameters' + str(idx) + '_' + str(i) + ".csv", index=False)
        plt.close()

        fig, (ax1) = plt.subplots(1, figsize=(10, 8))
        xx, tt = np.meshgrid(self.x, self.t[:len(self.u_sol)])
        levels = np.linspace(-3.5, 3.5, 750)
        cs = ax1.contourf(tt, xx, np.asarray(self.u_sol), levels=levels, cmap=cm.bwr)
        a = fig.colorbar(cs, ticks=[-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,  0.0, 0.5,  1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        a.ax.set_ylim(-3.5, 3.5)
        ax1.set_ylabel("x")

        if best_policy:
            plt.savefig(save_dir + '/' + name + 'solution_best' + str(idx) + '_' + str(i) + ".png")
        else:
            plt.savefig(save_dir + '/' + name + 'solution' + str(idx) + '_' + str(i) + ".png")
        plt.close()

        plt.ion()
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(self.nr_actuators)
        ax1.plot(np.array(self.history_a)[:, 7], color='red', label=r'$u_1$')
        ax2.plot(np.array(self.history_a)[:, 6], color='red', alpha=1.0, label=r'$u_2$')
        ax3.plot(np.array(self.history_a)[:, 5], color='red', alpha=1.0, label=r'$u_3$')
        ax4.plot(np.array(self.history_a)[:, 4], color='red', alpha=1.0, label=r'$u_4$')
        ax5.plot(np.array(self.history_a)[:, 3], color='red', alpha=1.0, label=r'$u_5$')
        ax6.plot(np.array(self.history_a)[:, 2], color='red', alpha=1.0, label=r'$u_6$')
        ax7.plot(np.array(self.history_a)[:, 1], color='red', alpha=1.0, label=r'$u_7$')
        ax8.plot(np.array(self.history_a)[:, 0], color='red', alpha=1.0, label=r'$u_8$')
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax3.legend(loc="upper right")
        ax4.legend(loc="upper right")
        ax5.legend(loc="upper right")
        ax6.legend(loc="upper right")
        ax7.legend(loc="upper right")
        ax8.legend(loc="upper right")
        fig.supxlabel('t')

        if best_policy:
            plt.savefig(save_dir + '/' + name + 'actions_best' + str(idx) + '_' + str(i) + ".png")
        else:
            plt.savefig(save_dir + '/' + name + 'actions' + str(idx) + '_' + str(i) + ".png")
        plt.close()

        # =====================================================
        # Plot delle misurazioni dei sensori
        # =====================================================

        if len(self.history_obs) > 0:
            history_obs = np.array(self.history_obs)   # (T, 1, nr_sensors)
            history_obs = history_obs.squeeze(1)       # (T, nr_sensors)

            fig, axes = plt.subplots(self.nr_sensors, figsize=(10, 2*self.nr_sensors), sharex=True)

            for j in range(self.nr_sensors):
                axes[j].plot(history_obs[:, j], label=f'Sensor {j+1}')
                axes[j].legend(loc="upper right")
                axes[j].set_ylabel('u')

            axes[-1].set_xlabel('t (RL steps)')

            if best_policy:
                plt.savefig(save_dir + '/' + name + 'sensors_best' + str(idx) + '_' + str(i) + ".png")
            else:
                plt.savefig(save_dir + '/' + name + 'sensors' + str(idx) + '_' + str(i) + ".png")

            plt.close()