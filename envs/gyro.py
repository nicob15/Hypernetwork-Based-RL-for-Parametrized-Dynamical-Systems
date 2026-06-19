import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import os
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import imageio
    from IPython.display import clear_output as clc
    from IPython.display import display
except ImportError:
    pass

class Gyro(gym.Env):
    def __init__(self, T=80, dt=0.1, parametric_target=False, parametric_gyro=False, navigation_mode='naive',
                 rel_position=True, max_action=1, seed=0, random_init=False, eval=False, memory=1, state_ind=0):

        self.goal = [0.5, 0.5]
        self.start_position = [1.5, 0.5]
        self.start_time = 0.

        self.max_action = max_action
        self.intensity = 0.1
        self.amplitude = 0.25
        self.frequency = 2*np.pi/10
        self.target_radius = 0.005

        self.memory = memory
        self.navigation_dim = {
            "naive": 0,
            "velocity_T": 2,
            "velocity": 2,
            "vorticity": 1,
            "velocity_history": 2*self.memory,
            "vorticity_history": self.memory
        }
        self.navigation_mode = navigation_mode

        self.dt = dt
        self.max_ubtime = T
        self.ubtime = 0

        self.random_init = random_init
        self.parametric_target = parametric_target
        self.parametric_gyro = parametric_gyro
        self.rel_position = rel_position

        if self.rel_position:
            self.param_dim = 0
        else:
            self.param_dim = 2
    
        if self.parametric_gyro:
            self.param_dim += 2

        self.seed = seed
        self.eval = eval
        self.R = 1.0
        self.alpha = 0.2

        if parametric_gyro:
            self.alpha *= 2

        self.start = self.start_position.copy()
        if navigation_mode=="naive" or navigation_mode=="velocity_T":
            self.start = np.concatenate([[self.start_time], self.start])
            self.state_ind = 0
        else:
            self.state_ind = 1

        if "velocity" in self.navigation_mode:
            start_velocity = self.doublegyreVEC(self.start_time, self.start_position[:], [0,0],
                                                self.intensity, self.amplitude, self.frequency)
            self.start = np.concatenate([self.start, start_velocity], axis=0)
        elif "vorticity" in self.navigation_mode:
            start_vorticity = self.local_vorticity(self.start_position[0], self.start_position[1], t=self.start_time, nx=10, ny=10)
            self.start = np.concatenate([self.start, start_vorticity], axis=0)

        if self.memory > 1:
            if "history" in self.navigation_mode:
                padding = np.zeros(2 + self.navigation_dim[self.navigation_mode] - len(self.start))
                self.start = np.concatenate([self.start, padding], axis=0)
            else:
                raise ValueError("In order to keep measure memory use navigation_mode: <<sth_history>>")

        state_dim = self.start.shape[0] + self.param_dim

        self.observation_space = Box(low=np.zeros(shape=state_dim, dtype=np.float32),
                                     high=np.ones(shape=state_dim, dtype=np.float32),
                                     dtype=np.float32)
        self.action_space = Box(low=np.array([-self.max_action, -self.max_action], dtype=np.float32),
                                high=np.array([self.max_action, self.max_action], dtype=np.float32),
                                dtype=np.float32)
        
    def local_vorticity(self, x, y, t, nx=10, ny=10):
        """
        returns the vorticity in a target position at time t, computing gradients in a rectangle 
        centered in (x, y), discretized in a nx • ny grid; nx=ny=10 as standard to preserve efficiency
        """

        Lx = (2.)/200
        Ly = (1.)/100
        dx = Lx / nx
        dy = Ly / ny

        x_rec = np.linspace(x-Lx, x+Lx, 2*nx+1) # make sure the number is odd so that the vorticity
        y_rec = np.linspace(y-Ly, y+Ly, 2*ny+1) # is computed in (x,y)

        u, v = self.double_gyre_flow(self.amplitude, self.frequency, x_rec, y_rec, t=np.array([t]))
        du_dy = np.gradient(u[0], dy, axis=1)
        dv_dx = np.gradient(v[0], dx, axis=0)
        vorticity = dv_dx - du_dy

        return np.array([vorticity[nx, ny]])

    def doublegyreVEC(self, t, yin, ctrl, intensity, amplitude, frequency):
        x = yin[0]
        y = yin[1]
        u_x = ctrl[0]
        u_y = ctrl[1]

        a = amplitude * np.sin(frequency * t)
        b = 1 - 2 * a

        f = a * x ** 2 + b * x
        df = 2 * a * x + b

        u = -np.pi * intensity * np.sin(np.pi * f) * np.cos(np.pi * y) + self.alpha * u_x
        v = np.pi * intensity * np.cos(np.pi * f) * np.sin(np.pi * y) * df + self.alpha * u_y

        return np.array([u, v])

    def rk1singlestep(self, fun, dt, t0, y0, u0):
        f1 = fun(t0, y0, u0)

        yout = y0 + (dt) * (f1)
        return yout
    
    def rk4singlestep(self, fun, dt, t0, y0, u0):
        k1 = fun(t0, y0, u0)
        k2 = fun(t0+(dt/2), y0+(k1*dt/2), u0)
        k3 = fun(t0+(dt/2), y0+(k2*dt/2), u0)
        k4 = fun(t0+dt, y0+(k3*dt), u0)

        yout = y0 + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return yout

    def step(self, u):
        t_inst = 10 * self.state[0]
        x_pos = self.state[1]
        y_pos = self.state[2]

        dt = self.dt
        goal = self.goal
        A = self.intensity
        eps = self.amplitude
        om = self.frequency

        terminated = False
        truncated = False

        state_cost = ((x_pos - goal[0]) ** 2 + (y_pos - goal[1]) ** 2)
        action_cost = self.R * ((self.alpha * u[0] ** 2) + (self.alpha * u[1] ** 2))
        costs = state_cost + action_cost

        yout = self.rk1singlestep(lambda t, y, u: self.doublegyreVEC(t, y, u, A, eps, om), dt, t_inst, np.array([x_pos, y_pos]), u)
        state = np.array([((t_inst + dt) * om/(2*np.pi)) % 1, yout[0], yout[1]], dtype=np.float32)

        if "velocity" in self.navigation_mode:
            next_velocity = self.doublegyreVEC(state[0], yout[:], [0,0], self.intensity, self.amplitude, self.frequency)
            state = np.concatenate([state, next_velocity])
            lag = 2
        elif "vorticity" in self.navigation_mode:
            next_vorticity = self.local_vorticity(yout[0], yout[1], t=state[0], nx=10, ny=10)
            state = np.concatenate([state, next_vorticity])
            lag = 1
        if self.memory > 1:
            buffer = self.state[3:]
            state = np.concatenate([state, buffer[:-lag]], axis=0)

        self.state = state.copy()        
        self.ubtime += dt

        if ((x_pos-goal[0])**2 + (y_pos-goal[1])**2) < self.target_radius:
            self.target_reached = True
            terminated = True
            truncated = True
            self.ep_target_reached.append(self.target_reached)

        if self.ubtime > self.max_ubtime:
            terminated = True
            truncated = True
        #     if(self.near_target >= 100):
        #         print(f"Near target for {self.near_target} time steps...")
        #         self.target_reached = True
        #     self.ep_target_reached.append(self.target_reached)
        #     truncated = True

        self.ep_traj.append(self.state)
        self.ep_acts.append(u)

        if self.rel_position:
            state[1:3] = self.goal - state[1:3]
        else:
            state = np.concatenate([state, self.goal], axis=0)

        if self.parametric_gyro:
            state = np.concatenate([state, self.mu], axis=0)

        return state[self.state_ind:], -costs, terminated, truncated, {'state_cost': state_cost, 'action_cost': action_cost, 'costs': costs,
                                                          'target_reached': self.target_reached}

    
    def reset(self, i=0, seed=None, options=None):

        self.near_target = 0
        self.target_reached = False

        if self.random_init == False:
            start = np.concatenate([self.start_time, self.start_position])
        else:
            start = [0, np.random.uniform(low=0.1, high=1.9), np.random.uniform(low=0.1, high=0.9)]
    
        if self.parametric_target:
            if self.eval:
                self.goal = [np.random.uniform(low=0.1, high=1.9), np.random.uniform(low=0.1, high=0.9)]
            else:
                self.goal = [np.random.uniform(low=0.1, high=1.9), np.random.uniform(low=0.1, high=0.9)]

        if self.parametric_gyro:
            if self.eval:
                self.amplitude = np.random.uniform(low=0.0, high=0.5)
                self.frequency = np.random.uniform(low=0.5, high=2*np.pi/3)
                self.mu = [self.amplitude, self.frequency]
            else:
                self.amplitude = np.random.uniform(low=0.0, high=0.5)
                self.frequency = np.random.uniform(low=0.5, high=2*np.pi/3)
                self.mu = [self.amplitude, self.frequency]

        if "velocity" in self.navigation_mode:
            start_velocity = self.doublegyreVEC(0, start[1:], [0,0], self.intensity, self.amplitude, self.frequency)
            start = np.concatenate([start, start_velocity], axis=0)
        elif "vorticity" in self.navigation_mode:
            start_vorticity = self.local_vorticity(start[1], start[2], t=0, nx=10, ny=10)
            start = np.concatenate([start, start_vorticity], axis=0)
        if self.memory > 1:
            padding = np.zeros(3 + self.navigation_dim[self.navigation_mode] - len(start))
            start = np.concatenate([start, padding], axis=0)


        self.state = np.array(start, dtype=np.float32)
        self.ubtime = 0

        self.ep_traj = []
        self.ep_acts = []
        self.ep_target_reached = []
        self.ep_traj.append(self.state)

        if self.rel_position:
            state = np.array(start, dtype=np.float32)
            state[1:3] = self.goal - state[1:3]
        else:
            state = np.concatenate([self.state, self.goal], axis=0)
        if self.parametric_gyro:
            state = np.concatenate([state, self.mu], axis=0)

        return state[self.state_ind:], {}

    def double_gyre_flow(self, amplitude, frequency, x, y, t, obs_indexing='xy'):
        '''
        Solve the double gyre flow problem

        Inputs
            amplitude                   (`float`)
            frequency                   (`float`)
            horizontal discretization   (`np.array[float]`, shape: (ny,))
            vertical discretization     (`np.array[float]`, shape: (nx,))
            time vector                 (`np.array[float]`, shape: (ntimes,))

        Output
            horizontal velocity matrix  (`np.array[float]`, shape: (ntimes, nx * ny)
            vertical velocity matrix    (`np.array[float]`, shape: (ntimes, nx * ny)
        '''

        xgrid, ygrid = np.meshgrid(x, y, indexing=obs_indexing)  # spatial grid

        u = np.zeros((len(t), len(x), len(y)))  # horizontal velocity
        v = np.zeros((len(t), len(x), len(y)))  # vertical velocity

        intensity = self.intensity  # intensity parameter

        f = lambda x, t: amplitude * np.sin(frequency * t) * x ** 2 + x - 2 * amplitude * np.sin(frequency * t) * x

        # compute solution
        for i in range(len(t)):
            u[i] = (-np.pi * intensity * np.sin(np.pi * f(xgrid, t[i])) * np.cos(np.pi * ygrid)).T
            v[i] = (np.pi * intensity * np.cos(np.pi * f(xgrid, t[i])) * np.sin(np.pi * ygrid) * (
                        2 * amplitude * np.sin(frequency * t[i]) * xgrid + 1.0 - 2 * amplitude * np.sin(
                    frequency * t[i]))).T

        return u, v

    def trajectory_gif(self, x, y, u, v, Lx, nx, Ly, ny, traj, acts, offset=0.1, name='name.gif',
                       title=None, fontsize=None, figsize=None, axis=False, save=True):
        """
        Trajectory gif
        Input: trajectory with dimension (sequence length, data shape), related plot function for a snapshot, plot options, save option and save path
        """
        def vorticity(u, v):
            dx = Lx / nx
            dy = Ly / ny
            du_dy = np.gradient(u, dy, axis=1)
            dv_dx = np.gradient(v, dx, axis=0)
            return dv_dx - du_dy

        arrays = []

        for i in range(0, traj.shape[0], 5):

            fig, ax = plt.subplots()
            ax.contourf(x, y, vorticity(u[i], v[i]).T, cmap='viridis', levels=100)
            ax.streamplot(x, y, u[i].T, v[i].T, color='black', linewidth=1, density=1)
            plt.axis('off')
            plt.axis([0 - offset, Lx + offset, 0 - offset, Ly + offset])
            plt.title(f'Solution at time t = {round(i, 3)}')
            plt.grid(True)

            ax.scatter(self.goal[0], self.goal[1], c='r', marker="$g$", s=70, alpha=1.0, zorder=4)
            ax.scatter(traj[i][1], traj[i][2], c='y', marker="h", s=90, zorder=4)

            #p = ax.scatter(2 * traj[i, 1], traj[i, 2], s=10, cmap=plt.get_cmap('spring'), zorder=3)
            p2 = ax.contourf(x, y, vorticity(u[i], v[i]).T, cmap='viridis', levels=100)

            norm = np.linalg.norm(acts, axis=1)

            plt.quiver(traj[i, 1], traj[i, 2], acts[i, 0], acts[i, 1], norm[i], cmap='OrRd',
                       zorder=2)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            #plt.colorbar(p, cax=cax)
            plt.colorbar(p2, cax=cax)

            fig.canvas.draw()
            if not axis:
                plt.axis('off')
            fig = plt.gcf()
            display(fig)
            if save:
                arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
            plt.close()
            clc(wait=True)

        if save:
            kargs = {'duration': 1}
            imageio.mimsave(name, arrays, **kargs)

    def render(self, name, idx, best_policy, agent_type, cost, i=999):
        try:
            plt
        except NameError:
            return

        save_dir = "figures/" + self.navigation_mode + '/' + "/testing_seed_" + str(self.seed)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # spatial discretization
        nx = 50
        ny = 25
        Lx = 2.0
        Ly = 1.0
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        radius = self.target_radius
        thetas = np.linspace(-np.pi, np.pi, 200)

        # temporal discretization
        dt = self.dt
        T = self.max_ubtime
        t = np.arange(0, T + dt, dt)

        u, v = self.double_gyre_flow(self.amplitude, self.frequency, x, y, t)

        def vorticity(u, v):
            dx = Lx / nx
            dy = Ly / ny
            du_dy = np.gradient(u, dy, axis=1)
            dv_dx = np.gradient(v, dx, axis=0)
            return dv_dx - du_dy

        offset = 0.1

        fig, ax = plt.subplots()
        ax.contourf(x, y, vorticity(u[-1], v[-1]).T, cmap='viridis', levels=100)
        ax.streamplot(x, y, u[-1].T, v[-1].T, color='black', linewidth=1, density=1)
        plt.axis('off')
        plt.axis([0 - offset, Lx + offset, 0 - offset, Ly + offset])
        plt.title(f'Solution at time t = {round(0, 3)}')
        plt.grid(True)

        self.traj = np.array(self.ep_traj)
        self.acts = np.array(self.ep_acts)

        ax.scatter(self.goal[0], self.goal[1], c='r', marker="$g$", s=70, alpha=1.0, zorder=4)
        ax.plot(self.goal[0]+radius*np.cos(thetas), self.goal[1]+radius*np.sin(thetas), c='r', zorder=4)
        ax.set_aspect('equal', adjustable='box')
        ax.scatter(self.traj[0][1], self.traj[0][2], c='y', marker="$s$", s=70, zorder=4)

        z = np.arange(self.traj[:-1].shape[0])
        p = ax.scatter(self.traj[:-1, 1], self.traj[:-1, 2], c=z, s=10, cmap=plt.get_cmap('spring'), zorder=3)

        norm = np.linalg.norm(self.acts, axis=1)

        plt.quiver(self.traj[:-1, 1], self.traj[:-1, 2], self.acts[:, 0], self.acts[:, 1], norm, cmap='OrRd', zorder=2)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(p, cax=cax)

        if best_policy:
            plt.savefig(save_dir + '/' + name + '_' + self.navigation_mode + '_rel-position:_' + str(self.rel_position) + '_solution_best_' + str(idx) + '_' + str(i) + ".png", dpi=300)
        else:
            plt.savefig(save_dir + '/' + name + '_' + self.navigation_mode + '_rel-position:_' + str(self.rel_position) + '_solution_' + str(idx) + '_' + str(i) + ".png", dpi=300)
        plt.close()

        # if self.ep_target_reached[0]:
        #     self.trajectory_gif(x, y, u, v, Lx, nx, Ly, ny, self.traj[:-1], self.acts,
        #                         name=save_dir + '/' + name + '_' + self.navigation_mode + '_solution_' + str(idx) + '_' + str(i) + ".gif")