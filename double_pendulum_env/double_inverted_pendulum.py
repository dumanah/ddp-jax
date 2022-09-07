from functools import partial
import math
import gym
from typing import Union
from gym import spaces, logger
from gym.error import DependencyNotInstalled
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit
from diffrax import diffeqsolve, ODETerm, Dopri5


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):
        self.g = 9.81  # gravity constant
        self.m0 = 0.5  # mass of cart
        self.m1 = 0.1  # mass of pole 1
        self.m2 = 0.1  # mass of pole 2
        self.L1 = 1  # length of pole 1
        self.L2 = 1  # length of pole 2
        self.l1 = self.L1 / 2  # distance from pivot point to center of mass
        self.l2 = self.L2 / 2  # distance from pivot point to center of mass
        self.I1 = (
            self.m1 * (self.L1**2) / 12
        )  # moment of inertia of pole 1 w.r.t its center of mass
        self.I2 = (
            self.m2 * (self.L2**2) / 12
        )  # moment of inertia of pole 2 w.r.t its center of mass

        # calculations will help build state matrixes
        self.d1 = self.m0 + self.m1 + self.m2
        self.d2 = self.m1 * self.l1 + self.m2 * self.L1
        self.d3 = self.m2 * self.l2
        # self.d4 = self.m1 * (self.l1**2) + self.m2 * (self.L1**2) + self.I1
        self.d4 = (self.m1 / 3 + self.m2) * (self.L1**2)
        self.d5 = self.m2 * self.L1 * self.l2
        # self.d6 = self.m2 * (self.l2**2) + self.I2
        self.d6 = self.m2 * (self.L2**2) / 3
        self.f1 = (self.m1 + self.m2) * self.l1 * self.g
        self.f2 = self.m2 * self.l2 * self.g

        self.tau = 0.02  # seconds between state updates
        self.max_force = 100
        self.counter = 0

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # # (never fail the episode based on the angle)
        self.theta_threshold_radians = 100000 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.y_threshold = self.L1 * 2

        self.diffrax_solver = Dopri5()

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_done = None

    @partial(jit, static_argnums=(0,))
    def state_eq(self, t, state, u):
        x, theta, phi, x_dot, theta_dot, phi_dot = state

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        cosphi = jnp.cos(phi)
        sinphi = jnp.sin(phi)
        costheta_phi = jnp.cos(theta - phi)
        sintheta_phi = jnp.sin(theta - phi)

        D = jnp.array(
            [
                [self.d1, self.d2 * costheta, self.d3 * cosphi],
                [self.d2 * costheta, self.d4, self.d5 * costheta_phi],
                [self.d3 * cosphi, self.d5 * costheta_phi, self.d6],
            ]
        )

        invD = jnp.linalg.inv(D)

        C = jnp.array(
            [
                [0, -self.d2 * sintheta * theta_dot, -self.d3 * sinphi * phi_dot],
                [0, 0, self.d5 * sintheta_phi * phi_dot],
                [0, -self.d5 * sintheta_phi * theta_dot, 0],
            ]
        )

        G = jnp.array([[0], [-self.f1 * sintheta], [-self.f2 * sinphi]])
        H = jnp.array([[1], [0], [0]])

        x_dot_dot, theta_dot_dot, phi_dot_dot = jnp.squeeze(
            -invD @ C @ jnp.array([[x_dot], [theta_dot], [phi_dot]])
            + (-invD @ G)
            + (invD @ H) * u[0]
        )
        return jnp.array(
            [x_dot, theta_dot, phi_dot, x_dot_dot, theta_dot_dot, phi_dot_dot]
        )

    def next_state(self, st, u):

        solution = diffeqsolve(
            ODETerm(self.state_eq),
            self.diffrax_solver,
            t0=0,
            t1=self.tau,
            dt0=self.tau,
            y0=st,
            args=u,
        )
        return jnp.asarray(solution.ys[0])

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        state = self.state
        self.state = self.next_state(state, action)
        x, theta, phi, x_dot, theta_dot, phi_dot = self.state

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = jnp.array([0.0, jnp.pi, jnp.pi, 0.0, 0.0, 0.0])
        self.steps_beyond_done = None
        self.counter = 0
        return self.state

    def render(self, mode="human", close=False):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        screen_width = 1280
        screen_height = 1000

        world_width = self.x_threshold * 2
        scale_width = screen_width / world_width
        polewidth = 20.0
        polelen = scale_width * self.L1
        cartwidth = 100.0
        cartheight = 60.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((238, 238, 238))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale_width + screen_width / 2.0  # MIDDLE OF CART
        carty = 500  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.hline(self.surf, 0, screen_width, carty, (48, 56, 65))
        gfxdraw.aapolygon(self.surf, cart_coords, (48, 56, 65))
        gfxdraw.filled_polygon(self.surf, cart_coords, (48, 56, 65))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[1])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (255, 87, 34))
        gfxdraw.filled_polygon(self.surf, pole_coords, (255, 87, 34))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (0, 173, 181),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (0, 173, 181),
        )

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole2_axs_x = pole_coords[2][0] + r
        pole2_axs_y = pole_coords[2][1] + r
        pole2_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (
                coord[0] + pole2_axs_x,
                coord[1] + pole2_axs_y,
            )
            pole2_coords.append(coord)

        gfxdraw.aapolygon(self.surf, pole2_coords, (255, 87, 34))
        gfxdraw.filled_polygon(self.surf, pole2_coords, (255, 87, 34))

        gfxdraw.aacircle(
            self.surf,
            int(pole2_axs_x),
            int(pole2_axs_y),
            int(polewidth / 2),
            (0, 173, 181),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(pole2_axs_x),
            int(pole2_axs_y),
            int(polewidth / 2),
            (0, 173, 181),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
