"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Union
import numpy as np

from diffrax import (
    diffeqsolve,
    ODETerm,
    Dopri5,
    Euler,
    Tsit5,
)  # (written for jax) that has different ode solving methods which forward-mode diff. (faster) can be applied.
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit
from functools import partial


class CartPoleContinuousEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self):
        self.gravity = 9.81
        self.masscart = 1
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        # because it will be used many times in calculations
        self.polemass_length = self.masspole * self.length
        self.max_force = 100
        # will be used later in calculations
        self.polemass_gravity = self.masspole * self.gravity
        self.tau = 0.02  # seconds between state updates
        self.diffrax_solver = Tsit5()  # choose ode type to solve with diffrax

        # Angle at which to fail the episode
        self.theta_threshold_radians = 2 * math.pi
        self.x_threshold = 4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
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
        
    @partial(jit,static_argnums=(0,))
    def state_eq(self,t, state, u):
        x, x_dot, theta, theta_dot = state
        force = u[0]
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        calc_den_help = self.masscart + (self.masspole * (sintheta**2))

        x_dot_dot = (
            force
            + self.polemass_length * sintheta * (theta_dot**2)
            - self.polemass_gravity * costheta * sintheta
        ) / calc_den_help
        theta_dot_dot = (
            -force * costheta
            - self.polemass_length * sintheta * costheta * (theta_dot**2)
            + self.total_mass * self.gravity * sintheta
        ) / (self.length * calc_den_help)

        return jnp.array([x_dot, x_dot_dot, theta_dot, theta_dot_dot])

    @partial(jit, static_argnums=(0,))
    def next_state(self, st, u):
        # notice the position of t is different from odeint

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
        x, x_dot, theta, theta_dot = self.state

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
        return np.asarray(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = jnp.array([0.0, 0.0, jnp.pi, 0.0])
        self.steps_beyond_done = None
        return self.state

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

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
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

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
