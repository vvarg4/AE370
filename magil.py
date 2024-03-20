from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

G = 6.674e-11
MASS_EARTH = 5.972e24
V_EARTH = 29.8e3
R_EARTH = 149e9
MASS_SUN = 1.989e30

SECOND = 1
MINUTE = SECOND * 60
HOUR  = MINUTE * 60
DAY = HOUR * 24
YEAR = DAY * 7


@dataclass
class Body:
    r: np.array
    v: np.array
    m: float

@dataclass
class State:
    time: float
    bodies: list[Body]

def forward_euler(states: list[State], dt: float) -> State:
    last = states[-1]

    bodies = []
    for this in last.bodies:
        force = np.array([0.0, 0.0])
        for other in last.bodies:
            if other is not this:
                force += G * this.m * other.m * (other.r - this.r) / np.linalg.norm(other.r - this.r)**3.0
        accel = force / this.m

        v = this.v + accel * dt
        r = this.r + this.v * dt
        bodies.append(Body(r, v, this.m))

    state = State(last.time + dt, bodies)
    return state


def main():
    initial = State(0.0, [
        Body(np.array([0, 0.0]), np.array([0.0, 0.0]), MASS_SUN),
        Body(np.array([R_EARTH, 0.0]), np.array([0.0, V_EARTH]), MASS_EARTH),
    ])

    xs = []
    ys = []

    states = [initial]
    for _ in range(1000 * 4):
        next = forward_euler(states, DAY / 4)
        states.append(next)
        xs.append(next.time * YEAR / SECOND)
        ys.append(next.bodies[1].r[0])

    plt.plot(xs, ys)
    plt.title("Earth-Sun System")
    plt.xlabel("Year")
    plt.ylabel("X-position of Earth")
    plt.show()


main()