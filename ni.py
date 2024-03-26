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


def accel(bodies: list[Body], index: int) -> np.array:
    this = bodies[index]
    force = np.array([0.0, 0.0])
    for other in bodies:
        if other is not this:
            force += G * this.m * other.m * (other.r - this.r) / np.linalg.norm(other.r - this.r) ** 3.0
    accel = force / this.m
    return accel

def forward_euler(states: list[State], dt: float) -> State:
    last = states[-1]

    bodies = []
    for i in range(len(last.bodies)):
        this = last.bodies[i]

        v = this.v + accel(last.bodies, i) * dt
        r = this.r + this.v * dt
        bodies.append(Body(r, v, this.m))

    state = State(last.time + dt, bodies)
    return state

def ab3(states: list[State], dt: float) -> State:
    last = states[-1]
    second_last = states[-2]
    third_last = states[-3]

    bodies = []
    for i in range(len(last.bodies)):
        this = last.bodies[i]
        second = second_last.bodies[i]
        third = third_last.bodies[i]

        predicted_r = this.r + dt * (23/12 * this.v - 4/3 * second.v + 5/12 * third.v)
        predicted_v = this.v + dt * (23/12 * accel(last.bodies, i) - 4/3 * accel(second_last.bodies, i) + 5/12 * accel(third_last.bodies, i))
        bodies.append(Body(predicted_r, predicted_v, this.m))

    state = State(last.time + dt, bodies)
    return state

def rk4(states: list[State], dt: float) -> State:
    last = states[-1]

    bodies = []
    for i in range(len(last.bodies)):
        this = last.bodies[i]

        k1v = accel(last.bodies, i)
        k1r = this.v
        
        k2v = accel([Body(this.r + k1r * (dt / 2), this.v + k1v * (dt / 2), this.m) for this in last.bodies], i)
        k2r = this.v + k1v * (dt / 2)

        k3v = accel([Body(this.r + k2r * (dt / 2), this.v + k2v * (dt / 2), this.m) for this in last.bodies], i)
        k3r = this.v + k2v * (dt / 2)

        k4v = accel([Body(this.r + k3r * dt, this.v + k3v * dt, this.m) for this in last.bodies], i)
        k4r = this.v + k3v * dt

        predicted_v = this.v + (dt/6) * (k1v + 2 * k2v + 2 * k3v + k4v)
        predicted_r = this.r + (dt/6) * (k1r + 2 * k2r + 2 * k3r + k4r)
        bodies.append(Body(predicted_r, predicted_v, this.m))
    
    state = State(last.time + dt, bodies)
    return state


def main():
    initial = State(0.0, [
        Body(np.array([0, 0.0]), np.array([0.0, 0.0]), MASS_SUN),
        Body(np.array([R_EARTH, 0.0]), np.array([0.0, V_EARTH]), MASS_EARTH),
    ])

    xs = []
    ys = []

    delta_t = DAY / 4

    states = [initial]
    # for _ in range(2):
    #     next = forward_euler(states, delta_t)
    #     states.append(next)
    #     xs.append(next.time * YEAR / SECOND)
    #     ys.append(next.bodies[1].r[0])

    # for _ in range(2, 1000 * 4):
    #     next = ab3(states, delta_t)
    #     states.append(next)
    #     xs.append(next.time * YEAR / SECOND)
    #     ys.append(next.bodies[1].r[0])

    for _ in range(1000 * 4):
        next = rk4(states, delta_t)
        states.append(next)
        xs.append(next.time * YEAR / SECOND)
        ys.append(next.bodies[1].r[0])

    print(np.max(ys))



    plt.plot(xs, ys)
    plt.title("Earth-Sun System")
    plt.xlabel("Year")
    plt.ylabel("X-position of Earth")
    plt.show()


main()