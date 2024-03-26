from dataclasses import dataclass
from typing import Callable

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
YEAR = DAY * 365.2422


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

        k2v = accel([Body(other.r + k1r * (dt / 2), other.v + k1v * (dt / 2), other.m) for other in last.bodies], i)
        k2r = this.v + k1v * (dt / 2)

        k3v = accel([Body(other.r + k2r * (dt / 2), other.v + k2v * (dt / 2), other.m) for other in last.bodies], i)
        k3r = this.v + k2v * (dt / 2)

        k4v = accel([Body(other.r + k3r * dt, other.v + k3v * dt, other.m) for other in last.bodies], i)
        k4r = this.v + k3v * dt

        predicted_v = this.v + (dt/6) * (k1v + 2 * k2v + 2 * k3v + k4v)
        predicted_r = this.r + (dt/6) * (k1r + 2 * k2r + 2 * k3r + k4r)

        bodies.append(Body(predicted_r, predicted_v, this.m))

    state = State(last.time + dt, bodies)
    return state

# def ivp_rk4(states: list[State], T: float, dt: float) -> tuple[np.array, np.array]:
#     last = states[-1]
#
#     num_steps = int(T/dt) + 1
#
#     times = np.linspace(0, T, num_steps)
#     positions = np.array([body.r for body in last.bodies]).reshape(1,len(last.bodies) * len(last.bodies[-1].r))
#
#     for i in range(num_steps - 1):
#         next_state = rk4(states, dt)
#         states.append(next_state)
#
#         bodies_positions = next_state.bodies[0].r
#         for i in range(1, len(next_state.bodies)):
#             bodies_positions = np.block([[bodies_positions, next_state.bodies[i].r]])
#         positions = np.block([[positions], [bodies_positions]])
#
#     return positions, times

# def rk4_error(states: list[State], T: float, dt: float, dt_baseline: float) -> float:
#
#     positions, _ = ivp_rk4(states, T, dt)
#     positions_baselines, _ = ivp_rk4(states, T, dt_baseline)
#
#     err = np.linalg.norm(positions[-1] - positions_baselines[-1])/np.linalg.norm(positions_baselines[-1])
#
#     return err


def ivp(initial: State, final_t: float, dt: float, method: Callable[[list[State], float], State], festeps: int = 0) -> State:
    states = [initial]
    for _ in range(festeps):
        states.append(forward_euler(states, dt))

    t = 0.0
    while t < final_t:
        this_step = min(final_t - t, dt)
        next_state = method(states, this_step)
        states.append(next_state)
        t += this_step

    return states[-1]


def main():
    initial = State(0.0, [
        Body(np.array([0, 0.0]), np.array([0.0, 0.0]), MASS_SUN),
        Body(np.array([R_EARTH, 0.0]), np.array([0.0, V_EARTH]), MASS_EARTH),
    ])

    ## plot ab3
    # for _ in range(2):
    #     next = forward_euler(states, dt)
    #     states.append(next)
    #     xs.append(next.time * YEAR / SECOND)
    #     ys.append(next.bodies[1].r[0])

    # for _ in range(2, 1000 * 4):
    #     next = ab3(states, dt)
    #     states.append(next)
    #     xs.append(next.time * YEAR / SECOND)
    #     ys.append(next.bodies[1].r[0])

    for name, (method, fe_steps) in {"ab3": (ab3, 2), "fe": (forward_euler, 0), "rk4": (rk4, 0)}.items():
        baseline = ivp(initial, YEAR, 30 * MINUTE, method, festeps=fe_steps)

        dt_list = [HOUR, 6 * HOUR, 12 * HOUR, DAY]
        errs = []
        for dt in dt_list:
            final = ivp(initial, YEAR, dt, method, festeps=fe_steps)

            err = np.linalg.norm(final.bodies[1].r - baseline.bodies[1].r)/np.linalg.norm(baseline.bodies[1].r)
            errs.append(err)
            print(f"{dt}: {err}")

        plt.plot(np.array(dt_list), np.array(errs), label=name)
    # plt.xscale("log")
    # plt.scatter(xs, ys)
    plt.legend()
    plt.title("Error Convergence")
    plt.xlabel("dt")
    plt.ylabel("error")
    plt.show()


main()