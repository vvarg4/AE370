import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

G = 6.674e-11
MASS_EARTH = 5.972e24
V_EARTH = 29.8e3
R_EARTH = 149e9
MASS_SUN = 1.989e30
R_MOON = 384.4e6
MASS_MOON = 7.348e22
V_MOON = 1.022e3

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
    force = np.array([0.0, 0.0, 0.0])
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

def ab4(states: list[State], dt: float) -> State:
    last = states[-1]
    second_last = states[-2]
    third_last = states[-3]
    fourth_last = states[-4]

    bodies = []
    for i in range(len(last.bodies)):
        this = last.bodies[i]
        second = second_last.bodies[i]
        third = third_last.bodies[i]
        fourth = fourth_last.bodies[i]

        predicted_r = this.r + dt * (55/24 * this.v - 59/24 * second.v + 37/24 * third.v - 9/24 * fourth.v)
        predicted_v = this.v + dt * (55/24 * accel(last.bodies, i) -
                                     59/24 * accel(second_last.bodies, i) +
                                     37/24 * accel(third_last.bodies, i) -
                                     9/24 * accel(fourth_last.bodies, i))
        bodies.append(Body(predicted_r, predicted_v, this.m))

    state = State(last.time + dt, bodies)
    return state

def ivp(initial: State, final_t: float, dt: float, method: Callable[[list[State], float], State], festeps: int = 0, verbose: bool = False) -> list[State]:
    states = [initial]
    for _ in range(festeps):
        states.append(forward_euler(states, dt))

    i = 0
    t = 0.0
    while t < final_t:
        this_step = min(final_t - t, dt)
        next_state = method(states, this_step)
        states.append(next_state)
        t += this_step
        if verbose and i % 100 == 0:
            print(f"\r{t/final_t*100:.2f}%", end="", flush=True)
        i += 1

    return states


def test_convergence():
    initial = State(0.0, [
        Body(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), MASS_SUN),
        Body(np.array([R_EARTH, 0.0, 0.0]), np.array([0.0, V_EARTH, 0.0]), MASS_EARTH),
        Body(np.array([R_EARTH + R_MOON, 0.0, 0.0]), np.array([0.0, V_EARTH + V_MOON, 0.0]), MASS_MOON)
    ])

    final_t = YEAR
    baseline_dt = 30 * MINUTE
    dt_list = np.array([HOUR, 12 * HOUR, DAY, DAY * 3, DAY * 7, DAY * 14])

    for name, (method, fe_steps) in {"ab3": (ab3, 2), "ab4": (ab4, 3), "fe": (forward_euler, 0)}.items():
        baseline = ivp(initial, final_t, baseline_dt, method, festeps=fe_steps)

        errs = []
        for dt in dt_list:
            final = ivp(initial, final_t, dt, method, festeps=fe_steps)

            err = np.linalg.norm(final[-1].bodies[1].r - baseline[-1].bodies[1].r) / np.linalg.norm(baseline[-1].bodies[1].r)
            errs.append(err)
        print(f"{name} done")

        plt.plot(np.array(dt_list) / HOUR, np.array(errs), label=name)

    plt.legend()
    plt.title("Error Convergence")
    plt.xlabel("dt (hours)")
    plt.ylabel("error")
    plt.show()


def test_method(method):
    initial = State(0.0, [
        Body(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), MASS_SUN),
        Body(np.array([R_EARTH, 0.0, 0.0]), np.array([0.0, V_EARTH, 0.0]), MASS_EARTH),
    ])

    final = ivp(initial, YEAR, 20 * MINUTE, method, festeps=3)

    correct_pos = np.array([R_EARTH, 0.0, 0.0])
    err = np.linalg.norm(final[-1].bodies[1].r - correct_pos) / np.linalg.norm(correct_pos)

    plt.scatter([state.bodies[1].r[0] for state in final], [state.bodies[1].r[1] for state in final])
    plt.show()

    return err

SELECTED_NAVSTARS = {
    "NAVSTAR 62 (USA 201)",
    "NAVSTAR 63 (USA 203)",
    "NAVSTAR 64 (USA 206)",
    "NAVSTAR 65 (USA 213)",
    "NAVSTAR 67 (USA 239)",
    "NAVSTAR 68 (USA 242)",
    "NAVSTAR 69 (USA 248)",
    "NAVSTAR 70 (USA 251)",
    "NAVSTAR 71 (USA 256)",
    "NAVSTAR 72 (USA 258)",
    "NAVSTAR 73 (USA 260)",
    "NAVSTAR 74 (USA 262)",
    "NAVSTAR 75 (USA 265)",
}


def main():
    # print("ab3 error:", test_method(ab3))
    # print("ab4 error:", test_method(ab4))

    initial = State(0.0, [
        Body(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), MASS_SUN),
        Body(np.array([R_EARTH, 0.0, 0.0]), np.array([0.0, V_EARTH, 0.0]), MASS_EARTH),
        Body(np.array([R_EARTH + R_MOON, 0.0, 0.0]), np.array([0.0, V_EARTH + V_MOON, 0.0]), MASS_MOON)
    ])
    added = {"Sun": 0, "Earth": 1, "Moon": 2}
    with open("satellites.json", "r") as satellites:
        satellites = json.load(satellites)
        for satellite in satellites:
            if satellite["name"] in SELECTED_NAVSTARS:
                position = np.array(satellite["position"], dtype=float)
                velocity = np.array(satellite["velocity"], dtype=float)
                initial.bodies.append(Body(
                    np.array([R_EARTH, 0.0, 0.0]) + position * 1000,
                    np.array([0.0, V_EARTH, 0.0]) + velocity * 1000,
                    1.0
                ))
                added[satellite["name"]] = len(initial.bodies) - 1

    final = ivp(initial, DAY * 5, MINUTE, ab3, festeps=2, verbose=True)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    final = final[::max(len(final)//1000, 1)]

    earth = added["Earth"]
    for name in set(added.keys()) - {"Sun", "Moon"}:
        sat = added[name]
        ax.scatter(
            [state.bodies[sat].r[0] - state.bodies[earth].r[0] for state in final],
            [state.bodies[sat].r[1] - state.bodies[earth].r[1] for state in final],
            [state.bodies[sat].r[2] - state.bodies[earth].r[2] for state in final],
            marker=',', s=1
        )
    # ax.scatter(
    #     [state.bodies[2].r[0] - state.bodies[1].r[0] for state in final],
    #     [state.bodies[2].r[1] - state.bodies[1].r[1] for state in final],
    #     [state.bodies[2].r[2] - state.bodies[1].r[2] for state in final],
    #     # linewidths=0,
    #     marker=',', s=1, c="gray"
    # )


    limits = np.r_[ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    limits = [np.min(limits, axis=0), np.max(limits, axis=0)]
    ax.set(xlim3d=limits, ylim3d=limits, zlim3d=limits, box_aspect=(1, 1, 1))
    plt.show()


main()
