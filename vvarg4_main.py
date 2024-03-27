from dataclasses import dataclass

import numpy as np
import math
import matplotlib.pyplot as plt

G = 6.67430e-11
MASS_EARTH = 5.972e24
V_EARTH = 29.8e3
R_EARTH = 149e9
MASS_SUN = 1.989e30
MASS_MOON = 7.349e22
MASS_VENUS = 48.685e23

SECOND = 1
MINUTE = SECOND * 60
HOUR  = MINUTE * 60
DAY = HOUR * 24
YEAR = DAY * 365


@dataclass
class Body:
    r: np.array
    v: np.array
    m: float

@dataclass
class State:
    time: float
    bodies: list[Body]

def test_stability_FE (states, true_final_state, T, dt_list, rtol):
    closeness = [0] * len(dt_list)
    for i in range(len(dt_list)):
        positions, _ = ivp_forward_euler(states, T, dt_list[i])
        true_pos = np.array([body.r for body in true_final_state.bodies]).reshape(1,len(true_final_state.bodies) * len(true_final_state.bodies[-1].r))
        err = np.linalg.norm(positions[-1] - true_pos)/np.linalg.norm(true_pos)
        closeness[i] = err - rtol
    return closeness

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

def ivp_forward_euler(states: list[State], T: float, dt: float) -> tuple[np.array, np.array]:
    last = states[-1]

    times = np.arange(0, T+dt, dt)
    positions = np.array([body.r for body in last.bodies]).reshape(1,len(last.bodies) * len(last.bodies[-1].r))

    for i in range(len(times)):
        next_state = forward_euler(states, dt)
        states.append(next_state)
        
        bodies_positions = next_state.bodies[0].r
        for i in range(1, len(next_state.bodies)):
            bodies_positions = np.block([[bodies_positions, next_state.bodies[i].r]])
        positions = np.block([[positions], [bodies_positions]])
    
    return positions, times

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

def ivp_rk4(states: list[State], T: float, dt: float) -> tuple[np.array, np.array]:
    last = states[-1]

    num_steps = int(T/dt) + 1

    times = np.linspace(0, T, num_steps)
    positions = np.array([body.r for body in last.bodies]).reshape(1,len(last.bodies) * len(last.bodies[-1].r))

    for i in range(num_steps - 1):
        next_state = rk4(states, dt)
        states.append(next_state)
        
        bodies_positions = next_state.bodies[0].r
        for i in range(1, len(next_state.bodies)):
            bodies_positions = np.block([[bodies_positions, next_state.bodies[i].r]])
        positions = np.block([[positions], [bodies_positions]])
    
    return positions, times

def rk4_error(states: list[State], T: float, dt: float, dt_baseline: float) -> float:
    
    positions, _ = ivp_rk4(states, T, dt)
    positions_baselines, _ = ivp_rk4(states, T, dt_baseline)

    err = np.linalg.norm(positions[-1] - positions_baselines[-1])/np.linalg.norm(positions_baselines[-1])
    
    return err

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

def ivp_ab4(states: list[State], T: float, dt: float) -> tuple[np.array, np.array]:
    last = states[-1]

    N = len(last.bodies)
    num_steps = int(T/dt) + 1

    times = np.linspace(0, T, num_steps)
    positions = np.array([body.r for body in last.bodies]).reshape(1, N * len(last.bodies[-1].r))

    for i in range(num_steps - 1):
        next_state = ab4(states, dt)
        states.append(next_state)
        
        bodies_positions = next_state.bodies[0].r
        for i in range(1, N):
            bodies_positions = np.block([[bodies_positions, next_state.bodies[i].r]])
        positions = np.block([[positions], [bodies_positions]])
    
    return positions, times

def ab4_error(states: list[State], T: float, dt: float, dt_baseline: float) -> float:
    
    positions, _ = ivp_ab4(states, T, dt)
    positions_baselines, _ = ivp_ab4(states, T, dt_baseline)

    err = np.linalg.norm(positions[-1] - positions_baselines[-1])/np.linalg.norm(positions_baselines[-1])
    
    return err

def test_err_ab4 (states, true_final_state, T, dt):
    
    positions, _ = ivp_ab4(states, T, dt)
    true_pos = np.array([body.r for body in true_final_state.bodies]).reshape(1,len(true_final_state.bodies) * len(true_final_state.bodies[-1].r))
    err = np.linalg.norm(positions[-1] - true_pos)/np.linalg.norm(true_pos)
    return err

def ivp_ab3(states: list[State], T: float, dt: float):
    last = states[-1]

    N = len(last.bodies)
    num_steps = int(T/dt) + 1

    times = np.linspace(0, T, num_steps)
    positions = np.array([body.r for body in last.bodies]).reshape(1, N * len(last.bodies[-1].r))

    for i in range(num_steps - 1):
        next_state = ab3(states, dt)
        states.append(next_state)
        
        bodies_positions = next_state.bodies[0].r
        for i in range(1, N):
            bodies_positions = np.block([[bodies_positions, next_state.bodies[i].r]])
        positions = np.block([[positions], [bodies_positions]])
    
    return positions, times

def test_err_ab3(states, true_final_state, T, dt) :
    positions, _ = ivp_ab3(states, T, dt)
    true_pos = np.array([body.r for body in true_final_state.bodies]).reshape(1,len(true_final_state.bodies) * len(true_final_state.bodies[-1].r))
    err = np.linalg.norm(positions[-1] - true_pos)/np.linalg.norm(true_pos)
    return err

def main():
    #EARTH
    initial_E = State(0.0, [
        Body(np.array([-6.426892426978119E+05, 1.092496276575358E+06, 5.511842399311601E+03]) * 1e3, 
             np.array([-1.424282321392213E-02, -4.418145875965810E-03, 3.964221587209598E-04]) * 1e3, MASS_SUN),
        Body(np.array([-1.404488704344683E+08, 5.034360043102103E+07, 3.921347984042019E+03]) * 1e3, 
             np.array([-1.038988278472088E+01, -2.822001912696788E+01, 9.555115139683323E-04]) * 1e3, MASS_EARTH)
    ])

    # dt_list = np.array([HOUR/4, HOUR/2, HOUR, HOUR * 6, DAY/2, DAY])
    dt_list = np.linspace(DAY*14, HOUR, 10000)
    T = YEAR
    true_final_state_E = State(T, [Body(np.array([-1.406208605071319E+08, 5.072979713648436E+07, 1.537752436613292E+04]) * 1e3, 
             np.array([-1.052328496849902E+01, -2.816083970852659E+01, 2.067846320530364E-03]) * 1e3, MASS_SUN),
        Body(np.array([-1.053706281960470E+06, 8.447647192829703E+05, 1.769267092239828E+04]) * 1e3, 
             np.array([-1.112225417967873E-02, -1.095148673993527E-02, 3.588818045937984E-04]) * 1e3, MASS_EARTH)
    ])

    #VENUS
    initial_V = State(0.0, [
        Body(np.array([-6.426892426978119E+05, 1.092496276575358E+06, 5.511842399311601E+03]) * 1e3, 
             np.array([-1.424282321392213E-02, -4.418145875965810E-03, 3.964221587209598E-04]) * 1e3, MASS_SUN),
        Body(np.array([-2.012858919439126E+07, 1.068629514063284E+08, 2.581303803624652E+06]) * 1e3, 
             np.array([-3.457424218622913E+01, -6.541472476477993E+00, 1.905051786693989E+00]) * 1e3, MASS_VENUS)
    ])
    true_final_state_V = State(T, [Body(np.array([-1.406208605071319E+08, 5.072979713648436E+07, 1.537752436613292E+04]) * 1e3, 
             np.array([-1.052328496849902E+01, -2.816083970852659E+01, 2.067846320530364E-03]) * 1e3, MASS_SUN),
        Body(np.array([8.850038279743193E+07, -6.086359498213534E+07, -5.996916265694849E+06]) * 1e3, 
             np.array([1.964471248657895E+01, 2.868557922953439E+01, -7.400764004281797E-01]) * 1e3, MASS_VENUS)
    ])
    # plot ab4
    states = [initial_E]
    err_rtol = 2
    err_list_ab4 = []
    err_list_ab3 = []
    for dt in dt_list:
        for _ in range(3):
            next = forward_euler(states, dt)
            states.append(next)

        err = test_err_ab4(states, true_final_state_E, T, dt)
        err_list_ab4.append(err)
        if np.allclose(err, err_rtol, rtol = 0.1):
            print(dt)
            break


    for dt in dt_list:
        for _ in range(2):
            next = forward_euler(states, dt)
            states.append(next)

        err = test_err_ab3(states, true_final_state_E, T, dt)
        err_list_ab3.append(err)
        if np.allclose(err, err_rtol, rtol = 0.1):
            print(dt)
            break

    # print("AB4 errors: ", err_list_ab4)
    # print("AB3 errors: ", err_list_ab3)
    # xs = []
    # ys = []
    # zs = []

    # dt = DAY / 1000


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

    # # plot rk4
    # for _ in range(1000 * 4):
    #     next = rk4(states, dt)
    #     states.append(next)
    #     xs.append(next.time * YEAR / SECOND)
    #     ys.append(next.bodies[1].r[0])

## SOLVE IVP RK4
    
    # positions, times = ivp_rk4(states, YEAR, DAY/100)
    # x_list_Earth = []
    # y_list_Earth = []
    # z_list_Earth = []
    # x_list_Moon = []
    # y_list_Moon = []
    # z_list_Moon = []
    # x_list_Sun = []
    # y_list_Sun = []
    # z_list_Sun = []

    # for pos in positions:
    #     x_list_Sun.append(pos[0])
    #     y_list_Sun.append(pos[1])
    #     z_list_Sun.append(pos[2])

    #     x_list_Earth.append(pos[3])
    #     y_list_Earth.append(pos[4])
    #     z_list_Earth.append(pos[5])    

    #     x_list_Moon.append(pos[6])
    #     y_list_Moon.append(pos[7])
    #     z_list_Moon.append(pos[8])
    
    # fig = plt.figure(figsize=(7,7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x_list_Sun, y_list_Sun, z_list_Sun, linewidth=2, color='y', label='Sun')
    # ax.plot(x_list_Earth, y_list_Earth, z_list_Earth, linewidth=1, color = 'b', label='Earth')
    # ax.plot(x_list_Moon, y_list_Moon, z_list_Moon, linewidth=1, color = 'k', label='Moon')

    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('z (m)')

    # plt.show()


    # T = 1000 * dt

    # dt_list = np.array([5e-2 * 100, 2.5e-2 * 100, 1e-2 * 100, 5e-3, 2.5e-3, 1e-3, 5e-4])*DAY

    # dt_baseline = 2.5e-4 * DAY

    # T_baseline = dt_baseline * 1000
    # err_list = []
    # for dt in dt_list:
    #     err = rk4_error(states, T_baseline, dt, dt_baseline)
    #     err_list.append(err)

    # plt.plot(xs,  ys)
    # plt.title("Earth-Sun System")
    # plt.xlabel("Year")
    # plt.ylabel("Y-position of Earth")
    # plt.show()

    # plt.loglog(dt_list,  err_list)
    # plt.plot(dt_list,  err_list)
    # plt.title("Error Convergence")
    # plt.xlabel("dt")
    # plt.ylabel("error")
    # plt.show()

main()