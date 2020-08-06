#!/usr/bin/env python3
# coding:utf-8

import casadi as ca
import time
from mpcc_plot import *

def def_func_err():
# ref:[3*1] pt:[3*1] -> err_sl:[3*1]
    pt_ref = ca.SX.sym('pt_ref', 3, 1)
    pt_x = ca.SX.sym('pt_x]', 3, 1)
    dx = pt_x[0] - pt_ref[0]
    dy = pt_x[1] - pt_ref[1]
    theta_p = pt_ref[2]
    err_sl= ca.vertcat(-np.cos(theta_p) * dx - np.sin(theta_p) * dy, np.sin(theta_p) * dx - np.cos(theta_p) * dy, pt_x[2] - pt_ref[2])
    return ca.Function("f_err", [pt_x, pt_ref], [err_sl], ['pt_x]', 'pt_ref'], ['err_sl'])

def def_state_trans():
# input:[5*1] -> output[3*1]
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    phi = ca.SX.sym('phi')
    v = ca.SX.sym('v')
    steering = ca.SX.sym('steering')
    input = ca.vertcat(x, y, phi, v, steering)
    output = ca.vertcat(x + v * dT * np.cos(phi), y + v * dT * np.sin(phi), phi + v * dT * np.tan(steering) / l)
    return ca.Function('state_trans', [input], [output], ['current_state'], ['next_state'])


def def_opt_problem(obstacles):
    horizon_status = ca.SX.sym('horizon_status', 5, N)
    initial_status = ca.SX.sym('initial_status', 3, 1)

    horizon_ref = ca.SX.sym('horizon_ref', 3, N)
    obstacle_path = ca.SX.sym('dynamic_obj', 3, obstacles.shape[1])

    Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    obj = 0
    constraint = []

    state_trans_5_to_3 = def_state_trans()
    f_sl_err_3_3_to_3 = def_func_err()

    next_state = initial_status
    last_input = np.zeros((2,1))
    for i in range(N):
        E = f_sl_err_3_3_to_3(horizon_status[:3, i], horizon_ref[:, i])
        dU = horizon_status[3:, i] - last_input
        # E = horizon_status[:3, i] - horizon_ref[:, i]
        obj = obj + ca.mtimes([E.T, Q, E]) + ca.mtimes([dU.T, R, dU])

        for obs_id in range(int(obstacles.shape[1] / N)):
            dx = horizon_status[0, i] - obstacle_path[0, i + N * obs_id]
            dy = horizon_status[1, i] - obstacle_path[1, i + N * obs_id]
            constraint.append(dx * dx / 0.25 + dy * dy / 0.04)

        for j in range(3):
            constraint.append(horizon_status[j, i] - next_state[j])
        next_state = state_trans_5_to_3(horizon_status[:, i])
        last_input = horizon_status[3:, i]

    nlp_prob = {}
    nlp_prob['f'] = obj
    nlp_prob['x'] = ca.reshape(horizon_status, -1, 1)
    nlp_prob['p'] = ca.vertcat(ca.reshape(horizon_ref, -1, 1), ca.reshape(initial_status, -1, 1), ca.reshape(obstacle_path, -1, 1))
    # nlp_prob['p'] = ca.vertcat(ca.reshape(horizon_ref, -1, 1), ca.reshape(initial_status, -1, 1))
    nlp_prob['g'] = ca.vertcat(*constraint)
    # ipot设置
    opts_setting = {'ipopt.max_iter': 500, 'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-5, 'ipopt.acceptable_obj_change_tol': 1e-6}
    opts_setting = {}
    return ca.nlpsol('cost_func', 'ipopt', nlp_prob, opts_setting)
    # return ca.nlpsol('cost_func', 'ipopt', nlp_prob)

def cal_boundary(obstacles):
    lbx = []
    ubx = []
    lbg = []
    ubg = []

    for _ in range(N):
        lbx.append(float("-inf"))
        ubx.append(float("inf"))
        lbx.append(float("-inf"))
        ubx.append(float("inf"))
        lbx.append(float("-inf"))
        ubx.append(float("inf"))

        lbx.append(0)
        ubx.append(v_max)
        lbx.append(-steering_max)
        ubx.append(steering_max)

    for i in range(N):
        for obs_id in range(int(obstacles.shape[1] / N)):
            lbg.append(1)
            ubg.append(float("inf"))

        for j in range(3):
            lbg.append(0)
            ubg.append(0)
    return lbx, ubx, lbg, ubg

def MPCC(start_time, initial_pose, obstacles):
#initial_pose:[3*1]
    cost_func = def_opt_problem(obstacles)
    lbx, ubx, lbg, ubg = cal_boundary(obstacles)

    # u0 = np.array([[initial_pose[0, 0] + n*dT*v_max, 0, 0, 0,0] for n in range(N)])
    u0 = np.array([[initial_pose[0, 0], 1, 0, 0, 0] for n in range(N)])

    start_simu_ = time.time()
    r_p = ref_path(start_time)
    c_p = np.concatenate((r_p.T.reshape(-1, 1), initial_pose.T.reshape(-1, 1), obstacles.T.reshape(-1, 1)))
    # c_p = np.concatenate((r_p.T.reshape(-1, 1), initial_pose.T.reshape(-1, 1)))
    res = cost_func(x0=ca.reshape(u0, -1, 1), p=c_p.T, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    u_sol = ca.reshape(res['x'], 5, N)
    return u_sol

def main():
    rospy.init_node("mpcc")
    loop_rate = rospy.Rate(1/dT)

    time = 0
    status = np.array([time * 1.0, 0, 0]).reshape(3, 1)
    state_trans_5_to_3 = def_state_trans()
    while not rospy.is_shutdown():
        obstacle_paths = dynamic_obstacles(time)
        # obstacle_paths = np.zeros(obstacle_paths.shape)
        u_sol = MPCC(time, status, obstacle_paths)

        plot_ref(time)
        plot_arrows(status)
        plot_horizon(u_sol)
        plot_obstacles(obstacle_paths)

        status = u_sol[:3, 1].full()
        # status = state_trans_5_to_3(u_sol[:, 0]).full()
        time += dT
        loop_rate.sleep()

if __name__ == '__main__':
    main()
