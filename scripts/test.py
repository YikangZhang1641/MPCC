#!/usr/bin/env python3
# coding:utf-8

import casadi as ca
import time
from mpcc_plot import *

def def_func_err():
# 定义err转换func
    pt_ref = ca.SX.sym('pt_ref', 3, 1)
    pt_x = ca.SX.sym('pt_x]', 3, 1)
    dx = pt_x[0] - pt_ref[0]
    dy = pt_x[1] - pt_ref[1]
    theta_p = pt_ref[2]
    err_sl= ca.vertcat(-np.cos(theta_p) * dx - np.sin(theta_p) * dy, np.sin(theta_p) * dx - np.cos(theta_p) * dy, pt_x[2] - pt_ref[2])
    return ca.Function("f_err", [pt_x, pt_ref], [err_sl], ['pt_x]', 'pt_ref'], ['err_sl'])

def def_state_trans():
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    phi = ca.SX.sym('phi')
    v = ca.SX.sym('v')
    steering = ca.SX.sym('steering')
    input = ca.vertcat(x, y, phi, v, steering)
    output = ca.vertcat(x + v * dT * np.cos(phi), y + v * dT * np.sin(phi), phi + v * dT * np.tan(steering) / l, 0, 0)
    return ca.Function('state_trans', [input], [output], ['current_state'], ['next_state'])


def MPCC(start_time, initial_pose, obstacles):

# 运动学模型

    horizon_status = ca.SX.sym('horizon_status', 5, N + 1)
    initial_status = ca.SX.sym('initial_status', 5, 1)

    horizon_ref = ca.SX.sym('horizon_ref', 3, N)
    obstacle_path = ca.SX.sym('dynamic_obj', 3, obstacles.shape[1])


# NLP问题
    # 惩罚矩阵
    Q = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 优化目标
    obj = 0  # 初始化优化目标值
    constraint = []
    lbg = []
    ubg = []

    constraint.append(horizon_status[0, 0] - initial_status[0])
    lbg.append(0)
    ubg.append(0)
    constraint.append(horizon_status[1, 0] - initial_status[1])
    lbg.append(0)
    ubg.append(0)
    constraint.append(horizon_status[2, 0] - initial_status[2])
    lbg.append(0)
    ubg.append(0)

    state_trans = def_state_trans()
    f_err = def_func_err()

    for i in range(N):
        E = f_err(horizon_status[:3, i+1], horizon_ref[:, i])
        obj = obj + ca.mtimes([E.T, Q, E]) + ca.mtimes([horizon_status[3:, i].T, R, horizon_status[3:, i]])

        dx = horizon_status[0, i+1] - obstacle_path[0, i]
        dy = horizon_status[1, i+1] - obstacle_path[1, i]
        constraint.append(dx * dx / 0.25 + dy * dy / 0.04)
        lbg.append(1)
        ubg.append(float("inf"))

        dx = horizon_status[0, i+1] - obstacle_path[0, N+i]
        dy = horizon_status[1, i+1] - obstacle_path[1, N+i]
        constraint.append(dx * dx / 0.25 + dy * dy / 0.04)
        lbg.append(1)
        ubg.append(float("inf"))

        dx = horizon_status[0, i+1] - obstacle_path[0, 2*N+i]
        dy = horizon_status[1, i+1] - obstacle_path[1, 2*N+i]
        constraint.append(dx * dx / 0.25 + dy * dy / 0.04)
        lbg.append(1)
        ubg.append(float("inf"))

        next_state = state_trans(horizon_status[:, i])
        constraint.append(horizon_status[0, i + 1] - next_state[0])
        lbg.append(0)
        ubg.append(0)
        constraint.append(horizon_status[1, i + 1] - next_state[1])
        lbg.append(0)
        ubg.append(0)
        constraint.append(horizon_status[2, i + 1] - next_state[2])
        lbg.append(0)
        ubg.append(0)


    status_list = ca.vertcat(ca.reshape(horizon_ref, -1, 1), ca.reshape(initial_status, -1, 1), ca.reshape(obstacle_path, -1, 1))
    opt_x = ca.reshape(horizon_status, -1, 1)
    nlp_prob = {'f': obj, 'x': opt_x, 'p': status_list, 'g': ca.vertcat(*constraint)}
    # ipot设置
    opts_setting = {'ipopt.max_iter': 500, 'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-5, 'ipopt.acceptable_obj_change_tol': 1e-6}
    # 最终目标，获得求解器
    cost_func = ca.nlpsol('cost_func', 'ipopt', nlp_prob, opts_setting)

    lbx = []  # 最低约束条件
    ubx = []  # 最高约束条件
    for _ in range(N + 1):
        lbx.append(float("-inf"))
        ubx.append(float("inf"))
        lbx.append(float("-inf"))
        ubx.append(float("inf"))
        lbx.append(float("-inf"))
        ubx.append(float("inf"))

        lbx.append(-v_max)
        ubx.append(v_max)
        lbx.append(-steering_max)
        ubx.append(steering_max)


    print(initial_pose)
    u0 = np.array([[initial_pose[0, 0], 0,0, 0,0] for n in range(N+1)])
    # u0 = np.array([[initial_pose[0, 0] + n * dT * v_max, initial_pose[1, 0], initial_pose[2, 0], 0, 0] for n in range(N+1)])

    start_simu_ = time.time()
    r_p = ref_path(start_time)
    c_p = np.concatenate((r_p.T.reshape(-1, 1), initial_pose.T.reshape(-1, 1), obstacles.T.reshape(-1, 1)))
    res = cost_func(x0=ca.reshape(u0, -1, 1), p=c_p.T, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    u_sol = ca.reshape(res['x'], 5, N + 1)
    # ff_value = next_horizon_status(u_sol, initial_pose)

    print(time.time() - start_simu_)
    return u_sol

def main():
    rospy.init_node("mpcc")
    loop_rate = rospy.Rate(10)

    status = np.zeros((5, 1))
    state_trans = def_state_trans()
    time = 0
    while not rospy.is_shutdown():
        obstacle_paths = dynamic_obstacles(time)
        u_sol = MPCC(time, status, obstacle_paths)
        # next_state = state_trans(u_sol[:, 0]).full()
        next_state = state_trans(state_trans(u_sol[:, 0])).full()

        plot_ref(time)
        plot_arrows(status)
        plot_horizon(u_sol)
        plot_obstacles(obstacle_paths)

        status = next_state
        time += dT
        loop_rate.sleep()

if __name__ == '__main__':
    main()
