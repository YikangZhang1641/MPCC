#!/usr/bin/env python3
# coding:utf-8

import casadi as ca
import numpy as np
import time
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point


def ref_path(start, N = 20, step = 0.2):
    arr = np.array([]).reshape(3, -1)
    for i in range(N):
        arr = np.concatenate((arr, ref_point(start + i * step)), axis = 1)
    return arr


def ref_point(t):
    x = t
    y = 0.5 * np.sin(t)
    theta = np.arctan(0.5 * np.cos(t))
    return np.array([x, y, theta]).reshape(3, -1)


def dynamic_obstacles(start, N = 20, step = 0.2):
    obs0 = []
    obs1 = []
    obs2 = []

    for i in range(N):
        t = start + i * step

        x0 = 10 - 0.6 * t
        y0 = 0.3
        theta0 = np.pi
        obs0.append([x0, y0, theta0])

        x1 = 15 - t
        y1 = 0
        theta1 = np.pi
        obs1.append([x1, y1, theta1])

        x2 = 5 - 1.5 * t
        y2 = -0.2
        theta2 = np.pi
        obs2.append([x2, y2, theta2])
    # return np.array([obs0, obs1, obs2])
    res = np.concatenate([obs0, obs1, obs2]).T
    print(res.shape)
    return res

def shift_movement(dT, x0, u, f):
    f_value = f(x0, u[:, 0])
    status = x0 + dT*f_value
    return status

def def_func_err():
# 定义err转换func
    pt_ref = ca.SX.sym('pt_ref', 3, 1)
    pt_x = ca.SX.sym('pt_x]', 3, 1)
    dx = pt_x[0] - pt_ref[0]
    dy = pt_x[1] - pt_ref[1]
    theta_p = pt_ref[2]
    err_sl= ca.vertcat(-np.cos(theta_p) * dx - np.sin(theta_p) * dy, np.sin(theta_p) * dx - np.cos(theta_p) * dy, pt_x[2] - pt_ref[2])
    return ca.Function("f_err", [pt_x, pt_ref], [err_sl], ['pt_x]', 'pt_ref'], ['err_sl'])



def MPCC(start_time, initial_pose, obstacles):
    T = 0.2  # （模拟的）系统采样时间【秒】
    N = 20  # 需要预测的步长【超参数】
    v_max = 2.0  # 最大前向速度【物理约束】
    steering_max = np.pi/18  # 最大转动角速度 【物理约束】
# 根据数学模型建模
# 系统状态
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    phi = ca.SX.sym('phi')
    states = ca.vertcat(x, y, phi)
    n_states = states.size()[0]
# 控制输入
    v = ca.SX.sym('v')
    steering = ca.SX.sym('steering')
    controls = ca.vertcat(v, steering)
    n_controls = controls.size()[0]
# 运动学模型
    l = 0.2
    rhs = ca.vertcat(v*np.cos(phi), v*np.sin(phi), v * np.tan(steering) / l)
    f_X_dot = ca.Function('f_X_dot', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    horizon_inputs = ca.SX.sym('horizon_inputs', n_controls, N)
    horizon_status = ca.SX.sym('horizon_status', n_states, N+1)
    horizon_ref = ca.SX.sym('horizon_ref', n_states, N)
    initial_status = ca.SX.sym('initial_status', n_states, 1)
    obstacle_path = ca.SX.sym('dynamic_obj', 3, N * obstacles.shape[0])

    horizon_status[:, 0] = initial_status
    
    for i in range(N):
        f_value = f_X_dot(horizon_status[:, i], horizon_inputs[:, i])
        horizon_status[:, i+1] = horizon_status[:, i] + f_value*T
    next_horizon_status = ca.Function('next_horizon_status', [horizon_inputs, horizon_ref, initial_status], [horizon_status], ['input_U', 'target_state', 'real_status'], ['horizon_states'])
    f_err = def_func_err()

# NLP问题
    # 惩罚矩阵
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([[0.1, 0.0], [0.0, 0.1]])
    # 优化目标
    obj = 0  # 初始化优化目标值
    constraint = []
    for i in range(N):
        E = f_err(horizon_status[:, i+1], horizon_ref[:, i])
        obj = obj + ca.mtimes([E.T, Q, E]) + ca.mtimes([horizon_inputs[:, i].T, R, horizon_inputs[:, i]])

        dx = horizon_status[0, i] - obstacle_path[0, i]
        dy = horizon_status[1, i] - obstacle_path[1, i]
        constraint.append(dx * dx / 0.25 + dy * dy / 0.04)

        dx = horizon_status[0, i] - obstacle_path[0, N+i]
        dy = horizon_status[1, i] - obstacle_path[1, N+i]
        constraint.append(dx * dx / 0.25 + dy * dy / 0.04)

        dx = horizon_status[0, i] - obstacle_path[0, 2*N+i]
        dy = horizon_status[1, i] - obstacle_path[1, 2*N+i]
        constraint.append(dx * dx / 0.25 + dy * dy / 0.04)

    status_list = ca.vertcat(ca.reshape(horizon_ref, -1, 1), ca.reshape(initial_status, -1, 1), ca.reshape(obstacle_path, -1, 1))
    nlp_prob = {'f': obj, 'x': ca.reshape(horizon_inputs, -1, 1), 'p': status_list, 'g': ca.vertcat(*constraint)}
    # ipot设置
    opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
    # 最终目标，获得求解器
    cost_func = ca.nlpsol('cost_func', 'ipopt', nlp_prob, opts_setting)

    # 开始仿真
    # 定义约束条件，实际上CasADi需要在每次求解前更改约束条件。不过我们这里这些条件都是一成不变的
    # 因此我们定义在while循环外，以提高效率
    # 状态约束
#    lbg = -2.0
#    ubg = 2.0
    # 控制约束
    lbx = []  # 最低约束条件
    ubx = []  # 最高约束条件
    for _ in range(N):
        lbx.append(-v_max)
        ubx.append(v_max)
        lbx.append(-steering_max)
        ubx.append(steering_max)

    u0 = np.zeros((N, 2))  # 系统初始控制状态，为了统一本例中所有numpy有关

    start_simu_ = time.time()
    r_p = ref_path(start_time, N, T)
    c_p = np.concatenate((r_p, initial_pose, obstacles), axis=1)
    res = cost_func(x0=ca.reshape(u0, -1, 1), p=np.reshape(c_p.T, (-1, 1)), lbx=lbx, ubx=ubx, lbg=1)

    u_sol = ca.reshape(res['x'], n_controls, N)
    ff_value = next_horizon_status(u_sol, r_p, initial_pose)
    next_pose = shift_movement(T, initial_pose, u_sol, f_X_dot)
    next_pose = ca.reshape(next_pose, 3, 1)
    print(time.time() - start_simu_)
    return next_pose.full(), ff_value

def plot_ref():
    publisher = rospy.Publisher("casadi/ref", MarkerArray, queue_size=1000)
    msg_arr = MarkerArray()
    for t in np.arange(0, 20, 0.1):
        wpx, wpy, wptheta = ref_point(t)
        msg = Marker()
        msg.header.frame_id = "base_link"
        msg.type = Marker.ARROW
        msg.action = Marker.ADD
        msg.id = len(msg_arr.markers)
        msg.pose.position.x = wpx
        msg.pose.position.y = wpy

        msg.scale.x = 0.05
        msg.scale.y = 0.01
        msg.scale.z = 0.01
        msg.pose.orientation.w = np.cos(wptheta / 2.0)
        msg.pose.orientation.z = np.sin(wptheta / 2.0)

        msg.color.r = 1
        msg.color.a = 1

        msg.lifetime = rospy.Duration(5)

        msg_arr.markers.append(msg)
    publisher.publish(msg_arr)

def plot_arrows(status):
    publisher = rospy.Publisher("casadi/status", MarkerArray, queue_size=1000)
    msg_arr = MarkerArray()
    # for i in range(len(xx)):
    pos_x, pos_y, pos_theta = status
    msg = Marker()
    msg.header.frame_id = "base_link"
    msg.id = len(msg_arr.markers)
    msg.type = Marker.ARROW
    msg.action = Marker.ADD
    msg.pose.position.x = pos_x
    msg.pose.position.y = pos_y

    msg.pose.orientation.w = np.cos(pos_theta / 2.0)
    msg.pose.orientation.z = np.sin(pos_theta / 2.0)

    msg.scale.x = 0.5
    msg.scale.y = 0.2
    msg.scale.z = 0.2

    msg.color.b = 1
    msg.color.a = 0.5

    msg.lifetime = rospy.Duration(5)
    msg_arr.markers.append(msg)
    publisher.publish(msg_arr)

def plot_horizon(h_status):
    publisher = rospy.Publisher("casadi/horizon", MarkerArray, queue_size=1000)
    msg_arr = MarkerArray()
    msg = Marker()
    msg.header.frame_id = "base_link"
    msg.id = len(msg_arr.markers)
    msg.type = Marker.LINE_STRIP
    msg.action = Marker.ADD
    msg.scale.x = 0.001

    msg.color.g = 1
    msg.color.a = 1
    msg.lifetime = rospy.Duration(5)

    for i in range(h_status.shape[1]):
        p = Point()
        p.x = h_status[0, i]
        p.y = h_status[1, i]
        msg.points.append(p)

    msg_arr.markers.append(msg)
    publisher.publish(msg_arr)


def plot_obstacles(obstacle_paths):
    publisher = rospy.Publisher("casadi/obstacles", MarkerArray, queue_size=1000)
    msg_arr = MarkerArray()
    for i in range(obstacle_paths.shape[0]):
        # obs_path = obstacle_paths[i, :]
        # obs_path = obstacle_paths
    # for sp in range(5):
    #     j = int(sp * obstacle_paths.shape[1] / 5)
        for j in range(20):
            msg = Marker()
            msg.header.frame_id = "base_link"
            msg.id = len(msg_arr.markers)
            msg.type = Marker.SPHERE
            msg.action = Marker.ADD

            msg.scale.x = 0.5
            msg.scale.y = 0.2
            msg.scale.z = 0.2

            msg.pose.position.x = obstacle_paths[0][i * 20 + j]
            msg.pose.position.y = obstacle_paths[1][i * 20 + j]
            msg.pose.orientation.w = np.cos(obstacle_paths[2][i * 20 + j] / 2.0)
            msg.pose.orientation.z = np.sin(obstacle_paths[2][i * 20 + j] / 2.0)

            msg.color.g = 1
            msg.color.a = 0.5 - 0.03 * j

            msg.lifetime = rospy.Duration(5)
            msg_arr.markers.append(msg)
    publisher.publish(msg_arr)

def main():
    rospy.init_node("mpcc")
    loop_rate = rospy.Rate(10)

    status = np.array([0,0,0]).reshape(3, 1)
    time = 0
    while not rospy.is_shutdown():
        obstacle_paths = dynamic_obstacles(time)
        status, prediction = MPCC(time, status, obstacle_paths)
        time += 0.1

        plot_ref()
        plot_arrows(status)
        plot_horizon(prediction)
        plot_obstacles(obstacle_paths)
        loop_rate.sleep()

if __name__ == '__main__':
    main()
