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


def static_obstacles():
    obs = [()]
    return obs


def shift_movement(dT, x0, u, f):
    # 小车运动到下一个位置
    f_value = f(x0, u[:, 0])
    status = x0 + dT*f_value
    # 时间增加
    # t = t0 + T
    # 准备下一个估计的最优控制，因为u[:, 0]已经采纳，我们就简单地把后面的结果提前
    # u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return status

def def_func_err():
# 定义err转换func
    pt_ref = ca.SX.sym('pt_ref', 3, 1)
    pt_x = ca.SX.sym('pt_x]', 3, 1)
    err_sl = ca.SX.sym('err_sl', 3, 1)
    dx = pt_x[0] - pt_ref[0]
    dy = pt_x[1] - pt_ref[1]
    theta_p = pt_ref[2]
    err_sl= ca.vertcat(-np.cos(theta_p) * dx - np.sin(theta_p) * dy, np.sin(theta_p) * dx - np.cos(theta_p) * dy, pt_x[2] - pt_ref[2])
    return ca.Function("f_err", [pt_x, pt_ref], [err_sl], ['pt_x]', 'pt_ref'], ['err_sl'])



def MPCC(start_time, initial_pose):
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
    status_init = ca.SX.sym('status_init', n_states, 1)

    horizon_status[:, 0] = status_init
    
    for i in range(N):
        f_value = f_X_dot(horizon_status[:, i], horizon_inputs[:, i])
        horizon_status[:, i+1] = horizon_status[:, i] + f_value*T
    next_horizin_status = ca.Function('next_horizin_status', [horizon_inputs, horizon_ref, status_init], [horizon_status], ['input_U', 'target_state', 'read_status'], ['horizon_states'])
    f_err = def_func_err()

# NLP问题
    # 惩罚矩阵
    Q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.array([[0.1, 0.0], [0.0, 0.1]])
    # 优化目标
    obj = 0  # 初始化优化目标值
    for i in range(N):
        # 在N步内对获得优化目标表达式
        # horizon_error = horizon_status[:, i] - horizon_ref[:, i]
        E = f_err(horizon_status[:, i+1], horizon_ref[:, i])

        obj = obj + ca.mtimes([E.T, Q, E]) + ca.mtimes([horizon_inputs[:, i].T, R, horizon_inputs[:, i]])

    # 约束条件定义
#    g = []  # 用list来存储优化目标的向量
#    for i in range(N+1):
#        # 这里的约束条件只有小车的坐标（x,y）必须在-2至2之间
#        # 由于xy没有特异性，所以在这个例子中顺序不重要（但是在更多实例中，这个很重要）
#        g.append(horizon_status[0, i])
#        g.append(horizon_status[1, i])
    # 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为约束条件
    # 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
#    nlp_prob = {'f': obj, 'x': ca.reshape(horizon_inputs, -1, 1), 'p': horizon_ref, 'g': ca.vertcat(*g)}
    nlp_prob = {'f': obj, 'x': ca.reshape(horizon_inputs, -1, 1), 'p': ca.horzcat(horizon_ref, status_init)}
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
    c_p = ref_path(start_time, N, T)
    res = cost_func(x0=ca.reshape(u0, -1, 1), p=np.concatenate((c_p, initial_pose), axis=1), lbx=lbx, ubx=ubx)

    u_sol = ca.reshape(res['x'], n_controls, N)
    ff_value = next_horizin_status(u_sol, c_p, initial_pose)
    next_pose = shift_movement(T, initial_pose, u_sol, f_X_dot)
    next_pose = ca.reshape(next_pose, 3, 1)
    print(time.time() - start_simu_)
    return next_pose.full(), ff_value

def plot_ref():
    publisher = rospy.Publisher("casadi/ref", MarkerArray, queue_size=1000)
    msg_arr = MarkerArray()
    for t in np.arange(0, 10, 0.1):
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

def main():
    rospy.init_node("mpcc")
    loop_rate = rospy.Rate(10)

    status = np.array([0,0,0]).reshape(3, 1)
    time = 0
    while not rospy.is_shutdown():
        status, prediction = MPCC(time, status)
        time += 0.1

        plot_ref()
        plot_arrows(status)
        plot_horizon(prediction)
        loop_rate.sleep()

if __name__ == '__main__':
    main()
