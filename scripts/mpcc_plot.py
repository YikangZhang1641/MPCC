import rospy
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np

dT = 0.2  # （模拟的）系统采样时间【秒】
N = 10  # 需要预测的步长【超参数】
v_max = 3.0  # 最大前向速度【物理约束】
steering_max = np.pi / 18  # 最大转动角速度 【物理约束】
l = 0.2


def ref_path(start):
    arr = np.array([]).reshape(3, -1)
    for i in range(N):
        arr = np.concatenate((arr, ref_point(start + i * dT)), axis = 1)
    # print(arr)
    return arr


def ref_point(t):
    x = t
    y = 0.5 * np.sin(t)
    theta = np.arctan(0.5 * np.cos(t))
    return np.array([x, y, theta]).reshape(3, -1)

def dynamic_obstacles(start):
    obs0 = []
    obs1 = []
    obs2 = []
    obs3 = []
    obs4 = []
    obs5 = []

    for i in range(N):
        t = start + i * dT

        x0 = 10 - 0.6 * t
        # x0 = -10
        y0 = -0.3
        theta0 = np.pi
        obs0.append([x0, y0, theta0])

        x1 = 15 - t
        # x1 = -10
        y1 = 0.5
        theta1 = np.pi
        obs1.append([x1, y1, theta1])

        x2 = 5 - 1 * t
        # x2 = -10
        y2 = 0.2
        theta2 = np.pi
        obs2.append([x2, y2, theta2])

        x3 = 5 + 0.6 * t
        y3 = - 0.6
        theta3 = 0
        obs3.append([x3, y3, theta3])

        x4 = -10 + 2 * t
        y4 = -0.4
        theta4 = 0
        obs4.append([x4, y4, theta4])

        x5 = 10 + 0.2 * t
        y5 = 0.3
        theta5 = 0
        obs5.append([x5, y5, theta5])
    res = np.concatenate([obs0, obs1, obs2, obs3, obs4, obs5]).T
    return res

def plot_ref(start_time):
    publisher = rospy.Publisher("casadi/ref", MarkerArray, queue_size=1000)
    msg_arr = MarkerArray()
    for t in np.arange(0, 4, 0.2):
        wpx, wpy, wptheta = ref_point(t + start_time)
        msg = Marker()
        msg.header.frame_id = "base_link"
        msg.type = Marker.ARROW
        msg.action = Marker.ADD
        msg.id = len(msg_arr.markers)
        msg.pose.position.x = wpx
        msg.pose.position.y = wpy

        msg.scale.x = 0.05
        msg.scale.y = 0.02
        msg.scale.z = 0.02
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
    pos_x, pos_y, pos_theta = status[:3]
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
    msg.scale.x = 0.01

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
    for i in range(int(obstacle_paths.shape[1] / N)):
        # obs_path = obstacle_paths[i, :]
        # obs_path = obstacle_paths
    # for sp in range(5):
    #     j = int(sp * obstacle_paths.shape[1] / 5)
        for j in range(N):
            msg = Marker()
            msg.header.frame_id = "base_link"
            msg.id = len(msg_arr.markers)
            msg.type = Marker.SPHERE
            msg.action = Marker.ADD

            msg.scale.x = 0.5
            msg.scale.y = 0.2
            msg.scale.z = 0.2

            msg.pose.position.x = obstacle_paths[0][i * N + j]
            msg.pose.position.y = obstacle_paths[1][i * N + j]
            msg.pose.orientation.w = np.cos(obstacle_paths[2][i * N + j] / 2.0)
            msg.pose.orientation.z = np.sin(obstacle_paths[2][i * N + j] / 2.0)

            msg.color.g = 1
            msg.color.a = 0.5 - 0.03 * j

            msg.lifetime = rospy.Duration(5)
            msg_arr.markers.append(msg)
    publisher.publish(msg_arr)

