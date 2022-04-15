#include <casadi/casadi.hpp>
#include <iostream>
#include <chrono>

#include "geometry_msgs/Point.h"
#include "geometry_msgs/PoseStamped.h"
#include "ros/ros.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#define PI 3.141

double dT = 0.2;
double N = 20;
double v_max = 3.0;
double steering_max = PI / 18;
double l = 1;
int n_pos = 3;
int n_control = 2;
int n_sigma = 2;

class obstacle {
 public:
  obstacle(double x_, double y_, double phi_) : x(x_), y(y_), phi(phi_) {}

  void SetVT(double v_, double dT_) { v = v_, dT = dT_; }

  void SetUncertainty(double a, double b) {
    sigma_x = std::min(a, MAX_SIGMA_X);
    sigma_y = std::min(b, MAX_SIGMA_Y);
  }
  double x, y, phi;
  double v, dT;
  double sigma_x, sigma_y;
  double MAX_SIGMA_X = 1.0, MAX_SIGMA_Y = 0.6;
};

void GenerateMovement(std::vector<std::vector<obstacle>>& obs_list, double v,
                      double start_time, double start_x, double start_y,
                      double theta) {
  std::vector<obstacle> obs_vec;
  for (int i = 0; i < N; i++) {
    double t = start_time + i * dT;

    double x = start_x + v * t * std::cos(theta);
    double y = start_y + v * t * std::sin(theta);
    obstacle ob(x, y, theta);
    if (obs_vec.size() == 0) {
      ob.SetUncertainty(0.5, 0.2);
    } else {
      obstacle last = obs_vec[obs_vec.size() - 1];
      ob.SetUncertainty(last.sigma_x + 0.03, last.sigma_y + 0.03);
    }
    obs_vec.emplace_back(ob);
  }
  obs_list.emplace_back(obs_vec);
}

std::vector<std::vector<obstacle>> GetObstacles(double start_time) {
  std::vector<std::vector<obstacle>> obs_list;
  GenerateMovement(obs_list, 1.5, start_time, 10, 0, -PI);
  GenerateMovement(obs_list, 0.8, start_time, 18, -0.5, -PI);
  GenerateMovement(obs_list, 1.0, start_time, 12, -3, PI *5 / 6);
  GenerateMovement(obs_list, 0.4, start_time, 2, 0.2, 0);

  GenerateMovement(obs_list, 0.9, start_time, 2, -3, PI/10);
  GenerateMovement(obs_list, 0.5, start_time, 6, 7, -PI/2);
  GenerateMovement(obs_list, 1.3, start_time, 3, 4,  -PI / 3.0);
  GenerateMovement(obs_list, 0.6, start_time, 4, -0.3, 0);

  GenerateMovement(obs_list, 1.3, start_time, 3, 0, 0);

  return obs_list;
}

std::vector<std::vector<double>> GetRefPath(double start_time) {
  std::vector<std::vector<double>> path;
  for (int i = 0; i < N; i++) {
    double t = start_time + i * dT;
    std::vector<double> pt;
    pt.emplace_back(0 + t * 1.5);
    pt.emplace_back(0 + 0.5*std::sin(t*1.5));
    pt.emplace_back(0);
    path.emplace_back(pt);
  }
  return path;
}

visualization_msgs::MarkerArray plot_ref(
    std::vector<std::vector<double>>& ref_path) {
  visualization_msgs::MarkerArray marker_array;
  for (int i = 0; i < ref_path.size(); i++) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "/base_link";
    marker.header.stamp = ros::Time::now();
    marker.id = marker_array.markers.size();
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;

    marker.scale.x = 0.1;
    marker.scale.y = 0.03;
    marker.scale.z = 0.03;

    marker.pose.position.x = ref_path[i][0];
    marker.pose.position.y = ref_path[i][1];

    marker.pose.orientation.w = std::cos(ref_path[i][2] / 2.0);
    marker.pose.orientation.z = std::sin(ref_path[i][2] / 2.0);

    marker.color.r = 1;
    marker.color.a = 1;
    marker.lifetime = ros::Duration(5);
    marker_array.markers.emplace_back(marker);
  }
  return marker_array;
}
visualization_msgs::MarkerArray plot_obstacles(
    std::vector<std::vector<obstacle>>& obstacle_paths) {
  visualization_msgs::MarkerArray marker_array;
  for (int obs_id = 0; obs_id < obstacle_paths.size(); obs_id++) {
    for (int t = 0; t < obstacle_paths[0].size(); t++) {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "/base_link";
      marker.header.stamp = ros::Time::now();
      marker.id = marker_array.markers.size();
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;

      marker.scale.x = obstacle_paths[obs_id][t].sigma_x;
      marker.scale.y = obstacle_paths[obs_id][t].sigma_y;
      marker.scale.z = obstacle_paths[obs_id][t].sigma_y;

      marker.pose.position.x = obstacle_paths[obs_id][t].x;
      marker.pose.position.y = obstacle_paths[obs_id][t].y;

      marker.pose.orientation.w = std::cos(obstacle_paths[obs_id][t].phi / 2.0);
      marker.pose.orientation.z =
          std::sin(obstacle_paths[obs_id][t].phi / 2.0);

      marker.color.g = 1;
      marker.color.a = 0.5 - 0.4 / N * t;
      marker.lifetime = ros::Duration(5);
      marker_array.markers.emplace_back(marker);
    }
  }
  // std::cout << "obs size: " << marker_array.markers.size() << std::endl;
  return marker_array;
}

visualization_msgs::MarkerArray plot_horizon(
    std::vector<std::vector<double>>& initial_guess) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  marker.header.frame_id = "/base_link";
  marker.header.stamp = ros::Time::now();
  marker.id = marker_array.markers.size();
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;

  marker.scale.x = 0.1;

  for (int i = 0; i < N; i++) {
    geometry_msgs::Point p;
    p.x = initial_guess[i][0];
    p.y = initial_guess[i][1];
    marker.points.emplace_back(p);
  }
  // marker.pose.position.x = initial_guess[i * (n_pos + n_control) + 0];
  // marker.pose.position.y = initial_guess[i * (n_pos + n_control) + 1];

  // marker.pose.orientation.z =
  //     std::cos(initial_guess[i * (n_pos + n_control) + 2] / 2.0);
  // marker.pose.orientation.w =
  //     -std::sin(initial_guess[i * (n_pos + n_control) + 2] / 2.0);

  marker.color.b = 1;
  marker.color.a = 1;
  marker.lifetime = ros::Duration(5);
  marker_array.markers.emplace_back(marker);
  // }

  visualization_msgs::Marker arrow;
  arrow.header.frame_id = "/base_link";
  arrow.header.stamp = ros::Time::now();
  arrow.id = marker_array.markers.size();
  arrow.type = visualization_msgs::Marker::ARROW;
  arrow.action = visualization_msgs::Marker::ADD;

  arrow.pose.position.x = initial_guess[0][0];
  arrow.pose.position.y = initial_guess[0][1];

  arrow.pose.orientation.w = std::cos(initial_guess[0][2] / 2.0);
  arrow.pose.orientation.z = std::sin(initial_guess[0][2] / 2.0);

  arrow.scale.x = 0.5;
  arrow.scale.y = 0.2;
  arrow.scale.z = 0.2;

  arrow.color.b = 1;
  arrow.color.a = 0.5;
  arrow.lifetime = ros::Duration(5);
  marker_array.markers.emplace_back(arrow);
  return marker_array;
}
