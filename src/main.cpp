#include "mpcc/main.h"

using casadi::DM;
using casadi::Function;
using casadi::MX;
using casadi::SX;
using Matrix = casadi::Matrix<double>;
using casadi::Dict;
using casadi::DMDict;
using casadi::Slice;
using casadi::SXDict;

/* ref:[3*1] pt:[3*1] -> err_sl:[3*1] */
Function def_point_err() {
  SX pt_x = SX::sym("pt_x", n_pos, 1);
  SX pt_ref = SX::sym("pt_ref", n_pos, 1);
  SX dx = pt_x(0) - pt_ref(0);
  SX dy = pt_x(1) - pt_ref(1);
  SX theta_p = pt_ref(2);

  std::vector<SX> input;
  input.emplace_back(pt_x);
  input.emplace_back(pt_ref);

  std::vector<SX> output_cell;
  output_cell.emplace_back(-SX::cos(theta_p) * dx - SX::sin(theta_p) * dy);
  output_cell.emplace_back(SX::sin(theta_p) * dx - SX::cos(theta_p) * dy);
  output_cell.emplace_back(pt_x(2) - pt_ref(2));
  std::vector<SX> output;
  output.emplace_back(SX::vertcat(output_cell));

  return Function("f_sl_err", input, output,
                  std::vector<std::string>{"pt_x", "pt_ref"},
                  std::vector<std::string>{"err_sl"});
}

/* input:[5*1] -> output[3*1] */
Function def_state_trans() {
  SX x = SX::sym("x");
  SX y = SX::sym("y");
  SX phi = SX::sym("phi");
  SX v = SX::sym("v");
  SX steering = SX::sym("steering");

  std::vector<SX> input_cell = std::vector<SX>{x, y, phi, v, steering};
  std::vector<SX> input = std::vector<SX>{SX::vertcat(input_cell)};

  std::vector<SX> output_cell;
  output_cell.emplace_back(x + v * dT * SX::cos(phi));
  output_cell.emplace_back(y + v * dT * SX::sin(phi));
  output_cell.emplace_back(phi + v * dT * SX::tan(steering) / l);
  std::vector<SX> output = std::vector<SX>{SX::vertcat(output_cell)};

  return Function("state_trans", input, output,
                  std::vector<std::string>{"current_state"},
                  std::vector<std::string>{"next_state"});
}

DMDict MPCC(std::vector<std::vector<double>>& initial_guess,
            std::vector<std::vector<double>>& ref_path,
            std::vector<std::vector<obstacle>>& obstacle_list) {
  double l = 0.2;
  int obs_num = obstacle_list.size();

  SX horizon_status = SX::vertcat(
      std::vector<SX>{SX::sym("s", n_pos, N), SX::sym("c", n_control, N)});
  SX initial_status = SX::sym("i", n_pos, 1);
  SX horizon_ref = SX::sym("r", n_pos, N);
  SX obstacle_path = SX::sym("o", obs_num * (n_pos + n_sigma), N);

  // std::cout << "\nhorizon_status" << horizon_status << std::endl;
  // std::cout << "\ninitial_status" << initial_status << std::endl;
  // std::cout << "\nhorizon_ref" << horizon_ref << std::endl;
  // std::cout << "\nobstacle_path\n" << obstacle_path << std::endl;

  std::vector<double> q({5, 0, 0, 0, 5, 0, 0, 0, 1});
  Matrix Q = Matrix::reshape(Matrix(q), 3, 3);

  std::vector<double> r({0.1, 0, 0, 0.1});
  Matrix R = Matrix::reshape(Matrix(r), 2, 2);

  Function f_sl_err_3_3_to_3 = def_point_err();
  Function state_trans_5_to_3 = def_state_trans();

  SX obj = SX::zeros(1, 1);
  std::vector<SX> constraints;
  SX next_state = initial_status;
  SX last_input = Matrix(n_control, 1);

  for (int i = 0; i < N; i++) {
    int line_start = i * (n_pos + n_control);
    SX pt_x = horizon_status(Slice(line_start, line_start + n_pos, 1));
    SX pt_ref = horizon_ref(Slice(i * n_pos, (i + 1) * n_pos, 1));
    // std::cout << "\npt_x" << pt_x << std::endl;
    // std::cout << "\npt_ref" << pt_ref << std::endl;

    SX E = f_sl_err_3_3_to_3({SXDict{{"pt_x", pt_x}, {"pt_ref", pt_ref}}})
               .at("err_sl");
    // std::cout << "\nE=" << E << std::endl;

    SX cur_input = horizon_status(
        Slice(line_start + n_pos, line_start + n_pos + n_control));
    // std::cout << "\nlast_input" << last_input << std::endl;
    // std::cout << "\ncur_input=" << cur_input << std::endl;

    SX dU = last_input - cur_input;
    // std::cout << "\ndU=" << dU << std::endl;

    obj += SX::mtimes(std::vector<SX>{E.T(), Q, E}) +
           SX::mtimes(std::vector<SX>{dU.T(), R, dU});

    SX obstales_at_time_i =
        obstacle_path(Slice(i * obs_num * (n_pos + n_sigma),
                            (i + 1) * obs_num * (n_pos + n_sigma)));
    // std::cout << "\nobstales_at_time_i\n" << obstales_at_time_i << std::endl;

    for (int obs_id = 0; obs_id < obs_num; obs_id++) {
      /* obstacle ellipse constraint */
      SX obs = obstales_at_time_i(
          Slice(obs_id * (n_pos + n_sigma), (obs_id + 1) * (n_pos + n_sigma)));

      std::cout << "\nobs\n" << obs << std::endl;
      SX dx = (horizon_status(0, i) - obs(0)) / obs(n_pos + 0);
      SX dy = (horizon_status(1, i) - obs(1)) / obs(n_pos + 1);
      constraints.emplace_back(dx * dx + dy * dy);
    }

    for (int j = 0; j < n_pos; j++) {
      /* <v,steering> constraint for adjacent pose states */
      constraints.emplace_back(horizon_status(j, i) - next_state(j));
    }

    SXDict state_trans_input;
    state_trans_input["current_state"] =
        horizon_status(Slice(line_start, line_start + n_pos + n_control));
    // std::cout << "\nstate_trans_input[current_state]\n" <<
    // state_trans_input["current_state"] << std::endl;

    next_state = state_trans_5_to_3(state_trans_input).at("next_state");
    last_input = horizon_status(
        Slice(line_start + n_pos, line_start + n_pos + n_control));
  }

  SXDict nlp_prob;
  std::vector<SX> tmp_p;
  tmp_p.emplace_back(SX::reshape(horizon_ref, -1, 1));
  tmp_p.emplace_back(SX::reshape(initial_status, -1, 1));
  tmp_p.emplace_back(SX::reshape(obstacle_path, -1, 1));
  // std::cout << "ref\n" << horizon_ref << std::endl;
  // std::cout << "tmp_p\n" << tmp_p << std::endl;

  nlp_prob["x"] = SX::reshape(horizon_status, -1, 1);  // decision vars
  nlp_prob["p"] = SX::vertcat(tmp_p);
  nlp_prob["f"] = obj;                       // objective
  nlp_prob["g"] = SX::vertcat(constraints);  // constraints
  // std::cout << "\nnlp_prob[x]\n" << nlp_prob["x"] << std::endl;
  // std::cout << "\nnlp_prob[p]\n" << nlp_prob["p"] << std::endl;
  // std::cout << "\nnlp_prob[g]\n" << nlp_prob["g"] << std::endl;

  Dict opts_setting;
  opts_setting["ipopt.max_iter"] = 500;
  opts_setting["ipopt.print_level"] = 0;
  opts_setting["print_time"] = 0;
  opts_setting["ipopt.acceptable_tol"] = 1e-5;
  opts_setting["ipopt.acceptable_obj_change_tol"] = 1e-6;

  Function cost_func = nlpsol("F", "ipopt", nlp_prob, opts_setting);

  std::vector<DM> lbx, ubx, lbg, ubg;
  for (int i = 0; i < N; i++) {
    ubx.emplace_back(std::numeric_limits<double>::max());
    ubx.emplace_back(std::numeric_limits<double>::max());
    ubx.emplace_back(std::numeric_limits<double>::max());
    lbx.emplace_back(-std::numeric_limits<double>::max());
    lbx.emplace_back(-std::numeric_limits<double>::max());
    lbx.emplace_back(-std::numeric_limits<double>::max());

    ubx.emplace_back(v_max);
    ubx.emplace_back(steering_max);
    lbx.emplace_back(-v_max);
    lbx.emplace_back(-steering_max);
  }

  for (int i = 0; i < N; i++) {
    for (int obs_id = 0; obs_id < obs_num; obs_id++) {
      lbg.emplace_back(1);
      ubg.emplace_back(std::numeric_limits<double>::max());
    }

    for (int _ = 0; _ < n_pos; _++) {
      lbg.emplace_back(0);
      ubg.emplace_back(0);
    }
  }

  DM dm_obstacle_list(obs_num * (n_pos + n_sigma), N);
  for (int i = 0; i < obstacle_list.size(); i++) {
    for (int j = 0; j < N; j++) {
      dm_obstacle_list(i * (n_pos + n_sigma) + 0, j) = obstacle_list[i][j].x;
      dm_obstacle_list(i * (n_pos + n_sigma) + 1, j) = obstacle_list[i][j].y;
      dm_obstacle_list(i * (n_pos + n_sigma) + 2, j) = obstacle_list[i][j].phi;
      dm_obstacle_list(i * (n_pos + n_sigma) + 3, j) =
          obstacle_list[i][j].sigma_x;
      dm_obstacle_list(i * (n_pos + n_sigma) + 4, j) =
          obstacle_list[i][j].sigma_y;
    }
  }
  // std::cout << "\ndm_obstacle_list\n" << dm_obstacle_list << std::endl;

  DM p = DM::vertcat(std::vector<DM>{DM::reshape(DM(ref_path).T(), -1, 1),
                                     DM(initial_guess).T()(Slice(0, 3)),
                                     DM::reshape(dm_obstacle_list, -1, 1)});

  DMDict guess;
  guess["x0"] = DM::reshape(DM(initial_guess).T(), -1, 1);
  // std::cout << "\ninitial_guess" << DM(initial_guess).T() << std::endl;
  // std::cout << "\nx0\n" << guess["x0"] << std::endl;

  // std::cout << "\nref_path " << ref_path << std::endl;
  // std::cout << "\nDM::reshape(DM(ref_path) " << DM(ref_path).T() << std::endl;
  // std::cout << "\nreshape\n"
  //           << DM::reshape(DM(ref_path).T(), -1, 1) << std::endl;
  // std::cout << "\ninitial_guess\n" << DM(initial_guess).T() << std::endl;
  // std::cout << "\ninitial\n" << DM(initial_guess).T()(Slice(0, 3)) << std::endl;
  // std::cout << "\np\n" << p << std::endl;
  // std::cout << "\nubx\n" << ubx << std::endl;
  // std::cout << "\nlbx\n" << lbx << std::endl;
  // std::cout << "\nubg\n" << ubg << std::endl;
  // std::cout << "\nlbg\n" << lbg << std::endl;

  guess["p"] = p;
  guess["ubx"] = ubx;
  guess["lbx"] = lbx;
  guess["ubg"] = ubg;
  guess["lbg"] = lbg;
  return cost_func(guess);
}

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "mpcc_node");
  ros::NodeHandle nh;
  ros::Rate rate(10);
  ros::Publisher pub_obs =
      nh.advertise<visualization_msgs::MarkerArray>("casadi/obstacles", 1);
  ros::Publisher pub_ref =
      nh.advertise<visualization_msgs::MarkerArray>("casadi/ref", 1);
  ros::Publisher pub_status =
      nh.advertise<visualization_msgs::MarkerArray>("casadi/status", 1);

  Function state_trans_5_to_3 = def_state_trans();
  double start_time = 0;
  std::vector<std::vector<double>> initial_guess(
      N, std::vector<double>(n_pos + n_control, 0));

  while (ros::ok()) {
    auto start = std::chrono::system_clock::now();
    std::vector<std::vector<obstacle>> obstacle_paths =
        GetObstacles(start_time);

    std::vector<std::vector<double>> ref_path = GetRefPath(start_time);

    DMDict res = MPCC(initial_guess, ref_path, obstacle_paths);

    pub_obs.publish(plot_obstacles(obstacle_paths));
    pub_ref.publish(plot_ref(ref_path));
    pub_status.publish(plot_horizon(initial_guess));

    // DM next_state =
    //     state_trans_5_to_3({DMDict{{"current_state", res["x"](Slice(0,
    //     5))}}})
    //         .at("next_state");
    // std::cout << "\nres[x]\n" << DM::reshape(res["x"], 5, -1) << std::endl;

    // for (int i = 0; i < N; i++) {
    //   initial_guess[i][0] = double(next_state(0));
    //   initial_guess[i][1] = double(next_state(1));
    //   initial_guess[i][2] = double(next_state(2));
    // }

    for (int i = 0; i < N - 1; i++) {
      initial_guess[i][0] = double(res["x"]((i + 1) * (n_pos + n_control) + 0));
      initial_guess[i][1] = double(res["x"]((i + 1) * (n_pos + n_control) + 1));
      initial_guess[i][2] = double(res["x"]((i + 1) * (n_pos + n_control) + 2));
    }
    initial_guess[N - 1][0] =
        double(res["x"]((N - 1) * (n_pos + n_control) + 0));
    initial_guess[N - 1][1] =
        double(res["x"]((N - 1) * (n_pos + n_control) + 1));
    initial_guess[N - 1][2] =
        double(res["x"]((N - 1) * (n_pos + n_control) + 2));

    start_time += dT / 2;
    auto end = std::chrono::system_clock::now();
    double duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "duration " << duration << std::endl;
    rate.sleep();
  }
}