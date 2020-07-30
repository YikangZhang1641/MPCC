#include <casadi/casadi.hpp>
using namespace casadi;

int main() {
  // Symbols/expressions
  MX x = MX::sym("x");
  MX y = MX::sym("y");
  MX z = MX::sym("z");
  MX f = pow(x, 2) + 100 * pow(z, 2);
  MX g = z + pow(1 - x, 2) - y;

  MXDict nlp;                   // NLP declaration
  nlp["x"] = vertcat(x, y, z);  // decision vars
  nlp["f"] = f;                 // objective
  nlp["g"] = g;                 // constraints

  // Create solver instance
  Function F = nlpsol("F", "ipopt", nlp);

  // Solve the problem using a guess
  F(DMDict{{"x0", DM({2.5, 3.0, 0.75})}, {"ubg", 0}, {"lbg", 0}});
}