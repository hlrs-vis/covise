#pragma once
#include <vector>
#include <map>
#include <string>

namespace prototype::core::simulation {
typedef size_t Timestep;
typedef std::string Unit;
typedef std::string Species;
typedef double Scalar;
typedef std::vector<Scalar> ScalarVec;
typedef std::map<std::string, ScalarVec> ScalarMap;
}
