#pragma once
#include <vector>
#include <map>
#include <string>
#include <variant>

namespace core::simulation {
typedef size_t Timestep;
typedef std::string Unit;
typedef std::string Species;
typedef double Scalar;
typedef std::vector<Scalar> ScalarVec;
typedef const ScalarVec *const_ScalarVecs;
typedef std::map<std::string, ScalarVec> ScalarMap;
typedef std::variant<const_ScalarVecs, std::string> ScalarByNameCollectorResult;
}
