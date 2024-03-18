#ifndef _ENERGY_ENNOVATIS_JSON_H
#define _ENERGY_ENNOVATIS_JSON_H

#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace ennovatis {
struct json_response_obj {
    int Average;
    std::string MaxTime;
    int MaxValue;
    std::string MinTime;
    int MinValue;
    int StandardDeviation;
    std::vector<std::string> Times;
    std::vector<int> Values;
    operator std::string() const;
};

void from_json(const nlohmann::json &j, json_response_obj &obj);

struct json_parser {
    std::unique_ptr<json_response_obj> operator()(const nlohmann::json &j)
    {
        return std::make_unique<json_response_obj>(j.template get<ennovatis::json_response_obj>());
    }
    
    std::unique_ptr<json_response_obj> operator()(const std::string &s)
    {
        return operator()(nlohmann::json::parse(s));
    }
};
} // namespace ennovatis
#endif