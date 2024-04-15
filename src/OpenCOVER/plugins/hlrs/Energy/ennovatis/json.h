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
    std::unique_ptr<json_response_obj> operator()(const nlohmann::json &j) const
    {
        return std::make_unique<json_response_obj>(j.template get<ennovatis::json_response_obj>());
    }

    /**
     * @brief This function is an operator overload that takes a string as input and returns a unique pointer to a json_response_obj.
     * 
     * @param s The string input.
     * @return std::unique_ptr<json_response_obj> A unique pointer to a json_response_obj if parsing went well otherwise a nullptr.
     */
    std::unique_ptr<json_response_obj> operator()(const std::string &s) const;
};
} // namespace ennovatis
#endif