#include "json.h"
#include <sstream>

using json = nlohmann::json;

namespace ennovatis {

json_response_obj::operator std::string() const
{
    std::ostringstream oss;
    oss << "Average: " << std::to_string(Average) << "\n"
        << "MaxTime: " << MaxTime << "\n"
        << "MaxValue: " << std::to_string(MaxValue) << "\n"
        << "MinTime: " << MinTime << "\n"
        << "MinValue: " << std::to_string(MinValue) << "\n"
        << "StandardDeviation: " << std::to_string(StandardDeviation) << "\n";
    oss << "Times: "
        << "\n";
    for (auto &time: Times)
        oss << time << "\n";
    oss << "Values: "
        << "\n";
    for (auto &value: Values)
        oss << std::to_string(value) << "\n";
    return oss.str();
}

void from_json(const nlohmann::json &j, json_response_obj &obj)
{
    j.at("Average").get_to(obj.Average);
    j.at("MaxTime").get_to(obj.MaxTime);
    j.at("MaxValue").get_to(obj.MaxValue);
    j.at("MinTime").get_to(obj.MinTime);
    j.at("MinValue").get_to(obj.MinValue);
    j.at("StandardDeviation").get_to(obj.StandardDeviation);
    j.at("Times").get_to(obj.Times);
    j.at("Values").get_to(obj.Values);
}

std::unique_ptr<json_response_obj> json_parser::operator()(const std::string &s) const
{
    try {
        return operator()(nlohmann::json::parse(s));
    } catch (json::parse_error &e) {
        return nullptr;
    }
}
} // namespace ennovatis