#include "date.h"
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

using namespace std;

namespace ennovatis {

std::chrono::system_clock::time_point date::str_to_time_point(const string &datetimeString, const char *format)
{
    tm tmStruct = {};
    istringstream ss(datetimeString);
    ss >> get_time(&tmStruct, format);
    return chrono::system_clock::from_time_t(mktime(&tmStruct));
}

std::string date::time_point_to_str(const chrono::system_clock::time_point &timePoint, const char *format)
{
    time_t time = chrono::system_clock::to_time_t(timePoint);
    tm *timeinfo = localtime(&time);
    char buffer[70];
    strftime(buffer, sizeof(buffer), format, timeinfo);
    return buffer;
}
} // namespace ennovatis