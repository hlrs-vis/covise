#ifndef _DATE_H
#define _DATE_H

#include <chrono>
#include <string>

namespace ennovatis {
struct date {
    static constexpr auto dateformat = "%d.%m.%Y";

    /**
     * @brief Converts a datetime string to a std::chrono::system_clock::time_point object.
     * Format need to be in the form of "%d.%m.%Y".
     * source: https://www.geeksforgeeks.org/date-and-time-parsing-in-cpp/
     * 
     * @param datetimeString The datetime string to be converted.
     * @param format The format of the datetime string.
     * @return The converted std::chrono::system_clock::time_point object.
     */
    [[nodiscard]] static std::chrono::system_clock::time_point str_to_time_point(const std::string &datetimeString,
                                                                                 const char *format);

    /**
     * @brief Returns a formatted string representation of the given time point.
     * source: https://www.geeksforgeeks.org/date-and-time-parsing-in-cpp/
     * 
     * @param timePoint The time point to format.
     * @param format The format string specifying the desired format.
     * @return The formatted string representation of the time point.
     */
    [[nodiscard]] static std::string time_point_to_str(const std::chrono::system_clock::time_point &timePoint,
                                                       const char *format);
};
} // namespace ennovatis
#endif