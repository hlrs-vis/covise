/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _REST_H
#define _REST_H

#include <curl/curl.h>
#include <string>
#include <chrono>

namespace ennovatis {

constexpr auto dateformat("%d.%m.%Y");

/**
 * @brief Converts a datetime string to a std::chrono::system_clock::time_point object.
 * Format need to be in the form of "%d.%m.%Y".
 * source: https://www.geeksforgeeks.org/date-and-time-parsing-in-cpp/
 * 
 * @param datetimeString The datetime string to be converted.
 * @param format The format of the datetime string.
 * @return The converted std::chrono::system_clock::time_point object.
 */
std::chrono::system_clock::time_point str_to_time_point(const std::string &datetimeString, const std::string &format);

/**
 * @brief Returns a formatted string representation of the given time point.
 * source: https://www.geeksforgeeks.org/date-and-time-parsing-in-cpp/
 * 
 * @param timePoint The time point to format.
 * @param format The format string specifying the desired format.
 * @return The formatted string representation of the time point.
 */
std::string time_point_to_str(const std::chrono::system_clock::time_point &timePoint, const std::string &format);

struct RESTRequest {
    std::string url; // URL
    std::string projEid; // project ID
    std::string channelId;
    std::chrono::system_clock::time_point dtf; // from
    std::chrono::system_clock::time_point dtt; // until
    int ts = 86400; // 1 day resolution
    int tsp = 0;
    int tst = 1;
    int etst = 1024;

    /**
     * @brief This function generates a string representation.
     * 
     * @return std::string The string returned by the operator.
     */
    std::string operator()() const;
};

/**
 * @brief Function to perform a CURL request
 * @Source: https://stackoverflow.com/a/51319043
 * 
 * @param url The URL to send the request to
 * @param response The response data received from the request (storage)
 * @return bool True if the request was successful, false otherwise
 */
bool performCurlRequest(const std::string &url, std::string &response);

/**
 * @brief Function to cleanup the CURL library (need to be called once for each application)
 */
void cleanupcurl();

} // namespace ennovatis

#endif
