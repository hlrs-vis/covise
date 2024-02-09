//TODO: Maybe exclude as own lib for whole covise?
#include "REST.h"
#include <curl/curl.h>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>

using namespace std;

namespace {

/**
 * @brief Callback function to handle curl's response
 * 
 * @param contents Pointer to the content of data
 * @param size Size of each data element
 * @param nmemb Number of data elements
 * @param userp Pointer to the user-defined data
 * @return size_t The total size of the response data
 */
size_t writeCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}
} // namespace

namespace ennovatis {

std::string RESTRequest::operator()()
{
    auto _dtf = time_point_to_str(dtf, dateformat);
    auto _dtt = time_point_to_str(dtt, dateformat);
    return url + "?projEid=" + projEid + "&dtf=" + _dtf + "&dtt=" + _dtt + "&ts=" + std::to_string(ts) +
           "&tsp=" + std::to_string(tsp) + "&tst=" + std::to_string(tst) + "&etst=" + std::to_string(etst) +
           "&cEid=" + channelId;
}

std::chrono::system_clock::time_point str_to_time_point(const std::string &datetimeString, const std::string &format)
{
    tm tmStruct = {};
    std::istringstream ss(datetimeString);
    ss >> std::get_time(&tmStruct, format.c_str());
    return std::chrono::system_clock::from_time_t(mktime(&tmStruct));
}

std::string time_point_to_str(const std::chrono::system_clock::time_point &timePoint, const std::string &format)
{
    time_t time = std::chrono::system_clock::to_time_t(timePoint);
    tm *timeinfo = localtime(&time);
    char buffer[70];
    strftime(buffer, sizeof(buffer), format.c_str(), timeinfo);
    return buffer;
}

bool performCurlRequest(const string &url, string &response)
{
    auto curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    auto res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        return false;
    }

    return true;
}

void cleanupcurl()
{
    curl_global_cleanup();
}
} // namespace ennovatis