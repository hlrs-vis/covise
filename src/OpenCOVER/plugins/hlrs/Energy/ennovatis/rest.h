#ifndef _REST_H
#define _REST_H

#include <curl/curl.h>
#include <string>
#include <chrono>

namespace ennovatis {

/**
 * @brief The RESTRequest struct represents a REST request object.
 * 
 * It contains the necessary information to make a REST request, such as the URL, project ID, channel ID,
 * time range, and resolution.
 */
struct RESTRequest {
    std::string url; // URL
    std::string projEid; // project ID
    std::string channelId; // channel ID
    std::chrono::system_clock::time_point dtf; // from
    std::chrono::system_clock::time_point dtt; // until
    int ts = 86400; // 1 day resolution
    int tsp = 0;
    int tst = 1;
    int etst = 1024;

    /**
     * @brief This function generates a string representation of the RESTRequest object.
     * 
     * @return std::string The string representation of the RESTRequest object.
     */
    [[nodiscard("String representation not used.")]] std::string operator()() const;
};

struct rest {
    /**
     * @brief Fetches Ennovatis data using a REST request.
     * 
     * This function sends a REST request to the specified URL using the provided RESTRequest object and
     * returns the response data as a string.
     * 
     * @param req The RESTRequest object containing the necessary information for the request.
     * @return std::string The response data received from the request.
     */
    [[nodiscard("Data stored in returned string.")]] static std::string fetchEnnovatisData(const ennovatis::RESTRequest &req);

    /**
     * @brief Performs a CURL request to the specified URL.
     * 
     * This function sends a CURL request to the specified URL and stores the response data in the provided string.
     * 
     * @param url The URL to send the request to.
     * @param response The response data received from the request (storage).
     * @return bool True if the request was successful, false otherwise.
     */
    [[nodiscard("Make sure failing gets recognized.")]] static bool performCurlRequest(const std::string &url,
                                                                                std::string &response);

    /**
     * @brief Cleans up the CURL cache.
     * 
     * This function needs to be called once for each application to properly clean up the CURL library.
     */
    static void cleanupcurl();
};

} // namespace ennovatis

#endif
