#ifndef _ENERGY_ENNOVATIS_REST_H
#define _ENERGY_ENNOVATIS_REST_H

#include <string>
#include <chrono>

namespace ennovatis {

struct rest_request_handler {

};

/**
 * @brief The RESTRequest struct represents a REST request object.
 * 
 * It contains the necessary information to make a REST request, such as the URL, project ID, channel ID,
 * time range, and resolution.
 */
struct rest_request {
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

/*
 * @brief The `rest` struct provides utility functions for performing REST requests to Ennovatis.
 */
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
    [[nodiscard("Data stored in returned string.")]] static std::string fetch_data(const rest_request &req);
};

} // namespace ennovatis

#endif
