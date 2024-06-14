#ifndef _ENERGY_ENNOVATIS_REST_H
#define _ENERGY_ENNOVATIS_REST_H

#include <string>
#include "building.h"
#include "utils/thread/threadworker.h"

namespace ennovatis {
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
    [[nodiscard]] std::string operator()() const;
};

/*
 * @brief The `rest_request_handler` struct provides utility functions for handling REST requests to Ennovatis via utils::threadworker.
*/
struct rest_request_handler {
    /**
     * @brief Fetches the channels from a given channel group and building.
     * 
     * This function fetches the channels from the specified channel group and building
     * and populates the REST request object with last used channelid. Results will be available in worker by accessing futures over threads.
     * 
     * @param group The channel group to fetch channels from.
     * @param b The building to fetch channels from.
     * @param req The REST request object to populate with fetched channels. (will be copied to worker threads)
     */
    void fetchChannels(const ChannelGroup &group, const ennovatis::Building &b, ennovatis::rest_request req);

    /**
     * @brief Retrieves the result.
     * 
     * This function returns a unique pointer to a vector of strings that represents the result if all threads added to the worker are finished.
     * 
     * @return A unique pointer to a vector of strings containing the result.
     */
    auto getResult() { return worker.getResult(); }
    auto checkStatus() { return worker.checkStatus(); }
    auto isRunning() { return worker.isRunning(); }

private:
    opencover::utils::ThreadWorker<std::string> worker;
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
    [[nodiscard]] static std::string fetch_data(const rest_request &req);
};

} // namespace ennovatis

#endif
