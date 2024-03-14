#ifndef _ENERGY_UTILS_REST_H
#define _ENERGY_UTILS_REST_H

#include <curl/curl.h>
#include <string>

namespace utils {

/**
 * @brief The `rest` struct provides utility functions for performing CURL requests and cleaning up the CURL library.
 */
struct rest {
    /**
     * @brief Performs a CURL request to the specified URL.
     * 
     * This function sends a CURL request to the specified URL and stores the response data in the provided string.
     * 
     * @param url The URL to send the request to.
     * @param response The response data received from the request (storage).
     * @return bool True if the request was successful, false otherwise.
     */
    [[nodiscard]] static bool performCurlRequest(const std::string &url,
                                                                                       std::string &response);

    /**
     * @brief Cleans up the CURL cache.
     * 
     * This function needs to be called once for each application to properly clean up the CURL library.
     */
    static void cleanupcurl();
};
} // namespace utils

#endif
