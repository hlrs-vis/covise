#ifndef _HTTPCLIENT_CURL_REQUEST_H
#define _HTTPCLIENT_CURL_REQUEST_H

#include "export.h"
#include <curl/curl.h>
#include <string>

namespace opencover {
namespace httpclient {
namespace curl {

struct CURLHTTPCLIENTEXPORT Request {
    /**
     * @brief Performs a CURL request to the specified URL.
     * 
     * This function sends a CURL request to the specified URL and stores the response data in the provided string.
     * 
     * @param url The URL to send the request to.
     * @param response The response data received from the request (storage).
     * @return bool True if the request was successful, false otherwise.
     */
    [[nodiscard]] static bool httpRequest(const std::string &url, std::string &response);

    /**
     * @brief Cleans up the CURL cache.
     * 
     * This function needs to be called once for each application to properly clean up the CURL library.
     */
    static void cleanup();
};

} // namespace curl
} // namespace httpclient
} // namespace opencover

#endif