#ifndef _HTTPCLIENT_CURL_REQUEST_H
#define _HTTPCLIENT_CURL_REQUEST_H

#include "export.h"
#include "methods.h"
#include <curl/curl.h>
#include <vector>
#include <variant>
#include <string>

namespace opencover {
namespace httpclient {
namespace curl {

using CURLOptionValue = std::variant<long, std::string, void*, curl_off_t, curl_write_callback>;
struct CURLHTTPCLIENTEXPORT Request {
    typedef std::vector<std::pair<CURLoption, CURLOptionValue>> Options;
    Request() { curl_global_init(CURL_GLOBAL_DEFAULT); };
    ~Request() { curl_global_cleanup(); };
    Request(const Request &) = delete;
    Request &operator=(const Request &other) = delete;

    /**
     * @brief Performs a CURL request to the specified URL.
     * 
     * This function sends a CURL request to the specified URL and stores the response data in the provided string.
     * 
     * @param method HTTPMethod object (GET, POST, PUT, DELETE).
     * @param response The response data received from the request (storage).
     * @param options Custom options parameter passed to curl_easy_setopt.
     * @return bool True if the request was successful, false otherwise.
     */
    [[nodiscard]] bool httpRequest(const HTTPMethod &method, std::string &response, const Options &options = Options()) const;

private:
    CURL *initCurl(const HTTPMethod &httpMethod, std::string &response, const Options &options) const;
};

} // namespace curl
} // namespace httpclient
} // namespace opencover

#endif