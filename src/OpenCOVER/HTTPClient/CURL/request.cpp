#include "request.h"
#include <curl/curl.h>
#include <string>

using namespace std;

namespace opencover {
namespace httpclient {
namespace curl {

CURL *Request::initCurl(const HTTPMethod &method, std::string &response, const Options &options) const
{
    auto curl = curl_easy_init();
    if (!curl)
        return nullptr;

    method.setupCurl(curl);

    //don't wait forever
    curl_easy_setopt(curl,CURLOPT_TIMEOUT, 10);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    //custom options
    for (const auto &option : options) {
        std::visit([&curl, &option](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>) {
                curl_easy_setopt(curl, option.first, arg.c_str());
            } else {
                curl_easy_setopt(curl, option.first, arg);
            }
        }, option.second);
    }
    return curl;
}

bool Request::httpRequest(const HTTPMethod &method, string &response, const Options &options) const
{
    auto curl = initCurl(method, response, options);
    if (!curl)
        return false;

    auto res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        auto error_msg = method.to_string() + string(curl_easy_strerror(res));
        fprintf(stderr, "curl_easy_perform() failed: %s\n", error_msg.c_str());
        return false;
    }
    method.cleanupCurl(curl);

    return true;
}

} // namespace curl
} // namespace httpclient
} // namespace opencover