#include "request.h"
#include <curl/curl.h>
#include <string>

using namespace std;

namespace opencover {
namespace httpclient {
namespace curl {

CURL *Request::initCurl(const HTTPMethod &method, std::string &response)
{
    auto curl = curl_easy_init();
    if (!curl)
        return nullptr;

    method.setupCurl(curl);

    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    return curl;
}

bool Request::httpRequest(const HTTPMethod &method, string &response)
{
    auto curl = initCurl(method, response);
    if (!curl)
        return false;

    auto res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        return false;
    }
    method.cleanupCurl(curl);

    return true;
}

} // namespace curl
} // namespace httpclient
} // namespace opencover