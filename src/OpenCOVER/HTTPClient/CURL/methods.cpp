#include "methods.h"
#include <string>

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
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}
} // namespace

namespace opencover {
namespace httpclient {
namespace curl {

void HTTPMethod::cleanupCurl(CURL *curl) const
{
    curl_easy_cleanup(curl);
}

void HTTPMethod::setupCurl(CURL *curl) const
{
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
}

void POST::setupCurl(CURL *curl) const
{
    HTTPMethod::setupCurl(curl);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, requestBody.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
}

void POST::cleanupCurl(CURL *curl) const
{
    HTTPMethod::cleanupCurl(curl);
}

} // namespace curl
} // namespace httpclient
} // namespace opencover