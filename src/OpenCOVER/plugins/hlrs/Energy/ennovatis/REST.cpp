#include "REST.h"
// #include <iostream>
#include <curl/curl.h>
#include <string>

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
size_t writeCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}
}

namespace ennovatis {
bool performCurlRequest(const string& url, string& response) {
    auto curl = curl_easy_init();
    if (!curl) {
        // cerr << "Failed to initialize CURL" << endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    auto res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        // std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
        return false;
    }

    return true;
}

void cleanupcurl() { curl_global_cleanup(); }
}