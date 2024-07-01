#ifndef _HTTPCLIENT_CURL_HTTPMETHODS_H
#define _HTTPCLIENT_CURL_HTTPMETHODS_H

#include "export.h"
#include <string>
#include <curl/curl.h>

namespace opencover {
namespace httpclient {
namespace curl {

class CURLHTTPCLIENTEXPORT HTTPMethod {
public:
    HTTPMethod() = delete;
    explicit HTTPMethod(const std::string &url): url(url) {}
    virtual void setupCurl(CURL *curl) const;
    virtual void cleanupCurl(CURL *curl) const;
    virtual const std::string to_string() const { return "URL: " + url + "\n"; }

protected:
    std::string url;
};

#define HTTP_METHOD(NAME) class CURLHTTPCLIENTEXPORT NAME: public HTTPMethod

HTTP_METHOD(GET)
{
public:
    GET() = delete;
    explicit GET(const std::string &url): HTTPMethod(url)
    {}
};

HTTP_METHOD(POST)
{
public:
    POST() = delete;
    ~POST()
    {
        curl_slist_free_all(headers);
    }

    explicit POST(const std::string &url, const std::string &requestBody): HTTPMethod(url), requestBody(requestBody)
    {
        initHeaders();
    }

    // copy constructor
    POST(const POST &other): HTTPMethod(other.url), requestBody(other.requestBody)
    {
        initHeaders();
    }

    // copy assignment opertator
    POST &operator=(const POST &other)
    {
        if (this != &other)
            initHeaders();
        requestBody = other.requestBody;
        return *this;
    }

    void setupCurl(CURL * curl) const override;
    void cleanupCurl(CURL * curl) const override;
    const std::string to_string() const override
    {
        return HTTPMethod::to_string() + "Requestbody: " + requestBody + "\n";
    }

private:
    void initHeaders()
    {
        headers = curl_slist_append(headers, "Content-Type: application/json");
    }
    struct curl_slist *headers = nullptr;
    std::string requestBody;
};

// not implemented
// HTTP_METHOD(PUT){};
// HTTP_METHOD(HEAD){};
// HTTP_METHOD(DELETE){};
// HTTP_METHOD(PATCH){};
// HTTP_METHOD(OPTIONS){};
// HTTP_METHOD(CONNECT){};
// HTTP_METHOD(TRACE){};

} // namespace curl
} // namespace httpclient
} // namespace opencover
#endif