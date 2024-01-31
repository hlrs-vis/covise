#include "../ennovatis/REST.h"
#include <gtest/gtest.h>

using namespace ennovatis;

namespace {
TEST(performCurlRequestTest, ValidUrl)
{
    std::string url = "example.com";
    std::string response;
    bool result = performCurlRequest(url, response);
    ASSERT_TRUE(result);
    // Add additional assertions to validate the response
}

TEST(performCurlRequestTest, InvalidUrl)
{
    std::string url = "https://api.invalid.com";
    std::string response;
    bool result = performCurlRequest(url, response);
    ASSERT_FALSE(result);
    // Add additional assertions to validate the response
}
} // namespace