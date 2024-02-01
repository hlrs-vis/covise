#include "../ennovatis/REST.h"
#include "../ennovatis/sax.h"

// #include "REST.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>

using namespace ennovatis;

namespace {
constexpr auto pathToJSON("/data/DiTEnS/ennovatis/channelids.json");

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

TEST(saxTest, ValidJSONSAXParsing)
{
    std::ifstream inputFilestream(pathToJSON);
    sax_channelid_parser slp;
    // no errors in parsing
    ASSERT_TRUE(nlohmann::json::sax_parse(inputFilestream, &slp));
}

TEST(saxTest, EnabledLogging)
{
    std::ifstream inputFilestream(pathToJSON);
    sax_channelid_parser slp;
    nlohmann::json::sax_parse(inputFilestream, &slp);
    EXPECT_TRUE(!slp.getDebugLogs().empty());
}
} // namespace