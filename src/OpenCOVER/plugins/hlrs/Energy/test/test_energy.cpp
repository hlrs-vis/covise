#include "../ennovatis/REST.h"
#include "../ennovatis/sax.h"

// #include "REST.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>

using namespace ennovatis;

namespace {
constexpr auto testDataDir(ENERGYCAMPUS_TEST_DATA_DIR);
std::string pathToJSON = testDataDir;
std::string pokemonJSON = pathToJSON + "/pokemon.json";

/**************** REST tests ****************/
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

TEST(performCurlRequestTest, ValidResponse)
{
    std::string url("https://hacker-news.firebaseio.com/v0/item/8863.json");
    std::string response;
    std::string ref_string(R"({"by":"dhouston","descendants":71,"id":8863,"kids":[9224,8917,8884,8887,8952,8869,8873,8958,8940,8908,9005,9671,9067,9055,8865,8881,8872,8955,10403,8903,8928,9125,8998,8901,8902,8907,8894,8870,8878,8980,8934,8943,8876],"score":104,"time":1175714200,"title":"My YC app: Dropbox - Throw away your USB drive","type":"story","url":"http://www.getdropbox.com/u/2/screencast.html"})");
    bool result = performCurlRequest(url, response);
    ASSERT_TRUE(result);
    // Add additional assertions to validate the response
    EXPECT_EQ(ref_string, response);
}

TEST(cleanupCurl, ValidCleanup)
{
    cleanupcurl();
    // Add additional assertions to validate the cleanup
}

/**************** SAX tests ****************/
TEST(saxTest, ValidJSONSAXParsing)
{
    std::ifstream pokemonFilestream(pokemonJSON);
    sax_channelid_parser slp;
    // no errors in parsing
    ASSERT_TRUE(nlohmann::json::sax_parse(pokemonFilestream, &slp));
    pokemonFilestream.close();
}

TEST(saxTest, ValidLogging)
{
    std::ifstream pokemonFilestream(pokemonJSON);
    std::ifstream resultFilestream(pathToJSON + "/test_pokemon_logging.txt");
    sax_channelid_parser slp;
    nlohmann::json::sax_parse(pokemonFilestream, &slp);
    pokemonFilestream.close();
    EXPECT_TRUE(!slp.getDebugLogs().empty());
    for (auto &log: slp.getDebugLogs()) {
        std::string line;
        std::getline(resultFilestream, line);
        EXPECT_EQ(line, log);
    }
}

/**************** Building tests ****************/
// TEST(buidlingText, ValidBuidling)
// {

} // namespace