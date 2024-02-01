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
    for (auto& log : slp.getDebugLogs())
    {
        std::string line;
        std::getline(resultFilestream, line);
        EXPECT_EQ(line, log);
    }
}
} // namespace