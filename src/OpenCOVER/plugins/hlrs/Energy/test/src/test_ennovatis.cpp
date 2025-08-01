#include <lib/ennovatis/rest.h>
#include <lib/ennovatis/sax.h>
#include <lib/ennovatis/date.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>

using namespace ennovatis;

namespace {
constexpr auto testDataDir(ENERGYCAMPUS_TEST_DATA_DIR);
std::string pathToJSON = testDataDir;
std::string pokemonJSON = pathToJSON + "/pokemon.json";

/**************** REST tests ****************/
// MOVE THIS TO HTTPClient/CURL/test/test_curl.cpp
// TEST(REST, ValidUrl)
// {
//     std::string url = "example.com";
//     std::string response;
//     bool result = utils::rest::performCurlRequest(url, response);
//     ASSERT_TRUE(result);
//     // Add additional assertions to validate the response
// }

// TEST(REST, InvalidUrl)
// {
//     std::string url = "https://api.invalid.com";
//     std::string response;
//     bool result = utils::rest::performCurlRequest(url, response);
//     ASSERT_FALSE(result);
//     // Add additional assertions to validate the response
// }

// TEST(REST, ValidResponse)
// {
//     std::string url("https://hacker-news.firebaseio.com/v0/item/8863.json");
//     std::string response;
//     std::string ref_string(
//         R"({"by":"dhouston","descendants":71,"id":8863,"kids":[9224,8917,8884,8887,8952,8869,8873,8958,8940,8908,9005,9671,9067,9055,8865,8881,8872,8955,10403,8903,8928,9125,8998,8901,8902,8907,8894,8870,8878,8980,8934,8943,8876],"score":104,"time":1175714200,"title":"My YC app: Dropbox - Throw away your USB drive","type":"story","url":"http://www.getdropbox.com/u/2/screencast.html"})");
//     bool result = utils::rest::performCurlRequest(url, response);
//     ASSERT_TRUE(result);
//     // Add additional assertions to validate the response
//     EXPECT_EQ(ref_string, response);
// }

// TEST(REST, ValidCleanup)
// {
//     utils::rest::cleanupcurl();
//     // Add additional assertions to validate the cleanup
// }

TEST(EnnovatisREST, ValidDateTimeStrConversion)
{
    std::string ref_string = "01.01.2000";
    auto tp = ennovatis::date::str_to_time_point(ref_string, ennovatis::date::dateformat);
    auto result = ennovatis::date::time_point_to_str(tp, ennovatis::date::dateformat);
    EXPECT_EQ(ref_string, result);
}

TEST(EnnovatisREST, ValidRequestStr)
{
    rest_request req;
    req.url = "https://wurstbrot.com/v0/item";
    req.projEid = "123";
    req.channelId = "456";
    req.dtf = ennovatis::date::str_to_time_point("01.01.2000", ennovatis::date::dateformat);
    req.dtt = ennovatis::date::str_to_time_point("01.02.2000", ennovatis::date::dateformat);
    req.ts = 86400;
    req.tsp = 0;
    req.tst = 1;
    req.etst = 1024;
    std::string result = req();
    std::string ref_string =
        "https://wurstbrot.com/v0/"
        "item?projEid=123&dtf=01.01.2000&dtt=01.02.2000&ts=86400&tsp=0&tst=1&etst=1024&cEid=456";
    EXPECT_EQ(ref_string, result);
}

/**************** SAX tests ****************/
TEST(EnnovatisSAXTest, ValidJSONSAXParsing)
{
    std::ifstream pokemonFilestream(pokemonJSON);
    sax_channelid_parser slp;
    // no errors in parsing
    ASSERT_TRUE(nlohmann::json::sax_parse(pokemonFilestream, &slp));
    pokemonFilestream.close();
}

TEST(EnnovatisSAXTest, ValidLogging)
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

} // namespace
