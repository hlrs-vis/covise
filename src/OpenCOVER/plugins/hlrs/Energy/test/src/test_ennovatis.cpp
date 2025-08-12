#include <lib/ennovatis/building.h>
#include <lib/ennovatis/channel.h>
#include <lib/ennovatis/csv.h>
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

/************* building tests **************/

TEST(Ennovatis, ValidBuildingConstructor)
{
    Building b = Building("Hoechstleistungsrechenzentrum", "000");

    EXPECT_EQ(b.getId(), "000");
    EXPECT_EQ(b.getName(), "Hoechstleistungsrechenzentrum");
    EXPECT_EQ(b.getStreet(), "");
    EXPECT_EQ(b.getX(), 0);
    EXPECT_EQ(b.getY(), 0);
    EXPECT_EQ(b.getHeight(), 0);
    EXPECT_EQ(b.getArea(), 0);
}

TEST(Ennovatis, ValidBuildingGetterSetter)
{
    Building b = Building("Hoechstleistungsrechenzentrum");
    b.setId("001");
    b.setName("HLRS");
    b.setStreet("Nobelstrasse");
    b.setX(1);
    b.setY(2);
    b.setHeight(3);
    b.setArea(4);

    EXPECT_EQ(b.getId(), "001");
    EXPECT_EQ(b.getName(), "HLRS");
    EXPECT_EQ(b.getStreet(), "Nobelstrasse");
    EXPECT_EQ(b.getX(), 1);
    EXPECT_EQ(b.getY(), 2);
    EXPECT_EQ(b.getHeight(), 3);
    EXPECT_EQ(b.getArea(), 4);
}

/************** channel tests **************/

TEST(Ennovatis, ValidChannelEmpty)
{
    Channel c;
    EXPECT_TRUE(c.empty());

    auto test_property = [&](auto setter, auto resetter) {
        setter();
        EXPECT_FALSE(c.empty());
        resetter();
        EXPECT_TRUE(c.empty());
    };

    test_property([&]() { c.name = "channel"; }, [&]() { c.name = ""; });
    test_property([&]() { c.id = "001"; }, [&]() { c.id = ""; });
    test_property([&]() { c.description = "description"; }, [&]() { c.description = ""; });
    test_property([&]() { c.type = "HLRS"; }, [&]() { c.type = ""; });
    test_property([&]() { c.unit = "HLRS"; }, [&]() { c.unit = ""; });
    test_property([&]() { c.group = ChannelGroup::Strom; }, [&]() { c.group = ChannelGroup::None; });
}

TEST(Ennovatis, ValidChannelClear)
{
    Channel c;
    c.name = "channel";
    c.id = "001";
    c.description = "description";
    c.type = "type";
    c.unit = "unit";
    // c.group = ChannelGroup::Strom; // TODO: clear function does not yet reset group of channel

    EXPECT_FALSE(c.empty());
    c.clear();
    EXPECT_TRUE(c.empty());
}

TEST(Ennovatis, ValidChannelCompare)
{
    Channel c1, c2;
    c1.id = "001";
    c2.id = "002";
    ChannelCmp cmp;
    EXPECT_TRUE(cmp(c1, c2));
    EXPECT_FALSE(cmp(c2, c1));
    EXPECT_FALSE(cmp(c1, c1));
}

TEST(Ennovatis, ValidChannelChannelGroupToStringConstExpr)
{
    EXPECT_EQ(ChannelGroupToStringConstExpr(ChannelGroup::Strom), "Strom");
    EXPECT_EQ(ChannelGroupToStringConstExpr(ChannelGroup::Wasser), "Wasser");
    EXPECT_EQ(ChannelGroupToStringConstExpr(ChannelGroup::Waerme), "Waerme");
    EXPECT_EQ(ChannelGroupToStringConstExpr(ChannelGroup::Kaelte), "Kaelte");
    EXPECT_EQ(ChannelGroupToStringConstExpr(ChannelGroup::None), "None");
}

TEST(Ennovatis, ValidChannelChannelGroupToString)
{
    EXPECT_EQ(ChannelGroupToString(ChannelGroup::Strom), "Strom");
    EXPECT_EQ(ChannelGroupToString(ChannelGroup::Wasser), "Wasser");
    EXPECT_EQ(ChannelGroupToString(ChannelGroup::Waerme), "Waerme");
    EXPECT_EQ(ChannelGroupToString(ChannelGroup::Kaelte), "Kaelte");
    EXPECT_EQ(ChannelGroupToString(ChannelGroup::None), "None");
}

/**************** csv tests ****************/

TEST(Ennovatis, ValidCSVUpdateBuildingsByBuildingID)
{
    Buildings buildings;
    EXPECT_TRUE(buildings.empty());

    std::istringstream csv(
        "PROJECT,PROJECT_ID,BUILDING_CHANNEL_DIR,BUILDING_ID,DATASOURCE_SUBCHANNEL_DIR,DATASOURCE_ID,CHANNEL,CHANNEL_ID,GLOBAL_ID,DESCRIPTION,TYPE,MAIN_SUB_TYPE,BASELINE,PREDICTION,ACRONYM,FUNCTION\n"
        "Project,P001,BuildingA-Dir,B001,SubDir,DS001,channel,CH001,GID001,description,POWER,SubType,0,0,ACR,Func\n"
    );
    bool result = csv_channelid_parser::update_buildings_by_buildingid(csv, buildings);
    EXPECT_TRUE(result);
    EXPECT_EQ(buildings.size(), 1);
}

TEST(Ennovatis, ValidDateTimeStrConversion)
{
    std::string ref_string = "01.01.2000";
    auto tp = ennovatis::date::str_to_time_point(ref_string, ennovatis::date::dateformat);
    auto result = ennovatis::date::time_point_to_str(tp, ennovatis::date::dateformat);
    EXPECT_EQ(ref_string, result);
}

TEST(Ennovatis, ValidRequestStr)
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
TEST(Ennovatis, ValidJSONSAXParsing)
{
    std::ifstream pokemonFilestream(pokemonJSON);
    sax_channelid_parser slp;
    // no errors in parsing
    ASSERT_TRUE(nlohmann::json::sax_parse(pokemonFilestream, &slp));
    pokemonFilestream.close();
}

TEST(Ennovatis, ValidLogging)
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
