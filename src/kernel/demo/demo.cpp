#include "demo.h"
#include <cstdlib>
#include <filesystem>

const char* HLRS_DEMO_DIR_CHAR = getenv("HLRS_DEMO_DIR");
const char* HLRS_DEMO_PORT_CHAR = getenv("HLRS_DEMO_PORT");
const char* COVISEDIR = getenv("COVISEDIR");


std::string getRoot()
{
    if(HLRS_DEMO_DIR_CHAR) return HLRS_DEMO_DIR_CHAR;
     std::filesystem::path p("/data/hlrsDemo");
    if (std::filesystem::exists(p))
        return p.string();
    return "";
}


std::string coviseDemoDir()
{
    if (COVISEDIR)
        return std::string(COVISEDIR) + "/share/covise/demo";
    return "";
}


const std::string demo::root = getRoot();
const std::string demo::collection = coviseDemoDir() + "/demos.json";
const std::string demo::imageDir = demo::root + "/static/screenshots";
const std::string demo::logFile = coviseDemoDir() + "/launch_log.jsonl";
const std::string demo::indexHtml = coviseDemoDir() + "/index.html";
const int demo::port = HLRS_DEMO_PORT_CHAR && atoi(HLRS_DEMO_PORT_CHAR) ? atoi(HLRS_DEMO_PORT_CHAR) : 31095;
