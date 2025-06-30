#include "demo.h"
#include <cstdlib>

const char* HLRS_DEMO_DIR_CHAR = getenv("HLRS_DEMO_DIR");
const char* HLRS_DEMO_PORT_CHAR = getenv("HLRS_DEMO_PORT");


const std::string demo::root = HLRS_DEMO_DIR_CHAR ? HLRS_DEMO_DIR_CHAR : "";
const std::string demo::collection = demo::root + "/templates/demos.json";
const std::string demo::imageDir = demo::root + "/static/screenshots";
const std::string demo::logFile = demo::root + "/templates/launch_log.jsonl";
const std::string demo::indexHtml = demo::root + "/templates/index.html";
const int demo::port = HLRS_DEMO_PORT_CHAR && atoi(HLRS_DEMO_PORT_CHAR) ? atoi(HLRS_DEMO_PORT_CHAR) : 31095;
