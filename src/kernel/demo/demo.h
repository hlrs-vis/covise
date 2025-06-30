#ifndef HLRS_DEMO_COMMONH
#define HLRS_DEMO_COMMONH

#include <util/coExport.h>
#include <string>

#if defined(hlrsDemoCommon_EXPORTS)
#define HLRSDEMOEXPORT COEXPORT
#else
#define HLRSDEMOEXPORT COIMPORT
#endif

namespace demo{

extern HLRSDEMOEXPORT const std::string root; // Path to the HLRS demo directory
extern HLRSDEMOEXPORT const std::string collection; // Path to the demos.json file
extern HLRSDEMOEXPORT const std::string imageDir; // Path to the static files e.g. images
extern HLRSDEMOEXPORT const std::string logFile; // Path to the static files e.g. images
extern HLRSDEMOEXPORT const std::string indexHtml; // Path to the static files e.g. images
extern HLRSDEMOEXPORT const int port; // Port for the demo server

}

#endif // HLRS_DEMO_COMMONH