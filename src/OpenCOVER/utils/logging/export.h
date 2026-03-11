#pragma once
#include <util/coExport.h>

#if defined(coLog_EXPORTS)
#define LOGGINGUTIL COEXPORT
#else
#define LOGGINGUTIL COIMPORT
#endif
