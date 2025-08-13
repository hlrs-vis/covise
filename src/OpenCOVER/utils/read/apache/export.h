#pragma once
#include <util/coExport.h>

#if defined(coApacheArrow_EXPORTS)
#define ARROWUTIL COEXPORT
#else
#define ARROWUTIL COIMPORT
#endif
