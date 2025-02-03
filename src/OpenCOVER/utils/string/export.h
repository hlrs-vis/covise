#ifndef UTIL_STRING_EXPORT_H
#define UTIL_STRING_EXPORT_H
#include <util/coExport.h>

#if defined(coStringUtil_EXPORTS)
#define STRING_EXPORT COEXPORT
#else
#define STRING_EXPORT COIMPORT
#endif

#endif // UTIL_STRING_EXPORT_H