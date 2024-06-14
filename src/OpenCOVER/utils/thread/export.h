#ifndef COVER_UTILS_THREAD_EXPORT_H
#define COVER_UTILS_THREAD_EXPORT_H

#include <util/coExport.h>

#if defined(coThreadUtil_EXPORTS)
#define THREADUTIL COEXPORT
#else
#define THREADUTIL COIMPORT
#endif

#endif // COVER_UTILS_THREAD_EXPORT_H