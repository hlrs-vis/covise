#ifndef COVER_UTILS_READ_CSV_EXPORT_H
#define COVER_UTILS_READ_CSV_EXPORT_H

#include <util/coExport.h>

#if defined(coReadCSVUtil_EXPORTS)
#define CSVUTIL COEXPORT
#else
#define CSVUTIL COIMPORT
#endif

#endif // COVER_UTILS_READ_CSV_EXPORT_H