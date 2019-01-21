#ifndef QTUTIL_EXPORT_H
#define QTUTIL_EXPORT_H

#include <util/coExport.h>

#if defined (coQtUtil_EXPORTS)
#define QTUTIL_EXPORT COEXPORT
#else
#define QTUTIL_EXPORT COIMPORT
#endif

#endif
