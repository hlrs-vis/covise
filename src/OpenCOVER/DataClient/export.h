#ifndef COVER_DATACLIENT_EXPORT_H
#define COVER_DATACLIENT_EXPORT_H

#include <util/coExport.h>

#if defined(coDataClient_EXPORTS)
#define DATACLIENTEXPORT COEXPORT
#else
#define DATACLIENTEXPORT COIMPORT
#endif

#endif // COVER_DATACLIENT_EXPORT_H