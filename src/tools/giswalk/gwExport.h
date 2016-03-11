#ifndef GWEXPORT_H
#define GWEXPORT_H

#include <util/coExport.h>

#ifdef GWAPP_EXPORT
#define GWAPPEXPORT COEXPORT
#else
#define GWAPPEXPORT COIMPORT
#endif

#endif
