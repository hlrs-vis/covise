#ifndef MULTITOUCH_PLUGIN_EXPORT_H
#define MULTITOUCH_PLUGIN_EXPORT_H

#include <util/coExport.h>

#ifdef CoverMultitouch_EXPORTS
#define MULTITOUCHEXPORT COEXPORT
#else
#define MULTITOUCHEXPORT COIMPORT
#endif

#endif
