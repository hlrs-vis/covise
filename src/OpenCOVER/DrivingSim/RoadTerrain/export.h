#ifndef ROADTERRAIN_EXPORT_H
#define ROADTERRAIN_EXPORT_H

#include <util/coExport.h>

#if defined(coRoadTerrain_EXPORTS)
#define ROADTERRAINEXPORT COEXPORT
#else
#define ROADTERRAINEXPORT COIMPORT
#endif

#endif
