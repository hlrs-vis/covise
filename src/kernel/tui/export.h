#ifndef CO_TUI_EXPORT_H
#define CO_TUI_EXPORT_H

#include <util/coExport.h>

#if defined(COVISE_TUI)
#define TUIEXPORT COEXPORT
#else
#define TUIEXPORT COIMPORT
#endif


#endif
