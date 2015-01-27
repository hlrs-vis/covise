#include "osx-util.h"

#include <AppKit/NSRunningApplication.h>

void MakeMeTheFrontProcess()
{
#if !defined(__USE_WS_X11__)
   [[NSRunningApplication currentApplication] activateWithOptions: NSApplicationActivateAllWindows];
#endif
}
