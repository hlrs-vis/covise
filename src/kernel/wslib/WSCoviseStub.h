/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#include <winsock2.h>
#endif

#include "WSExport.h"

namespace covise
{
class covise__ColormapPin;
class covise__Colormap;

bool operator==(const covise::covise__ColormapPin &p1, const covise::covise__ColormapPin &p2);
bool operator!=(const covise::covise__ColormapPin &p1, const covise::covise__ColormapPin &p2);

bool operator==(const covise::covise__Colormap &c1, const covise::covise__Colormap &c2);
bool operator!=(const covise::covise__Colormap &c1, const covise::covise__Colormap &c2);
}

#ifndef SOAP_CMAC
#define SOAP_CMAC WSLIBEXPORT
#define SOAP_FMAC1 WSLIBEXPORT
#define SOAP_FMAC3 WSLIBEXPORT
#endif

#include "stdsoap2.h"

#ifdef WSLIB_EXPORT
#include <coviseStub.h>
#else
#include <wslib/coviseStub.h>
#endif
