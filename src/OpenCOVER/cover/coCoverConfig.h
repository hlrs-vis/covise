/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COVERCONFIG
#define CO_COVERCONFIG

/*! \file
 \brief  convenience class for reading configuration data

 \author Andreas Kopecki <kopecki@hlrs.de>
 \author (C) 2005
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/coExport.h>
#include <string>
namespace opencover
{
class COVEREXPORT coCoverConfig
{
private:
    coCoverConfig();
    virtual ~coCoverConfig();

public:
    static bool getScreenConfigEntry(int pos, std::string &name, int *hsize, int *vsize, int *x, int *y, int *z, float *h, float *p, float *r);
    static bool getWindowConfigEntry(int pos, std::string &name, int *pipeNum, int *ox, int *oy, int *sx, int *sy, bool *decoration = 0, bool *stereo = 0, bool *resize = 0, bool *embedded = 0, bool *pbuffer = 0);
};
}
#endif
