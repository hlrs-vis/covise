/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/coExport.h>
#include <string>
namespace vive
{
class VVCORE_EXPORT vvVIVEConfig
{
private:
    vvVIVEConfig();
    virtual ~vvVIVEConfig();

public:
    static bool getScreenConfigEntry(int pos, std::string &name, int *hsize, int *vsize, int *x, int *y, int *z, float *h, float *p, float *r);
};
}
