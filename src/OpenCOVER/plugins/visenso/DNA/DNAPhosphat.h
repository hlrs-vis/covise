/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PHOSPHAT_H
#define _PHOSPHAT_H

#include "DNABaseUnit.h"
class DNABaseUnitConnectionPoint;

class DNAPhosphat : public DNABaseUnit
{
public:
    DNAPhosphat(osg::Matrix m, int num = 0);
    virtual ~DNAPhosphat();

    virtual void showConnectionGeom(bool, std::string){};

private:
    DNABaseUnitConnectionPoint *d1;
    DNABaseUnitConnectionPoint *d2;
};

#endif
