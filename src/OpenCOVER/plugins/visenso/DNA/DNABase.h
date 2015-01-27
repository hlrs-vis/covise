/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BASE_H
#define _BASE_H

#include "DNABaseUnit.h"
class DNABaseUnitConnectionPoint;

class DNABase : public DNABaseUnit
{
public:
    DNABase(osg::Matrix m, float size, const char *interactorName, std::string geofilename, float boundingRadius, bool left, int num = 0);
    virtual ~DNABase();

protected:
    DNABaseUnitConnectionPoint *d1;
    std::list<DNABaseUnitConnectionPoint *> enabledCP;

    osg::ref_ptr<osg::Node> connectionGeom;

    virtual void createGeometry();
    virtual void showConnectionGeom(bool b, std::string connName);
};

#endif
