/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DESOXYRIBOSE_H
#define _DESOXYRIBOSE_H

#include <string>
#include "DNABaseUnit.h"
class DNABaseUnitConnectionPoint;

class DNADesoxyribose : public DNABaseUnit
{
public:
    DNADesoxyribose(osg::Matrix m, int num = 0);
    virtual ~DNADesoxyribose();

protected:
    virtual void createGeometry();
    virtual void showConnectionGeom(bool b, std::string connName);

    osg::ref_ptr<osg::Node> connectionGeom1;
    osg::ref_ptr<osg::Node> connectionGeom2;
    osg::ref_ptr<osg::Node> connectionGeom3;

private:
    DNABaseUnitConnectionPoint *p1;
    DNABaseUnitConnectionPoint *p2;
    DNABaseUnitConnectionPoint *b1;
};

#endif
