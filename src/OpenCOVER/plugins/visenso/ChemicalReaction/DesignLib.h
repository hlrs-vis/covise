/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DESIGN_LIB_H
#define _DESIGN_LIB_H

#include "Design.h"

#include <osg/Geode>
#include <osg/Vec3>

#include <iostream>

class DesignLib
{
public:
    static DesignLib *Instance();

    Design *getDesign(std::string symbol);

    std::vector<Design *> designList;

private:
    DesignLib();
    static DesignLib *instance;
};

#endif
