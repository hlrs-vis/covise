/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CARBON_H
#define _CARBON_H
#include "Atom.h"
class Carbon : public Atom
{
public:
    Carbon(string symbol, const char *interactorName, osg::Matrix m, float size, std::vector<osg::Vec3> connections, osg::Vec4 color);
    bool isMethan();
    bool isEthan();
    bool isPropan();
    bool isLinearButan();
    bool isLinearAlkane(int numC);

    Carbon *getNextCarbon(Carbon *);
    int getNumAtoms(string symbol);

private:
    bool isAlkaneMiddle();
    bool isAlkaneEnd();
};
#endif
