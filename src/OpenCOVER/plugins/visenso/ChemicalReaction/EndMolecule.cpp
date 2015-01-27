/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EndMolecule.h"

EndMolecule::EndMolecule(Design *_design)
    : Molecule(_design)
{
}

EndMolecule::~EndMolecule()
{
}

osg::Vec3 EndMolecule::getPosition()
{
    return position;
}

void EndMolecule::setPosition(osg::Vec3 pos)
{
    position = pos;
}
