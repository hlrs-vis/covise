/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _END_MOLECULE_H
#define _END_MOLECULE_H

#include "Molecule.h"

#include <osg/Geode>
#include <osg/Vec3>

#include <iostream>

class EndMolecule : public Molecule
{
public:
    EndMolecule(Design *_design);
    virtual ~EndMolecule();

    osg::Vec3 getPosition();
    void setPosition(osg::Vec3 pos);

private:
    osg::Vec3 position;
};

#endif
