/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _START_MOLECULE_H
#define _START_MOLECULE_H

#include "Molecule.h"

#include <PluginUtil/coVR2DTransInteractor.h>

#include <osg/Geode>
#include <osg/Vec3>

#include <iostream>
using namespace opencover;
using namespace covise;

class StartMolecule : public Molecule, public coVR2DTransInteractor
{
public:
    StartMolecule(Design *_design, osg::Vec3 initPos);
    ~StartMolecule();

    osg::Vec3 getPosition();
    void setPosition(osg::Vec3 pos);

protected:
    void stopInteraction();

private:
    void buildFromDesign();

    osg::Vec3 initPosition;
};

#endif
