/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MOLECULE_H
#define _MOLECULE_H

#include "Design.h"
#include "Atom.h"

#include <osg/Geode>
#include <osg/Vec3>
#include <osgText/Text>

#include <iostream>

class Molecule
{
public:
    Molecule(Design *_design);
    virtual ~Molecule();

    virtual osg::Vec3 getPosition() = 0;
    virtual void setPosition(osg::Vec3 pos) = 0;

    void resetAtoms();
    void animateAtoms(float animationTime);
    void showName();

    Design *design;
    std::vector<Atom *> atoms;

private:
    osg::ref_ptr<osgText::Text> textDrawable;
    osg::ref_ptr<osg::Geode> textGeode;
};

#endif
