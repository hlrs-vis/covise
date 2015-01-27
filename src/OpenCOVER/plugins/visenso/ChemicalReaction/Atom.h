/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ATOM_H
#define _ATOM_H

#include "Design.h"

#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/Vec3>
#include <osgText/Text3D>

#include <iostream>

class StartMolecule;
class EndMolecule;

class Atom
{
public:
    Atom(AtomConfig _startConfig, StartMolecule *_startMolecule);
    ~Atom();

    osg::Node *getNode()
    {
        return transform;
    };
    int getElement()
    {
        return startConfig.element;
    };

    void reset();
    void animate(float animationTime);

    AtomConfig startConfig;
    AtomConfig endConfig;

    StartMolecule *startMolecule;
    EndMolecule *endMolecule;

private:
    osg::ref_ptr<osg::MatrixTransform> transform;

    osg::ref_ptr<osg::Geode> sphereGeode;
    osg::ref_ptr<osg::ShapeDrawable> sphereDrawable;
    osg::ref_ptr<osg::Sphere> sphereGeometry;

    osg::ref_ptr<osgText::Text3D> textDrawable;
    osg::ref_ptr<osg::Geode> textGeode;
};

#endif
