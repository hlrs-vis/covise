/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ARROW_H
#define CO_ARROW_H

/*! \file
 \brief  osg arrow node

 \author Martin Aumueller <aumueller@uni-koeln.de>
 \author (C) 2004
         ZAIK Center for Applied Informatics,
         Robert-Koch-Str. 10, Geb. 52,
         D-50931 Koeln,
         Germany

 \date
 */

#include <osg/Geode>
#include <osg/Material>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <iostream>

namespace osg
{
class Cylinder;
class Cone;
class ShapeDrawable;
class TessellationHints;
class StateSet;
class Material;
};

#include <util/coTypes.h>

namespace opencover
{

class PLUGIN_UTILEXPORT coArrow : public osg::Geode
{
public:
    coArrow(float radius = 1.0, float length = 1.0, bool originAtTip = false, bool draw = true);
    ~coArrow();
    virtual void setColor(osg::Vec4 color);
    virtual void setAmbient(osg::Vec4 ambient);
    virtual void setVisible(bool visible);
    void drawArrow(osg::Vec3 base, float radius, float length);

private:
    osg::Vec4 _color;
    osg::Vec4 _ambient;

    osg::Cylinder *cylinder;
    osg::ShapeDrawable *cylinderDraw;
    osg::Cone *cone;
    osg::ShapeDrawable *coneDraw;

    osg::TessellationHints *hints;
    osg::StateSet *stateSet;
    osg::Material *material;
    bool drawn;
};
}
#endif
