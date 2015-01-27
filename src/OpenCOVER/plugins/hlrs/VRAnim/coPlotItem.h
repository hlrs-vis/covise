/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_PLOTITEM_H
#define CO_PLOTITEM_H

#include <sys/types.h>
#include <OpenVRUI/coAction.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osgDB/ReadFile>
#include <osg/Program>
#include <osg/Shader>
#include <osg/Point>
#include <osg/ShadeModel>

#include "anim.h"

namespace covise
{
class coTrackerButtonInteraction;
}
class coPlotItem;
using namespace covise;
using namespace opencover;
using namespace vrui;
extern struct animation str;
extern struct plotter plo;
extern struct plotdata dat;
extern float colorindex[MAXCOLORS + MAXNODYNCOLORS + 1][4];

class coPlotItem : public coAction, public coUIElement
{
public:
    coPlotItem(int i);
    virtual ~coPlotItem();

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    virtual int hit(vruiHit *);

    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    void setPos(float x, float y, float z = 0);
    vruiTransformNode *getDCS();

    virtual float getWidth() const
    {
        return plo.b * 0.05 + plo.b * 1.05;
    };
    virtual float getHeight() const
    {
        return plo.h * 0.05 + plo.h * 1.05;
    };
    virtual float getXpos() const
    {
        return myX;
    };
    virtual float getYpos() const
    {
        return myY;
    };
    virtual void update();
    osg::ref_ptr<osg::Geode> createGeode();

    osg::ref_ptr<osg::Geode> createPlotter(int plotterid);
    osg::ref_ptr<osg::DrawArrays> lineDrawArray;

protected:
    OSGVruiTransformNode *myDCS;
    float myX, myY;
};
#endif
