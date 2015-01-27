/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   TransparentVisitor that uses distance from structure
//		  to determine transparency
//
// Author:        Philip Weber
//
// Creation Date: 2006-02-29
//
// **************************************************************************

#ifndef TRANS_VISITOR_H
#define TRANS_VISITOR_H

#include <math.h>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Drawable>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Geometry>
#include <osg/StateAttribute>
#include <osg/Vec3>
#include <osg/Shape>
#include <osg/BoundingBox>
#include <osg/Switch>
#include <osg/AlphaFunc>
#include <osg/BlendFunc>
#include <vector>

using namespace std;
using namespace osg;

class TransparentVisitor : public osg::NodeVisitor
{
private:
    osg::Material *material;
    osg::StateSet *stateset;
    osg::Drawable *drawable;
    osg::Node *currentnode;
    bool _transparent_mode;
    float getDistance(osg::Vec3, osg::Vec3);

protected:
    bool _transparent;
    float _alpha;
    bool _highdetail;
    float _distance;
    float _area;
    float _scale;
    virtual void calculateAlphaAndBin(float) = 0;

public:
    TransparentVisitor();
    void setDistance(float);
    void setArea(float);
    void setScale(float);
    void enableTransparency(bool);
    virtual void apply(osg::Geode &);
};
#endif
