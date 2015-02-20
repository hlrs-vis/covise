/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MARKER_H
#define _MARKER_H
#include <util/common.h>

#include <osg/MatrixTransform>
#include <osg/Billboard>
#include <osg/Group>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Array>
#include <osg/Switch>
#include <osg/Material>
#include <osgText/Text>
#include <osg/AlphaFunc>
#include <osg/BlendFunc>

#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coPopupHandle.h>

namespace opencover
{
class SystenCover;
}

using namespace covise;
using namespace opencover;
class comb;
class hiveLayer;

class Bee : public coVRPlugin
{
private:
protected:
    osg::Geode *geode;

public:
    static Bee *plugin;
    enum position
    {
        North = 0,
        NorthEast,
        SouthEast,
        South,
        SouthWest,
        NorthWest,
        Top,
        Bottom,
        NoFreePosition
    };
    osg::Vec3Array *offsets;

    Bee();
    virtual ~Bee();
    bool init();

    std::list<hiveLayer *> hiveLayers;

    osg::Vec3Array *vert;
    osg::DrawArrayLengths *primitives;
    osg::Vec4Array *colArr;
    osg::Vec3Array *normalArr;
    osg::Geometry *geom;

    // this will be called in PreFrame
    void preFrame();
    int numCombs;
    int addComb(comb *c, position p, comb *newc);
    void setHeight(int n, float h);
    float getHeight(int n);
    void setCap(int n, float c);
    float growSpeed;

    virtual bool destroy();
};

class comb
{
public:
    comb(comb *c, Bee::position p);
    ~comb();
    comb *neighbors[8];
    int number;
    float targetHeight;
    void update();
    int getNumNeighbors() const;
    comb *getNeighborComb(Bee::position p, int i);
    Bee::position freePosition();
};

class hiveLayer
{
public:
    hiveLayer(comb *c, Bee::position p);
    ~hiveLayer();
    comb *root;
    std::list<comb *> growing; // combs growing in height
    std::list<comb *> all;
    std::list<comb *> outerCombs; // combs that have empty neighbor positions
    hiveLayer *nextLayer;
    void update();
    void addNew(int num);
};

#endif
