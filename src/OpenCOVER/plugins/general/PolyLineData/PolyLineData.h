/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * PolyLineData.h
 *
 *  Created on: 30.01.2009
 *      Author: Lukas Pinkowski
 */

#ifndef POLYLINEDATA_H_
#define POLYLINEDATA_H_

#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>

#include <OpenVRUI/coMenu.h>

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Array>

namespace osg
{
class Geode;
class Geometry;
class DrawArrays;
class Material;
class StateSet;
}

using namespace vrui;
using namespace opencover;

class PolyLineDataPlugin : public coVRPlugin,
                           public coMenuListener,
                           public coTUIListener
{
public:
    static PolyLineDataPlugin *plugin;

    PolyLineDataPlugin();
    virtual ~PolyLineDataPlugin();

    void tabletEvent(coTUIElement *);
    void tabletPressEvent(coTUIElement *);
    void tabletReleaseEvent(coTUIElement *);

    void preFrame();
    virtual void menuEvent(coMenuItem *menuItem);
    void drawInit();

    void message(int toWhom, int type, int len, const void *buf);

    bool init();

private:
    std::vector<osg::Vec3Array *> points;
    std::vector<osg::Geometry *> lineGeosets;
    std::vector<osg::DrawArrays *> primitives;

    osg::Material *mtl;
    osg::StateSet *linegeostate;
    osg::Vec4Array *fgcolor;
    osg::Geode *geode;

    void updateData();
    void updateData(const char *data, int len);

    void addGeo();
};

#endif /* POLYLINEDATA_H_ */
