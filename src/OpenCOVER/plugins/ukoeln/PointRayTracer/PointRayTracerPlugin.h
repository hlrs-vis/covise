/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POINT_RAY_TRACER_PLUGIN_H
#define POINT_RAY_TRACER_PLUGIN_H

#include "PointRayTracerGlobals.h"
#include "PointReader.h"
#include "PointRayTracerDrawable.h"

#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>

#include <cover/coVRPlugin.h>
#include <cover/coVRFileManager.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coButtonMenuItem;
}

namespace opencover
{
class coVRLabel;
}

using namespace vrui;
using namespace opencover;


class PointRayTracerPlugin : public coVRPlugin, public coMenuListener
{

public:
    static PointRayTracerPlugin *plugin;
    static PointRayTracerPlugin *instance();

    PointRayTracerPlugin();
    virtual ~PointRayTracerPlugin();

     bool init(); //called before files are loaded
     bool init2(); //called after all files have been loaded

    static int sloadPts(const char *filename, osg::Group *loadParent, const char *);
    static int unloadPts(const char *filename, const char *);
    int loadPts(const char* filename);

     void preFrame();
     void preDraw(osg::RenderInfo &info);
     void expandBoundingSphere(osg::BoundingSphere &bs);

     void key(int type, int keySym, int mod);

private:

    PointReader* m_reader;

    coSubMenuItem *prtSubMenuEntry;
    coRowMenu *prtMenu;
    coButtonMenuItem *nextItem;
    coButtonMenuItem *prevItem;


    void menuEvent(coMenuItem *menuItem);

    osg::ref_ptr<osg::Geode> m_geode;
    osg::ref_ptr<PointRayTracerDrawable> m_drawable;

    point_vector                m_points;
    visionaray::aabb            m_bbox;

    std::vector<host_bvh_type>  m_bvh_vector;

    float                       m_pointSize;
    bool                        m_cutUTMdata;
    bool                        m_useCache;

    int                         m_numPointClouds;
    int                         m_currentPointCloud;
    bool                        m_currentPointCloud_has_changed;
    void                        showNextPointCloud();
    void			showPreviousPointCloud();

    /*
    bool                        m_visibility_has_changed;
    std::vector<bool>           m_visibility_vector;

    void toggleVisibility(int index);
    */
};


#endif //POINT_RAY_TRACER_PLUGIN_H
