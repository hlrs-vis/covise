/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RecordPath_PLUGIN_H
#define _RecordPath_PLUGIN_H
/****************************************************************************\
**                                                            (C)2005 HLRS  **
**                                                                          **
** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** April-05  v1	    				       		                         **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include <cover/coTabletUI.h>
#include <PluginUtil/coColorMap.h>

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>
#include <PluginUtil/coSphere.h>
#include <array>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Label.h>
#include <cover/ui/FileBrowser.h>

constexpr size_t MAXSAMPLES = 1200;

using namespace covise;
using namespace opencover;

class CNCPlugin : public coVRPlugin, public ui::Owner
{
public:
    CNCPlugin();
    virtual ~CNCPlugin();
    static CNCPlugin *instance();
    bool init();


    int loadGCode(const char *filename, osg::Group *loadParent);
    static int sloadGCode(const char *filename, osg::Group *loadParent, const char *covise_key);
    static int unloadGCode(const char *filename, const char *covise_key);

    void straightFeed(double x, double y, double z, double a, double b, double c, double feedRate);
private:

    // this will be called in PreFrame
    void preFrame();
    ui::Menu *PathTab = nullptr;
    ui::Button*record = nullptr, *playPause = nullptr;
    ui::Action *reset = nullptr, *saveButton = nullptr;
    ui::Button *viewPath = nullptr, *viewlookAt = nullptr, *viewDirections = nullptr;
    ui::Label *numSamples = nullptr;
    ui::EditField *recordRateTUI = nullptr;
    ui::EditField *lengthEdit = nullptr;
    ui::EditField*radiusEdit = nullptr;
    ui::FileBrowser *fileNameBrowser = nullptr;
    ui::SelectionList *renderMethod = nullptr;
    covise::ColorMapSelector colorMap;

    osg::ref_ptr<osg::StateSet> geoState;
    osg::ref_ptr<osg::Material> linemtl;
    osg::ref_ptr<osg::LineWidth> lineWidth;
    void setTimestep(int t) override;

    osg::Vec4 getColor(float pos);
    int frameNumber = 0;
    osg::Group *parentNode = nullptr;
    osg::Vec3Array *vert = nullptr;
    osg::Vec4Array *color = nullptr;
    osg::DrawArrayLengths *primitives = nullptr;

    static CNCPlugin *thePlugin;

    void save();

    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;
};
#endif
