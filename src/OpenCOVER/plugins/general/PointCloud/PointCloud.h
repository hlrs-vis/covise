/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NEWPROJECTNAME_PLUGIN_H_
#define _NEWPROJECTNAME_PLUGIN_H_

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coSquareButtonGeometry.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <cover/coVRPlugin.h>
#include <list>

#include <cover/coVRShader.h>
#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFrame.h>
#include <config/CoviseConfig.h>
#include "Points.h"
//#include "PointCloudDrawable.h"
#include "PointCloudGeometry.h"
#include "PointCloudInteractor.h"
#include <cover/coTabletUI.h>

namespace vrui
{
class coFrame;
class coPanel;
class coButtonMenuItem;
class coButton;
class coPotiItem;
class coLabelItem;
}

using namespace opencover;
using namespace vrui;


class nodeInfo
{
public:
    osg::Node *node;
};

class fileInfo
{
public:
    std::string filename;
    std::list<nodeInfo> nodes;
    int pointSetSize;
    PointSet *pointSet;
};

/** Plugin
  @author 
*/
class PointCloudPlugin : public coMenuListener, public coValuePotiActor, public coVRPlugin, public coTUIListener, public ui::Owner
{

    /** File entry class for Image Plugin
   **/
    class ImageFileEntry
    {
    public:
        string menuName;
        string fileName;
        coMenuItem *fileMenuItem;

        ImageFileEntry(const char *menu, const char *file, coMenuItem *menuitem)
        {
            menuName = menu;
            fileName = file;
            fileMenuItem = menuitem;
        }
    };

private:
    std::list<fileInfo> files;
    int num_points;
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    void createGeodes(osg::Group *, string &);
    int pointSetSize;
    PointSet *pointSet;
    osg::Vec3 vecBase;
    std::vector<ImageFileEntry> pointVec;
    void clearData();
    void selectedMenuButton(coMenuItem *);
    void readMenuConfigData(const char *, std::vector<ImageFileEntry> &, coRowMenu &);
    bool intensityOnly;
    float intensityScale;
    bool intColor;
    bool polar;
    float pointSizeValue;
    bool adaptLOD;
    coTUITab *PCTab;
    coTUIToggleButton *adaptLODTui;
    coTUIFloatSlider *pointSizeTui;
    static PointCloudInteractor *s_pointCloudInteractor;

protected:
    osg::MatrixTransform *planetTrans;

    //coSubMenuItem *imanPluginInstanceMenuItem;
    //coRowMenu *imanPluginInstanceMenu;
    // //coCheckboxMenuItem* enablePointCloudPlugin;
    //coRowMenu *loadMenu;
    //coSubMenuItem *loadMenuItem;
    //coButtonMenuItem *deleteMenuItem;
    ui::Menu *pointCloudMenu = nullptr;
    ui::Menu *loadMenu = nullptr;
    ui::Group *fileGroup = nullptr;
    ui::Group *selectionGroup = nullptr;
    ui::Button *singleSelectButton = nullptr;
    ui::Button *deselectButton = nullptr;
    //ui::Button *deleteButton = nullptr;
    ui::ButtonGroup *selectionButtonGroup = nullptr;
    ui::Group *viewGroup = nullptr;
    ui::Button *adaptLODButton = nullptr;
    ui::Slider *pointSizeSlider = nullptr;

    //void menuEvent(coMenuItem *);
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);

public:
    PointCloudPlugin();
    ~PointCloudPlugin();
    bool init();
    void preFrame();
    void postFrame();
    float pointSize()
    {
        return pointSizeValue;
    };
    static int loadPTS(const char *filename, osg::Group *loadParent, const char *covise_key);
    static int unloadPTS(const char *filename, const char *covise_key);
    int unloadFile(std::string filename);
    void tabletEvent(coTUIElement *);
    static PointCloudPlugin *plugin;
    ui::Group *FileGroup;
};

#endif
