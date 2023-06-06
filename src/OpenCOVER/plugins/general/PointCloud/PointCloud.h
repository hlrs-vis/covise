/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NEWPROJECTNAME_PLUGIN_H_
#define _NEWPROJECTNAME_PLUGIN_H_

#include <cover/coVRPlugin.h>
#include <vector>

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <config/CoviseConfig.h>
#include "Points.h"
//#include "PointCloudDrawable.h"
#include "PointCloudGeometry.h"
#include "PointCloudInteractor.h"
//#include "plugins/general/NurbsSurface/NurbsSurface.h"

#include "FileInfo.h"
#include <string>
#include <vrb/client/SharedState.h>
#include <osg/Switch>

namespace opencover {
namespace ui {
class Element;
class Group;
class Slider;
class Menu;
class Button;
}
}

using namespace opencover;
/** Plugin
  @author 
*/
class PointCloudPlugin : public coVRPlugin, public ui::Owner
{

    /** File entry class for Image Plugin
   **/
    class ImageFileEntry
    {
    public:
        string menuName;
        string fileName;
        ui::Element *fileMenuItem;

        ImageFileEntry(const char *menu, const char *file, ui::Element *menuitem)
        {
            menuName = menu;
            fileName = file;
            fileMenuItem = menuitem;
        }
    };

private:
    int num_points;
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    void createGeodes(osg::Group *, const std::string &);
    int pointSetSize;
    PointSet *pointSet = nullptr;
    osg::Vec3 vecBase;
    std::vector<ImageFileEntry> pointVec;
    void clearData();
    void selectedMenuButton(ui::Element *);
    void readMenuConfigData(const char *, std::vector<ImageFileEntry> &, ui::Group *);
    bool intensityOnly;
    float intensityScale;
    bool intColor;
    bool polar;
    
    float pointSizeValue = 4;
    bool m_usePoitSprites=true;
    
    bool adaptLOD = true; // LOD enable/disable
    bool updateLOD = false; // update LOD state in next preFrame()
    float lodScale = 1.f; // levelOfDetail is multiplied by loadScale for each node
    float lodFarDistance = 40; // distance at which lod is set to minimum (25...inf)
    float lodNearDistance = 15; // distance until which lod is set to maximum (0...15)
    
    static PointCloudInteractor *s_pointCloudInteractor;
    static PointCloudInteractor *secondary_Interactor;
    std::vector<ScannerPosition> positions;
    void message(int toWhom, int type, int len, const void *buf); ///< handle incoming messages
    void calcMinMax(PointSet& pointSet);
    void addButton(FileInfo &fInfo);
    void saveMoves();
    string FileToMove = "";

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
    //ui::Group *fileGroup = nullptr;
    ui::Group *loadGroup = nullptr;
    ui::Group *selectionGroup = nullptr;
    ui::Button *singleSelectButton = nullptr;
    ui::Button *translationButton = nullptr;
    ui::Button *rotPointsButton = nullptr;
	ui::Button *rotAxisButton = nullptr;
    ui::Button *moveButton = nullptr;
	ui::Button *saveButton = nullptr;
	ui::Button *fileButton = nullptr;
    ui::Button *deselectButton = nullptr;
    ui::Button *createNurbsSurface = nullptr;
    //ui::Button *deleteButton = nullptr;
    ui::ButtonGroup *selectionButtonGroup = nullptr;
    ui::ButtonGroup *fileButtonGroup = nullptr;
    ui::Group *viewGroup = nullptr;
    ui::Button *adaptLODButton = nullptr;
    ui::Slider *pointSizeSlider = nullptr;

    ui::Slider* lodFarDistanceSlider = nullptr;
    ui::Slider* lodNearDistanceSlider = nullptr;


    
    //NurbsSurface *nurbsSurface = nullptr;

    void changeAllLOD(float lod);
    void changeAllPointSize(float pointSize);
    void UpdatePointSizeValue(void);
public:
    PointCloudPlugin();
    ~PointCloudPlugin();
	std::vector<FileInfo> files;
    bool init();
    void preFrame();
    void postFrame();
	inline float pointSize()
	{
		return pointSizeValue;
	}
    inline bool usePointSprites()
    {
        return m_usePoitSprites;
    }
    static int loadPTS(const char *filename, osg::Group *loadParent, const char *covise_key);
    static int unloadPTS(const char *filename, const char *covise_key);
    int unloadFile(const std::string &filename);
    static PointCloudPlugin *plugin;
    ui::Group *FileGroup;
	std::vector<pointSelection>& getInteractor();
};

#endif
