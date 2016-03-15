/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef _VOLUME_PLUGIN_H
#define _VOLUME_PLUGIN_H

#include <OpenVRUI/coMenu.h>

#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>

#include <cover/coVRTui.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coCollabInterface.h>

#include <cover/coVRPlugin.h>

#include <virvo/vvvecmath.h>
#include <virvo/vvtransfunc.h>

#include <virvo/osg/VolumeDrawable.h>

#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osg/Geometry>
#include <list>
#include <cover/coVRPlugin.h>

#include <boost/scoped_ptr.hpp>

namespace covise
{
class coDoGeometry;
class coDistributedObject;
}

namespace vrui
{
class coSubMenuItem;
class coCheckboxMenuItem;
class coSliderMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coLabelMenuItem;
class coMenuItem;
class coCombinedButtonInteraction;
}

namespace opencover
{
class coVRPlugin;
class RenderObject;
}

using namespace vrui;
using namespace opencover;
using boost::scoped_ptr;

#define INITIAL_FPS 20.0f // initially desired frame rate
#define MIN_QUALITY 0.05f // smallest valid quality
#define MAX_QUALITY 4.0f // highest valid quality

#define INITIAL_POS_X 0.0f // initial object position in mm
#define INITIAL_POS_Y 0.0f
#define INITIAL_POS_Z 0.0f

class vvVolDesc;
class coDefaultFunctionEditor;
class FileEntry;

extern scoped_ptr<coCOIM> VolumeCoim;

/** Plugin for volume rendering.
  <BR>(C)2002-2005 RUS, 2004-2008 ZAIK
  @author Uwe Woessner
  @author Juergen Schulze-Doebold
  @author Martin Aumueller
*/
class VolumePlugin : public coMenuListener, public coVRPlugin, public coTUIListener
{
public:
    VolumePlugin();
    virtual ~VolumePlugin();
    static VolumePlugin *plugin;
    bool init();
    void preFrame();
    void message(int, int, const void *);
    void addObject(RenderObject *,
                   RenderObject *, RenderObject *,
                   RenderObject *, RenderObject *,
                   osg::Group *parent, int,
                   int, int, float *, float *, float *, int *,
                   int, int, float *, float *, float *, float);
    void removeObject(const char *, bool);
    void postFrame();
    void setTimestep(int);
    void key(int type, int keySym, int mod);
    bool updateVolume(const std::string &name, vvVolDesc *vd, bool mapTF = true, const std::string &filename = std::string());
    void saveVolume();
    void cropVolume();

    //tablet UI listener
    void tabletPressEvent(coTUIElement *tUIItem);
    void tabletEvent(coTUIElement *tUIItem);
    //void tabletDataEvent(coTUIElement* tUIItem, TokenBuffer &tb);

    static int loadVolume(const char *, osg::Group *parent, const char *ck = "");
    static int unloadVolume(const char *, const char *ck = "");

    scoped_ptr<coMenu> filesMenu;

private:
    enum
    {
        BgDefault,
        BgDark,
        BgLight
    } backgroundColor;
    std::list<FileEntry *> fileList;

    osg::Matrix invStartMove;
    osg::Matrix startBase;
    osg::Vec3 startPointerPosWorld;

    coDefaultFunctionEditor *editor; ///< transfer function editor

    /// The TabletUI Interface
    coTUIFunctionEditorTab *functionEditorTab;

    double start;

    scoped_ptr<coMenu> volumeMenu;
    scoped_ptr<coSubMenuItem> pinboardEntry;
    scoped_ptr<coSubMenuItem> filesItem;
    scoped_ptr<coCheckboxMenuItem> ROIItem;
    scoped_ptr<coCheckboxMenuItem> clipModeItem;
    scoped_ptr<coCheckboxMenuItem> preintItem;
    scoped_ptr<coCheckboxMenuItem> lightingItem;
    scoped_ptr<coSliderMenuItem> fpsItem;
    scoped_ptr<coCheckboxMenuItem> boundItem;
    scoped_ptr<coCheckboxMenuItem> interpolItem;
    scoped_ptr<coPotiMenuItem> colorsItem;
    scoped_ptr<coButtonMenuItem> cropItem;
    scoped_ptr<coButtonMenuItem> saveItem;
    scoped_ptr<coButtonMenuItem> tfeItem;
    scoped_ptr<coSliderMenuItem> hqItem;
    scoped_ptr<coCheckboxMenuItem> hqEnableItem;
    scoped_ptr<coCheckboxMenuItem> allVolumesActiveItem;
    scoped_ptr<coButtonMenuItem> cycleVolumeItem;
    scoped_ptr<coLabelMenuItem> currentVolumeItem;
    scoped_ptr<coCheckboxMenuItem> sideBySideItem;
    scoped_ptr<coButtonMenuItem> unloadItem;

    scoped_ptr<coSubMenuItem> blendModeItem;
    scoped_ptr<coMenu> blendModeMenu;
    scoped_ptr<coCheckboxMenuItem> alphaDefBlendItem;
    scoped_ptr<coCheckboxMenuItem> alphaDarkBlendItem;
    scoped_ptr<coCheckboxMenuItem> alphaLightBlendItem;
    scoped_ptr<coCheckboxMenuItem> minIntensityItem;
    scoped_ptr<coCheckboxMenuItem> maxIntensityItem;

    scoped_ptr<coSubMenuItem> multiChanItem;
    scoped_ptr<coMenu> multiChanMenu;
    scoped_ptr<coCheckboxMenuItem> multiDimTFItem;
    scoped_ptr<coCheckboxMenuItem> singleChannelItem;

    float lastRoll;
    float roiCellSize;
    float roiMaxSize;
    int discreteColors;
    bool instantMode; ///< true = instant classifications possible
    bool highQuality; ///< changing to highQuality rendering (until a mousebutton is pressed) is allowed
    bool switchToHighQuality; ///< high quality to be enabled via keyboard
    bool highQualityEnabled; ///< high quality rendering currently active
    float highQualityOversampling; ///< oversampling for hq mode
    float currentQuality; ///< current volume sampling
    bool roiMode;
    bool unregister;
    bool allVolumesActive;
    struct ROIData
    {
        float x;
        float y;
        float z;
        float size;
    };

    struct Volume
    {
        bool inScene;
        osg::ref_ptr<osg::MatrixTransform> transform; ///< transform node for side-by-side display
        osg::ref_ptr<osg::Geode> node; ///< node volume is attached to
        osg::ref_ptr<virvo::VolumeDrawable> drawable;
        void addToScene();
        osg::Geode *createImage(string &);
        void removeFromScene();
        osg::Vec3 min, max;
        osg::Vec3 roiPosObj;
        float roiCellSize;
        bool roiMode;
        bool preIntegration;
        bool lighting;
        bool interpolation;
        bool boundaries;
        virvo::VolumeDrawable::BlendMode blendMode;
        std::string filename;
        std::vector<vvTransFunc> tf;
        bool mapTF;
        int curChannel;
        bool multiDimTF;

        Volume();
        ~Volume();
    };
    typedef std::map<std::string, Volume> VolumeMap;
    VolumeMap volumes;
    VolumeMap::iterator currentVolume;
    vvVolDesc *volDesc;
    bool reenableCulling;
    uchar *tfeBackgroundTexture;
    void makeVolumeCurrent(VolumeMap::iterator vol);

    static void applyDefaultTransferFunction(void *userData);
    void applyAllTransferFunctions(void *userData);
    static void saveDefaultTransferFunction(void *userData);
    static void loadDefaultTransferFunction(void *userData);
    static void changeChannelFunction(int oldChan, int newChan, void *userData);
    bool pointerInROI(bool *wasMousePointer);
    bool roiVisible();
    int loadFile(const char *, osg::Group *parent);
    void sendROIMessage(osg::Vec3, float);
    void menuEvent(coMenuItem *);

    void setROIMode(bool);
    void setClippingMode(bool);
    coCombinedButtonInteraction *interactionA; ///< interaction for first button
    coCombinedButtonInteraction *interactionB; ///< interaction for second button
    coCombinedButtonInteraction *interactionHQ; ///< interaction for HQ mode
    int fpsMissed;
    float chosenFPS;

    struct TFApplyCBData
    {
        Volume *volume;
        virvo::VolumeDrawable *drawable;
        coDefaultFunctionEditor *tfe;
    };

    TFApplyCBData tfApplyCBData;

    virvo::VolumeDrawable *getCurrentDrawable();

    void updateTFEData();
    bool computeHistogram;
};

/** File entry for volume rendering plugin.
  @see VolumePlugin
*/
class FileEntry
{
public:
    char *menuName;
    char *fileName;
    coMenuItem *fileMenuItem;

    FileEntry(const char *, const char *);
    ~FileEntry();
};
#endif
