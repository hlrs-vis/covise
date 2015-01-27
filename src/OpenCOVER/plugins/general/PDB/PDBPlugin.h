/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   PDB databank protein loader
//
// Author:        Philip Weber
//
// Creation Date: 2005-11-29
//
// **************************************************************************

#ifndef PDB_PLUGIN_H
#define PDB_PLUGIN_H

// gcc:
#include <vector>
#ifndef WIN32
#include <sys/time.h>
#else
#include <sys/timeb.h>
#endif

// Covise:
#include <OpenVRUI/coMenu.h>
#include <cover/coVRPlugin.h>
// OSGcaveui
#include <osgcaveui/Marker.h>

// Openscenegraph:
#include <osg/Group>
#include <osg/Switch>
#include <osg/AutoTransform>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/BlendEquation>
#include <osg/AlphaFunc>
#include <osg/BoundingBox>

// Local:
#include "HighDetailTransVisitor.h"
#include "LowDetailTransVisitor.h"
#include "FrameVisitor.h"
#include "ComputeBBVisitor.h"
#include "GeodeCountVisitor.h"
#include "TopsanViewer.h"
#include "SequenceViewer.h"
#include "coFileBrowser.h"
#include "SizedPanel.h"

#define LOAD_FILE 150
//#define LOAD_ANI 151
#define FRAME_UPDATE 152
#define CLEAR_ALL 153
#define HIGH_DETAIL 154
#define ADJUST_HUE 155
#define MOVE 156
#define STOP_MOVE 157
#define SCALE_PROTEIN 158
#define RESET_SCALE 159
#define REMOVE_PROTEIN 160
#define SCALE_ALL 161
#define VIEW 162
#define MARK 163
#define PDB_ALIGN 165
#define LAYOUT_GRID 166
#define LAYOUT_CYL 167

// communications with PDBSequence

namespace vrui
{
class coSlider;
class coFrame;
class coPanel;
class coButtonMenuItem;
class coSubMenuItem;
class coCheckboxMenuItem;
class coPopupHandle;
class coButton;
class coToggleButton;
class coPotiItem;
class coLabelItem;
}

namespace vrml
{
class VrmlScene;
}

namespace opencover
{
class coVRPlugin;
}

class ViewerOsg;
class SystemCover;
class AlignPoint;
class ReaderWriterIV;
class SequenceMarker;

using namespace vrui;
using namespace opencover;

/** Wireframe box around protein, helps select a protein.
*/
class PDBPickBox : public cui::PickBox
{
    friend class SequenceMarker;

public:
    static LowDetailTransVisitor *lowdetailVisitor;
    static HighDetailTransVisitor *highdetailVisitor;
    static FrameVisitor *frameVisitor;

    PDBPickBox(osg::Switch *, cui::Interaction *, const Vec3 &, const Vec3 &, const Vec4 &, const Vec4 &, const Vec4 &, string &, float, osg::Matrix * = NULL);
    virtual ~PDBPickBox();
    virtual void cursorEnter(cui::InputDevice *);
    virtual void cursorLeave(cui::InputDevice *);
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void buttonEvent(cui::InputDevice *, int);
    void setSequenceMarker(SequenceMarker *);
    SequenceMarker *getSequenceMarker();
    void addTimestep(osg::Group *);
    void setTimestep(int);
    int getNumTimesteps();
    //void setFade(bool);
    //bool getFade();
    //bool getDetailLevel();
    //void setDetailLevel(bool);
    //void computeDetailLevel();
    //void computeFadeLevel();
    string getName();
    bool isMoving();
    void resetScale();
    void resetOrigin();
    void setOrigin(osg::Vec3);
    osg::Group *copy();
    float getHue();
    void setHue(float);
    void overrideHue(bool);
    bool isOverrideColor();
    bool getLock(string);
    void unlock(string);
    void setView(string &, bool);
    bool isViewEnabled(const string &);
    void addChild(osg::Node *);
    void addMarker(osg::Node *);

    void setMorph(bool b);
    bool isMorph();
    void setVersionMorph(osg::Group *node);
    osg::Group *getVersionMorph(string &type);

    osg::Matrix getScaleMat()
    {
        return _scale->getMatrix();
    }
    osg::Matrix getNodeMat()
    {
        return _node->getMatrix();
    }
    void setScaleMat(osg::Matrix mat)
    {
        _scale->setMatrix(mat);
    }
    void setNodeMat(osg::Matrix mat)
    {
        _node->setMatrix(mat);
    }

    int uid;

protected:
    bool _isMoving; ///< true=left button down; moving data set
    bool _isMorph;
    //bool _isScaling;            ///< true=middle button down; scaling data set
    bool _isFading;
    bool _isHighDetail;
    float _defaultscale;
    string _name;
    osg::Switch *_mainSwitch;
    int _currentTimestep;
    osg::Vec3 _origin;
    osg::Vec3 boxCenter;
    osg::MatrixTransform *adjustmt;
    float _hue;
    bool _overridecolor;

private:
    string _lockname;
    SequenceMarker *_mark;
    osg::ref_ptr<osg::Group> _versions;
    void setVersion(osg::Node *);
    osg::Node *getVersion(string &);
};

struct PDBPluginMessage
{
    int token;
    char hostname[20];
    char filename[20];
    osg::Matrix trans;
    float hue;
    float scale;
    bool hue_mode_on;
    bool high_detail_on;
    int framenumber;
    char viewtype[20];
    bool view_on;
    bool mark_on;
    int uid;
};

/** Wireframe box of allignment markers
*/
class SequenceMarker : public cui::Marker
{

public:
    SequenceMarker(PDBPickBox *, GeometryType, cui::Interaction *, float, float);
    virtual ~SequenceMarker();
    virtual void cursorEnter(cui::InputDevice *);
    virtual void cursorLeave(cui::InputDevice *);
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void buttonEvent(cui::InputDevice *, int);
    void setVisible(bool);
    bool isVisible();
    osg::Node *getBase();
    PDBPickBox *getProtein();

protected:
    bool _isMoving; ///< true=left button down; moving data set

private:
    bool _visible;
    PDBPickBox *_protein;
    bool _selected;
    osg::Vec3 _basepoint;
};

class PDBPlugin;
class AlignPickBox;
/** align point used in galaxy mode
  */
class AlignPoint : public cui::PickBox
{
public:
    AlignPoint(cui::Interaction *, const Vec3 &, const Vec3 &, const Vec4 &, const Vec4 &, const Vec4 &, float, float, SequenceViewer *sview, PDBPlugin *pdbp);
    virtual ~AlignPoint();
    virtual void cursorEnter(cui::InputDevice *);
    virtual void cursorLeave(cui::InputDevice *);
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void buttonEvent(cui::InputDevice *, int);
    void setAlign(int index);
    void setVisible(bool);
    //void addChild(osg::Group*);
    //void addChild(osg::Node*, std::string);
    //void addChild(osg::Geode*);
    void removeChildren();
    void reset();

    void setMovable(std::string name, bool set);

protected:
    bool _isMoving; ///< true=left button down; moving data set
    std::vector<AlignPickBox *> pbvec;

private:
    float _hue;
    float _basehue;
    osg::MatrixTransform *_protein_size;
    osg::ShapeDrawable *_shapeDrawable;
    vector<osg::Group *> list;
    vector<std::string> namelist;
    string mammothpath, PDBpath;
    void setColor(Vec4);
    osg::Vec4 calculateColor(float);
    SequenceViewer *_sview;
    cui::Interaction *inter;
    PDBPlugin *_pdbp;
};

class AlignPickBox : public cui::PickBox
{
    friend class AlignPoint;

public:
    AlignPickBox(osg::MatrixTransform *node, cui::Interaction *, const Vec3 &, const Vec3 &, const Vec4 &, const Vec4 &, const Vec4 &);
    virtual ~AlignPickBox();
    osg::MatrixTransform *getRoot();
    void setColor(osg::Vec4 v);
    virtual void cursorEnter(cui::InputDevice *);
    virtual void cursorLeave(cui::InputDevice *);
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void buttonEvent(cui::InputDevice *, int);
    void resetPos();
    void setMovable(bool b);

protected:
    bool _isMoving;
    bool _isSelected;
    osg::Vec3 boxCenter;
    osg::Vec3 _origin;
    osg::MatrixTransform *rootmt;
    osg::MatrixTransform *_mainSwitch;
    osg::Vec4 color;
    osg::Vec3 boxSize;
};

struct PDBSavedLayout
{
    osg::Matrix objmat;
    float scale;
    std::vector<std::pair<osg::Matrix, osg::Matrix> > protmat;
    std::vector<std::string> protname;
};

/** Plugin to load PBD structures from the protein DataBank
 */
class PDBPlugin : public coMenuListener, public coButtonActor, public coValuePotiActor, public cui::coFileBrowserListener, public cui::PickBoxListener, public cui::MarkerListener, public coVRPlugin
{

    friend class SystemCover;
    friend class ViewerOsg;
    friend class AlignPoint;

    /** File entry class for PDB Plugin
   */
    class PDBFileEntry
    {
    public:
        string menuName;
        string fileName;
        coMenuItem *fileMenuItem;

        PDBFileEntry(const char *menu, const char *file, coMenuItem *menuitem)
        {
            menuName = menu;
            fileName = file;
            fileMenuItem = menuitem;
        }
    };

    std::vector<PDBPickBox *> _aniProteins;

    coCheckboxMenuItem *alVisibleCheckbox;
    std::vector<std::pair<coToggleButton *, coToggleButton *> > alButtons;
    coToggleButton *alreset, *alclear;
    coPopupHandle *alHandle;
    coFrame *alFrame;
    SizedPanel *alPanel;
    float alButtonWidth, alButtonHeight;

    struct PDBSavedLayout *layouts[10];
    coSubMenuItem *layoutMenuItem, *saveLayoutMenuItem, *loadLayoutMenuItem;
    coRowMenu *layoutMenu, *saveLayoutMenu, *loadLayoutMenu;
    coButtonMenuItem *saveLayoutButtons[10];
    coButtonMenuItem *loadLayoutButtons[10];
    bool loadinglayout;
    int loadnum, layoutnum;

    enum layoutenum
    {
        NONE,
        GRID,
        CYLINDER
    };

    enum alignenum
    {
        ADD,
        ALIGN_DELETE,
        DELETE_ALL
    };

    layoutenum lastlayout;

    coPotiMenuItem *radiusPoti, *proteinsPerLevelPoti;

    ViewerOsg *viewer;
    VrmlScene *vrmlScene;
    Player *player;

    // popup name panel
    coPanel *namePanel;
    coLabel *nameMessage;
    coPopupHandle *nameHandle;
    coFrame *nameFrame;

    // Main menu items:
    coButtonMenuItem *browserButton;
    coButtonMenuItem *clearSceneButton;
    coButtonMenuItem *layoutButtonGrid;
    coButtonMenuItem *layoutButtonCylinder;
    coButtonMenuItem *resetButton;
    coCheckboxMenuItem *nameVisibleCheckbox;
    coCheckboxMenuItem *topsanVisibleCheckbox;
    coCheckboxMenuItem *sviewVisibleCheckbox;

    // Panel to contain dials and load button
    coPanel *panel;

    // Frame to contain the panel
    coFrame *frame;
    coFrame *messageFrame;

    // Declare popup handle for frame
    coPopupHandle *handle;
    coPopupHandle *messageHandle;

    // update message screen
    coLabel *messageLabel;

    // menu digit dial options:
    coValuePoti **potiChar;

    // menu dial labels
    coLabel **labelChar;

    // load protein button
    coButton *loadButton;

    // add menus for structures and animations
    coSubMenuItem *selectMenu;
    coSubMenuItem *structureMenu;
    coSubMenuItem *animationMenu;
    coSubMenuItem *faderMenu;
    coRowMenu *selectRow;
    coRowMenu *structureRow;
    coRowMenu *animationRow;
    coRowMenu *faderRow;

    // buttons to append to the menus
    coCheckboxMenuItem *loaderMenuCheckbox;
    coCheckboxMenuItem *fadeonCheckbox;
    coCheckboxMenuItem *highdetailCheckbox;
    //coCheckboxMenuItem* alignCheckbox;
    coPotiMenuItem *distviewMenuPoti;
    coPotiMenuItem *fadeareaMenuPoti;
    coPotiMenuItem *scalePoti;
    coButtonMenuItem *faderesetButton;
    coCheckboxMenuItem *movableCheckbox;

    // sub menu buttons
    coPopupHandle *proteinHandle;
    coFrame *proteinFrame;
    coPanel *proteinPanel;
    coLabel *proteinLabel;
    coButton *proteinDeleteButton;
    coButton *alignButton;
    coButton *proteinResetScaleButton;
    coValuePoti *proteinScalePoti;
    coValuePoti *proteinHuePoti;
    coToggleButton *coloroverrideCheckBox;

    coLabel *surfaceLabel;
    coLabel *cartoonLabel;
    coLabel *stickLabel;
    coLabel *sequenceLabel;
    coLabel *ribbonLabel;
    coToggleButton *surfaceCheckBox;
    coToggleButton *cartoonCheckBox;
    coToggleButton *stickCheckBox;
    coToggleButton *sequenceCheckBox;
    coToggleButton *ribbonCheckBox;

    coLabel *coloroverrideLabel;

    // currently selected pickbox
    PDBPickBox *_selectedPickBox;
    PDBPickBox *_prevselectedPickBox;

    // string dial variables
    string **stringChar;

    // removing protein flag
    bool _removingProtein;

    // command buffer string
    string cmdInput;

    // file name
    string relativePymolPath;
    string relativeTempPath;
    string currentPath;
    string animationurl;
    string pdburl;

    bool loadingMorph; // indicate if a morph is loading
    bool _restrictedMode;
    bool _galaxyMode;
    int loadDelay;
    int screenResizeDelay;
    int currentFrame;

    // image hack for data
    osg::MatrixTransform *imageMat;
    osg::Geode *imageGeode;

    // fade settings
    float fadedist;
    float viewdist;

    // default marker size
    float _markerSize;

    // radius and file size (galaxy mode)
    float _filesize;
    float _radius;
    int _ringnum;
    int _maxstructures;

    // default scale
    float _defaultScale;

    string loadFile;

    // host name of this machine
    string myHost;

    // scale node for collabiration mode
    osg::ref_ptr<osg::AutoTransform> scaleNode;

    // switch node for attaching vrml scenes
    osg::ref_ptr<osg::Switch> mainNode;

    //declare lists for the dynamic menus
    std::vector<PDBFileEntry> structureVec;
    std::vector<PDBFileEntry> animationVec;

    //align point
    AlignPoint *alignpoint;
    osg::ref_ptr<osg::Node> tmpNode;

    enum DataBankType
    {
        UCSD,
        YALE
    };

    enum ConversionType
    {
        CT_HELIX,
        CT_CYLINDER
    };

    void menuEvent(coMenuItem *);
    void loadPDB(string &, DataBankType);
    void clearSwitch();
    bool downloadPDBFile(string &, DataBankType);
    void selectedMenuButton(coMenuItem *);
    void readMenuConfigData(const char *, std::vector<PDBFileEntry> &, coRowMenu &);
    void showMessage(const string &);
    void hideMessage();
    void cleanFiles(string &);
    void loadSingleStructure(string &);
    void loadAniStructures(string &);
    void loadWarehouseFiles(vector<string> &);
    void updateFadeValues();
    void updateOSGCaveUI();
    void removeProtein();
    void resetScale();
    void resetPositions();
    void calcTrans(int, int, float, osg::Vec3f &);
    void reloadProtein(PDBPickBox *);
    void stopmove(PDBPluginMessage *);
    void move(PDBPluginMessage *);
    void setHue(PDBPluginMessage *);
    void setView(PDBPluginMessage *);
    void setScale(PDBPluginMessage *);
    void resetScale(PDBPluginMessage *);
    void enableMark(PDBPluginMessage *);
    void resetAllScale();
    void removeProtein(PDBPluginMessage *);
    void setAllScale(PDBPluginMessage *);
    osg::Geode *createImage(string &);
    osg::Node *loadStructure(string &);
    void moveSequenceMarker(SequenceMessage *);
    //void alignProtein(PDBPluginMessage*);
    void alignProteins(int i);
    void clearAlign();
    void alignReset();
    void setSequenceMarkSizeAll(float);

    void makeLayout(int i);
    void loadLayout(int i);
    void loadLayoutHelper();
    void writeSavedLayout();
    void readSavedLayout();

    void updateAlignMenu(alignenum e, int uid);
    void updateAlignMenu();

    // avoid warnings
    int chdir(const char *path);
    int system(const char *string);

protected:
    cui::Interaction *_interaction;
    std::vector<PDBPickBox *> _proteins;
    TopsanViewer *topsan;
    SequenceViewer *sview;

    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);
    bool fileBrowserEvent(cui::coFileBrowser *, std::string &, std::string &, int, int);
    void setAllMovable(bool);
    bool loadPDBFile(std::string, Switch *);
    osg::Node *loadVRMLFile(std::string &);
    void initCoverAnimation();
    void updateNumTimesteps();
    void pickBoxButtonEvent(cui::PickBox *, cui::InputDevice *, int);
    void pickBoxMoveEvent(cui::PickBox *, cui::InputDevice *);
    void layoutProteinsCylinder();
    void layoutProteinsGrid();
    void readSetFile(std::string, vector<string> *);
    void markerEvent(cui::Marker *, int, int);

    int uidcounter;

public:
    PDBPlugin();
    bool init();
    virtual ~PDBPlugin();
    void buttonEvent(coButton *);
    void preFrame();
    void setTimestep(int ts);
    void setMMString(char *, string);
    string getMyHost();
    void setNamePopup(string);
    bool loadDataImage(string);
    string getRelativeTempPath();
    string getCurrentPath();
    void message(int, int, const void *);
};
#endif

// EOF
