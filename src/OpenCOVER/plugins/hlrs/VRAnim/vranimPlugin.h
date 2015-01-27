/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRANIM_PLUGIN_H
#define _VRANIM_PLUGIN_H

//          scene
//           |
//          xform
//           |
//          scale
//           |
//          objectsRoot
//           |
//          multiBodyRoot_
//           |
//          glToPfTransf_
//           |
//       bodyTransform_   bodyTransform_   bodyTransform_   ...
//       bodyGeometry_    bodyGeometry_    bodyGeometry_    ...
//

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coPopupHandle.h>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Array>
#include <PluginUtil/coSphere.h>
#include <cover/coVRPlugin.h>
#include "coPlotItem.h"

#define VRANIM_MAXNOMENUITEMS 10

using namespace opencover;
using namespace covise;

namespace covise
{
class coSubMenuItem;
class coRowMenu;
class coCheckboxMenuItem;
class coCheckboxGroup;
class coButtonMenuItem;
class coSliderMenuItem;
}

class MultiBodyPlugin : public coVRPlugin, public coMenuListener, public coTUIListener
{
private:
    //scenegraph
    osg::ref_ptr<osg::Group> vranimroot_;
    osg::ref_ptr<osg::MatrixTransform> vranimglobalTransform_;
    osg::ref_ptr<osg::MatrixTransform> vranimfixedTransform_;
    osg::ref_ptr<osg::Group> vranimallbodiesGroup_;

    osg::ref_ptr<osg::Group> vranimbodyGroup_;
    osg::ref_ptr<osg::Group> vranimelbodyGroup_;
    osg::ref_ptr<osg::Group> vranimballGroup_;
    osg::ref_ptr<osg::Group> vranimivGroup_;
    osg::ref_ptr<osg::Group> vranimsensorGroup_;
    osg::ref_ptr<osg::Group> vranimlinElGroup_;

    osg::ref_ptr<osg::MatrixTransform> *vranimbodyTransform_; // numBodies transform nodes

    osg::ref_ptr<osg::Geode> *vranimbodyGeometry_; // numBodies geometry nodes
    osg::ref_ptr<osg::Geode> *vranimbodywireGeometry_;

    osg::ref_ptr<osg::MatrixTransform> vranimbodycsGeometry_; //? falsch? //besser osg::ref_ptr<osg::Geode> vranimbodycsGeometry_;  //?

    osg::ref_ptr<osg::Geode> *vranimelbodyGeometry_;
    osg::ref_ptr<osg::Geode> *vranimelbodywireGeometry_;

    osg::ref_ptr<osg::MatrixTransform> *vranimballTransform_; // numBodies transform nodes
    osg::ref_ptr<osg::Geode> *vranimballGeometry_;

    osg::ref_ptr<osg::MatrixTransform> *vranimivTransform_; // numBodies transform nodes
    osg::ref_ptr<osg::Node> *vranimivGeometry_;

    osg::ref_ptr<osg::MatrixTransform> *vranimlinElTransform_; // numBodies transform nodes
    osg::ref_ptr<osg::Geode> *vranimlinstdGeometry_;
    osg::ref_ptr<osg::Node> *vranimlinivGeometry_;

    osg::ref_ptr<osg::Geode> *vranimsensorGeometry_; // numBodies geometry nodes
    osg::ref_ptr<osg::TessellationHints> hint; //? wozu?

    // menus
    coSubMenuItem *multiBodyMenuButton_; // button in COVER main menu
    coRowMenu *multiBodyMenu_; // the multibody menu

    coSubMenuItem *anim_shaderadio_Button_;
    coRowMenu *anim_shaderadio_Menu_;
    coCheckboxMenuItem *anim_shadewire_Checkbox_;
    coCheckboxMenuItem *anim_shadeflexwire_Checkbox_;

    coCheckboxGroup *anim_shade_Radio_Group_;
    coCheckboxMenuItem *anim_shadeoff_Checkbox_;
    coCheckboxMenuItem *anim_shadeunlighted_Checkbox_;
    coCheckboxMenuItem *anim_shadeflat_Checkbox_;
    coCheckboxMenuItem *anim_shadegouraud_Checkbox_;

    coCheckboxGroup *anim_shadeflex_Radio_Group_;
    coCheckboxMenuItem *anim_shadeflexoff_Checkbox_;
    coCheckboxMenuItem *anim_shadeflexunlighted_Checkbox_;
    coCheckboxMenuItem *anim_shadeflexflat_Checkbox_;
    coCheckboxMenuItem *anim_shadeflexgouraud_Checkbox_;

    coButtonMenuItem *anim_interval_Button_;
    coButtonMenuItem *anim_calcstride_Button_;

    coButtonMenuItem *anim_savetrafo_Button_;
    coCheckboxMenuItem *anim_showcoordsystem_Checkbox_;
    coCheckboxMenuItem *anim_showsensors_Checkbox_;
    coCheckboxMenuItem *anim_showplotters_Checkbox_;

    coSubMenuItem *anim_hideradio_Button_;
    coRowMenu *anim_hideradio_Menu_;
    coCheckboxMenuItem *anim_nohide_Checkbox_;
    coCheckboxMenuItem **anim_hide_Checkbox_;

    coSubMenuItem *anim_fixmotionradio_Button_;
    coRowMenu *anim_fixmotionradio_Menu_;
    coCheckboxMenuItem *anim_nofixmotion_Checkbox_;
    coCheckboxGroup *anim_fixmotion_Radio_Group_;
    coCheckboxMenuItem **anim_fixmotion_Checkbox_;

    coSubMenuItem *anim_fixtranslationradio_Button_;
    coRowMenu *anim_fixtranslationradio_Menu_;
    coCheckboxMenuItem *anim_nofixtranslation_Checkbox_;
    coCheckboxGroup *anim_fixtranslation_Radio_Group_;
    coCheckboxMenuItem **anim_fixtranslation_Checkbox_;

    //tabletUI
    coTUITab *vranimTab;
    coTUILabel *infoLabel;
    coTUIFrame *leftFrame;
    coTUIFrame *rightFrame;

    coTUIFrame *bodiesFrame;
    coTUILabel *hideLabel;
    coTUILabel *fixMotionLabel;
    coTUILabel *fixTranslationLabel;
    coTUILabel *noBodyLabel;
    coTUIToggleButton *doNotHideButton;
    coTUIToggleButton *doNotFixRadioButton;
    coTUILabel **bodyLabel;
    coTUIToggleButton **hideBodyButton;
    coTUIToggleButton **fixMotionButton;
    coTUIToggleButton **fixTranslationButton;

    coTUIFrame *showFrame;
    coTUILabel *showLabel;
    coTUIToggleButton *showSensorsButton;
    coTUIToggleButton *showPlottersButton;
    coTUIToggleButton *showCoordSystemsButton;

    coTUIFrame *shadingFrame;
    coTUILabel *shadingLabel;
    coTUIComboBox *rigidComboBox;
    coTUIToggleButton *rigidBodiesAsWireButton;
    coTUIComboBox *flexComboBox;
    coTUIToggleButton *flexBodiesAsWireButton;

    coTUIFrame *trafoFrame;
    coTUIButton *saveTrafoButton;

    // misc
    osg::Geode *createBodyGeometry(int bodyId, int what);
    osg::Geode *createelBodyGeometry(int bodyId, int what);
    osg::Geode *createSensorGeometry(int SensorId);
    osg::Geode *createBallGeometry(int ballId);
    osg::Geode *createlinstdGeometry(int linElId);

    coSphere *sphere;

    void update(void);
    void updateRigidTransform(void);
    void updateBallTransform(void);
    void updateIvTransform(void);
    void updateElast(void);
    void updateSensor(void);
    void updateLineElements(void);
    void updatePlotter(void);
    void updatefixed(void);
    void updateDynColors(void);

    static double updatetime_;

    void saveTrafo(void);
    void doNotFix(void);

public:
    coPopupHandle **plotHandle;
    coPlotItem **plotItem;

    static MultiBodyPlugin *plugin_;
    static int debugLevel_;

    static int unloadDyn(const char *filename, const char *);
    static int loadDyn(const char *filename, osg::Group *parent, const char *);
    int loadFile(const char *dir, const char *basename, osg::Group *parent);
    int unloadFile();

    void setTimestep(int t);

    // constructor
    MultiBodyPlugin();

    // destructor
    virtual ~MultiBodyPlugin();

    bool init();
    bool destroy();

    // loop
    void preFrame();
    void postFrame();

    void key(int type, int keySym, int mod);

    void menus_create();
    void menus_delete();
    void scenegr_create();
    void scenegr_delete();

    //tabletUI
    void vranimtab_create();
    void vranimtab_delete();

    osg::Geode *createlinElGeometry(int linElId);

    // menu event for general items
    virtual void menuEvent(coMenuItem *menuItem);

    void setDefaultMaterial(osg::StateSet *geoState, bool transparent);

    //tabletUI
    void tabletEvent(coTUIElement *);

    osg::ref_ptr<osg::Switch> *vranimbodygeoSwitch_;
    osg::ref_ptr<osg::Switch> *vranimbodywireSwitch_;
    osg::ref_ptr<osg::Switch> *vranimbodycsSwitch_;

    osg::ref_ptr<osg::Switch> *vranimelbodygeoSwitch_;
    osg::ref_ptr<osg::Switch> *vranimelbodywireSwitch_;

    osg::ref_ptr<osg::Switch> vranimsensorSwitch_;
    osg::ref_ptr<osg::Material> globalmtl;
    osg::ref_ptr<osg::StateSet> unlightedStateSet;
    osg::ref_ptr<osg::StateSet> flatStateSet;
    osg::ref_ptr<osg::StateSet> shadedStateSet;
    osg::Vec3Array **global_vert;
    osg::Vec4Array *colArr;

    void reset_timestep(void);
    void ballcolor(float *color, float fcolor);
};
#endif
