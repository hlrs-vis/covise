/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VariantPlugin_H
#define _VariantPlugin_H
/****************************************************************************\
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Varant plugin                                               **
 **                                                                          **
 this plugin uses the "VariantPlugin" attribute, setted from the VarianMarker module
 to show/hide several VariantPlugins in the cover menu (VariantPlugins item)
 **                                                                          **
 ** Author: A.Gottlieb                                                       **
 **                                                                          **
 ** History:                                                                 **
 ** Jul-09  v1                                                               **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#ifdef VRUI
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coLabelMenuItem.h>
#else
#include <cover/ui/Owner.h>
#include <cover/ui/Menu.h>
#endif
#include <cover/coVRSelectionManager.h>
#include <util/coExport.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <cover/coVRLabel.h>

#include <osg/NodeVisitor>
#include "Variant.h"
#include <QtCore>
#include <qdom.h>
#include <QDir>
#include "coVRBoxOfInterest.h"
#include <config/CoviseConfig.h>
#include <cover/MarkerTracking.h>

using namespace covise;
using namespace opencover;
//------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

class VariantMarker
{
public:
    VariantMarker(std::string entryName);
    ~VariantMarker();
    std::set<MarkerTrackingMarker *> markerSet;
    std::string markerName;
    std::string variants;
    std::set<std::string> variantSet;
    float scale = -1.0;
};

class VariantPlugin : public coVRPlugin, public ui::Owner, /*public coMenuListener,*/ public coTUIListener, public coSelectionListener
{
    friend class mySensor;
public:
    static VariantPlugin *plugin;

    VariantPlugin();
    ~VariantPlugin() override;

    void preFrame() override;
    //this will be called by changing the selected object
    virtual bool selectionChanged() override;
    virtual bool pickedObjChanged() override;

#ifdef VRUI
    void menuEvent(coMenuItem *menu_VariantPluginitem);
#endif
    // this will be called if a COVISE object arrives
    bool init() override;
    void addNode(osg::Node *, const RenderObject *) override;
    void removeNode(osg::Node *node, bool /*isGroup*/, osg::Node *realNode) override;
    void message(int toWhom, int type, int len, const void *buf) override;
    void setMenuItem(Variant *var, bool state);
    void tabletEvent(coTUIElement *) override;

    void updateTUItemPos();

    osg::BoundingBox getBoundingBox(osg::Node *node);
    void clearTranslations(TRANS dir);
    void makeTranslations(TRANS dir);
    float getTypicalSice();

    osg::Vec3d createTransVec(Variant *var, TRANS dir);
    void hideAllLabel();
    void showAllLabel();
    int saveXmlFile();
    int readXmlFile();
    void setQDomElemState(Variant *var, bool state);
    void setQDomElemTRANS(Variant *var, osg::Vec3d vec);
    void setQDomElemLabels(bool state);
    int parseXML(QDomDocument *qxmlDoc);
    Variant *getVariant(std::string varName);
    void setVariant(std::string var);
    Variant *getVariant(osg::Node *varNode);
    Variant *getVariantbyAttachedNode(osg::Node *node);

    void printMatrix(osg::Matrix ma);
    void HideAllVariants();

private:
    coSensorList sensorList;
#ifdef VRUI
    coMenu *cover_menu;
    coSubMenuItem *button;
    coRowMenu *variant_menu;
    coSubMenuItem *variants;
    coSubMenuItem *options;
    coRowMenu *options_menu;
    coCheckboxMenuItem *showHideLabels;
    coSubMenuItem *roi; //Region of Interest
    coRowMenu *roi_menu;
    coCheckboxMenuItem *define_roi;
    coCheckboxMenuItem *active_roi;
#else
    ui::Menu *variant_menu=nullptr;
    ui::Menu *options_menu=nullptr;
    ui::Button *showHideLabels=nullptr;
    ui::Menu *roi_menu=nullptr;
    ui::Button *define_roi=nullptr;
    ui::Button *active_roi=nullptr;
#endif

    coTUITab *VariantPluginTab;
    coTUIToggleButton *VariantPluginTUIItem;
    coTUIComboBox *VariantPluginTUIcombo;
    coTUIFileBrowserButton *saveXML;
    coTUIFileBrowserButton *readXML;
    coTUIToggleButton *tui_showLabel;

    std::map<std::string, coTUIToggleButton *> tui_header_trans;

    std::list<Variant *> varlist;
    std::list<VariantMarker> variantMarkers;
    const VariantMarker *activatedMarker = nullptr;

    std::map<osg::Node *, Variant *> varmap;

    osg::BoundingBox box;

    QDomDocument *xmlfile;
    QDomElement qDE_Variant;

    coVRBoxOfInterest *boi;
    bool interActing;

    vrui::coTrackerButtonInteraction *_interactionA; ///< interaction for first button

    osg::Vec3 tmpVec;
    float scale;
    bool firsttime;
};
//------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

class VariantPluginNodeVisitor : public osg::NodeVisitor
{
public:
    VariantPluginNodeVisitor(coVRPlugin *plugin, osg::BoundingBox *box);

    virtual ~VariantPluginNodeVisitor(){};

    virtual void apply(osg::Node &node);

private:
    coVRPlugin *plugin;
    osg::BoundingBox *bbox;
};
#endif

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
