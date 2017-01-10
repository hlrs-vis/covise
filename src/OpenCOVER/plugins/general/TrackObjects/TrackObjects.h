/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _TrackObjects_PLUGIN_H
#define _TrackObjects_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: TrackObjects Plugin (does nothing)                          **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** Nov-01  v1	    				       		                             **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>
#include <cover/coInteractor.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coIconButtonToolboxItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <list>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/input/input.h>

using namespace vrui;
using namespace opencover;

class TObject
{
public:
    TObject(const std::string &n);
    virtual ~TObject();
    void setOffset(float x, float y, float z, float h, float p, float r);
    std::string name;
    coCoord offsetCoord;
    osg::Matrix offset;
};

class TrackObjects : public coVRPlugin, public coTUIListener, public coMenuListener
{
public:
    TrackObjects();
    virtual ~TrackObjects();
    bool init();

    // this will be called if an object with feedback arrives

    void menuEvent(coMenuItem *);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

    void preFrame();
    void updateTUI();

    TObject *getTObject(const std::string &name);

    std::list<coInteractor *> interactors;
    coButtonMenuItem *execButton;
    coIconButtonToolboxItem *ToolbarButton;
    coSubMenuItem *pinboardEntry;
    coRowMenu *TangibleSimulationMenu;

    coTUITab *TrackObjectsTab;
    coTUIToggleButton *trackObjects;
    coTUIButton *getOffset;
    coTUIButton *snap;
    coTUILabel *objectChoiceLabel;
    coTUIComboBox *objectChoice;
    coTUILabel *bodyChoiceLabel;
    coTUIComboBox *bodyChoice;

    TrackingBody *trackingBody; 

    coTUIEditFloatField *posX;
    coTUIEditFloatField *posY;
    coTUIEditFloatField *posZ;
    coTUIEditFloatField *rotH;
    coTUIEditFloatField *rotP;
    coTUIEditFloatField *rotR;
    TObject * currentObject;

    float x,y,z,h,p,r;
    std::vector<TObject*> TObjects;
    osg::Matrix oldMat;


private:
};
#endif
