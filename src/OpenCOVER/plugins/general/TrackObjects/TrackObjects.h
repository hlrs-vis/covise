/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TangiblePosition_PLUGIN_H
#define _TangiblePosition_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: TangiblePosition Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
#include "cover/coTabletUI.h"
#include "cover/coInteractor.h"
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coIconButtonToolboxItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <list>

using namespace vrui;
using namespace opencover;

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


    std::list<coInteractor *> interactors;
    coButtonMenuItem *execButton;
    coIconButtonToolboxItem *ToolbarButton;
    coSubMenuItem *pinboardEntry;
    coRowMenu *TangibleSimulationMenu;

    coTUITab *TrackObjectsTab;
    coTUIToggleButton *trackObjects;
    coTUIComboBox *objectChoice;
    
    coTUIEditFloatField *posX;
    coTUIEditFloatField *posY;
    coTUIEditFloatField *posZ;
    coTUIEditFloatField *rotH;
    coTUIEditFloatField *rotP;
    coTUIEditFloatField *rotR;
    
    float x,y,z,h,p,r;


private:
};
#endif
