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

class TangiblePositionPlugin : public coVRPlugin, public coTUIListener, public coMenuListener
{
public:
    TangiblePositionPlugin();
    virtual ~TangiblePositionPlugin();
    bool init();

    // this will be called if an object with feedback arrives
    void newInteractor(RenderObject *, coInteractor *i);

    void menuEvent(coMenuItem *);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

    void updateAndExec();

    std::list<coInteractor *> interactors;
    coButtonMenuItem *execButton;
    coIconButtonToolboxItem *ToolbarButton;
    coSubMenuItem *pinboardEntry;
    coRowMenu *TangibleSimulationMenu;

    coTUITab *TangibleTab;
    coTUIButton *RestartSimulation;

    // this will be called if a COVISE object arrives
    void addObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

private:
};
#endif
