/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//--------------------------------------------------------------------------//
//--------------------------------------------------------------------------//
//                       Cyber Classrom                                     //
//                       Visenso GmbH                                       //
//                       2012                                               //
//                                                                          //
//$Id$
//--------------------------------------------------------------------------//
//--------------------------------------------------------------------------//

#ifndef _DNA_PLUGIN_H
#define _DNA_PLUGIN_H

#include <map>
#include <list>
#include <string>

#include <cover/coVRPlugin.h>
#include <cover/coHud.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>

#include "DNABaseUnit.h"

namespace vrui
{
class coRowMenu;
}

class DNAPlugin : public opencover::coVRPlugin, public vrui::coMenuListener
{
public:
    DNAPlugin();

    virtual ~DNAPlugin();

    virtual bool init();

    virtual void preFrame();
    virtual void guiToRenderMsg(const char *msg);
    virtual void menuEvent(vrui::coMenuItem *menuItem);

private:
    std::list<DNABaseUnit *> dnaBaseUnits;
    vrui::coRowMenu *menu_;
    vrui::coButtonMenuItem *disconnMenuItem_;
    float barrierAngle;
    std::list<DNABaseUnitConnectionPoint *> availableConnectionPoints;

    opencover::coHud *hud_; // hud for messages
    int justConnected_; // number of frames from the last connection the last connection
    float showHud_; // count the frames the hud has been shown
    bool hudNotForward_; // flag if hud shows 'can not got to next presentationStep' message
};
#endif
