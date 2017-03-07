/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef LS1_PLUGIN_H
#define LS1_PLUGIN_H

#include <OpenVRUI/coMenuItem.h>
#include <cover/coVRPlugin.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coPotiMenuItem;
}

using namespace vrui;
using namespace opencover;

class LS1Plugin : public coVRPlugin, public coMenuListener
{
private:
    bool firsttime;
    coInteractor *inter;
    coSubMenuItem *pinboardButton;
    coRowMenu *ls1Submenu;
    coPotiMenuItem *tempPoti;

    void createSubmenu();
    void deleteSubmenu();

    const char *TempParamName;
    void menuEvent(coMenuItem *);
    void menuReleaseEvent(coMenuItem *);

public:
    static LS1Plugin *plugin;
    static char *currentObjectName;

    LS1Plugin();
    virtual ~LS1Plugin();
    bool init();
    bool destroy();
    void newInteractor(const RenderObject *r, coInteractor *i);
    void add(coInteractor *inter);
    void removeObject(const char *objName, bool replace);
    void remove(const char *objName);
    void preFrame();
};
#endif
