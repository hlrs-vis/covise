/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUBE_PLUGIN_H
#define CUBE_PLUGIN_H

#include <OpenVRUI/coMenuItem.h>
#include <cover/coVRPlugin.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coPotiMenuItem;
}

#include "CubeInteractor.h"

using namespace vrui;
using namespace opencover;

class CubePlugin : public coVRPlugin, public coMenuListener
{
private:
    bool firsttime;
    CubeInteractor *wireCube;
    coInteractor *inter;
    coSubMenuItem *pinboardButton;
    coRowMenu *cubeSubmenu;
    coPotiMenuItem *sizePoti;
    coCheckboxMenuItem *moveCheckbox;

    void createSubmenu();
    void deleteSubmenu();

    const char *sizeParamName, *centerParamName;
    void menuEvent(coMenuItem *);
    void menuReleaseEvent(coMenuItem *);

public:
    static CubePlugin *plugin;
    static char *currentObjectName;

    CubePlugin();
    virtual ~CubePlugin();
    bool init();
    void newInteractor(const RenderObject *r, coInteractor *i);
    void add(coInteractor *inter);
    void removeObject(const char *objName, bool replace);
    void remove(const char *objName);
    void preFrame();
};

#endif
