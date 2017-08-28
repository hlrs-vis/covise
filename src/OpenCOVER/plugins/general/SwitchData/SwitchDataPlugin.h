/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SWITCHDATA_PLUGIN_H
#define SWITCHDATA_PLUGIN_H

#include <OpenVRUI/coMenuItem.h>
#include <cover/coTabletUI.h>
#include <cover/coVRPlugin.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
}

namespace opencover
{
class coTUITab;
class coTUIToggleButton;
class coHud;
}

using namespace vrui;
using namespace opencover;

class SwitchDataPlugin : public coVRPlugin, public coMenuListener, public coTUIListener
{
private:
    bool firsttime;
    coInteractor *inter;
    coSubMenuItem *pinboardButton;
    coRowMenu *pluginSubmenu;
    coCheckboxGroup *checkboxgroup;
    std::vector<coCheckboxMenuItem *> checkboxes;
    int numChoices;
    char **choices;

    void createSubmenu(int numChoices, char **choices, int currentChoice);
    void deleteSubmenu();

    const char *choiceParamName;
    void menuEvent(coMenuItem *);
    void tabletEvent(coTUIElement *);
    void menuReleaseEvent(coMenuItem *);

    coTUITab *tuiTab;
    std::vector<coTUIToggleButton *> tuiButtons;
    void showHud(const std::string &text);

public:
    char *currentObjectName;

    SwitchDataPlugin();
    virtual ~SwitchDataPlugin();
    bool init();
    void preFrame();
    void newInteractor(const RenderObject *r, coInteractor *i);
    void add(coInteractor *inter);
    void removeObject(const char *objName, bool replace);
    void remove(const char *objName, bool deleteMenu = true);
};

#endif
