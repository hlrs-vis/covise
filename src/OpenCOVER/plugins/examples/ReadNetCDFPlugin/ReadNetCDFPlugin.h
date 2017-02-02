/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CDF_PLUGIN_H
#define CDF_PLUGIN_H

#include <OpenVRUI/coMenuItem.h>
#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <vector>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coSliderMenuItem;
class coPotiMenuItem;
}

// -----------------------------------------------------------------------------
// class CDFPlugin
// -----------------------------------------------------------------------------
class CDFPlugin : public opencover::coVRPlugin,
                  public vrui::coMenuListener,
                  public opencover::coTUIListener
{
public:
    static CDFPlugin *plugin;
    static char *currentObjectName;

    CDFPlugin();
    virtual ~CDFPlugin();
    bool init();
    void newInteractor(const opencover::RenderObject *r,
                       opencover::coInteractor *i);
    void add(opencover::coInteractor *inter);
    void removeObject(const char *objName, bool replace);
    void remove(const char *objName);

private:
    bool firsttime;
    opencover::coInteractor *interactor;
    vrui::coSubMenuItem *pinboardButton;
    vrui::coRowMenu *cubeSubmenu;
    vrui::coPotiMenuItem *sizePoti;
    vrui::coCheckboxMenuItem *cbxSwitch;
    vrui::coSliderMenuItem *sldSlider;

    void createSubmenu();
    void deleteSubmenu();

    void menuEvent(vrui::coMenuItem *);
    void tabletEvent(opencover::coTUIElement *tUIItem);
    void tabletPressEvent(opencover::coTUIElement *tUIItem);

    opencover::coTUILabel *lblNCFile;
    opencover::coTUILabel *lblFilename;
    opencover::coTUILabel *lblGridOutX;
    opencover::coTUILabel *lblGridOutY;
    opencover::coTUILabel *lblGridOutZ;
    opencover::coTUITab *paramTab;
    //opencover::coTUIFloatSlider *sldTabSlider;
    //opencover::coTUIButton      *btnTabSwitch;
    opencover::coTUIFileBrowserButton *fbbFileBrowser;
    opencover::coTUIComboBox *cbxGridOutX;
    opencover::coTUIComboBox *cbxGridOutY;
    opencover::coTUIComboBox *cbxGridOutZ;

    std::vector<opencover::coTUILabel *> *vecVarLabel;
    std::vector<opencover::coTUIComboBox *> *vecComboBox;
};

// -----------------------------------------------------------------------------

#endif
