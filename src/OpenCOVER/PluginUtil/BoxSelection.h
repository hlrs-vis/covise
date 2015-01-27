/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BOX_SELECTION_
#define _BOX_SELECTION_

#include <util/coExport.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coMenu;
class coRowMenu;
class coCheckboxMenuItem;
class coLabelMenuItem;
}

namespace opencover
{
class BoxSelectionInteractor;

class PLUGIN_UTILEXPORT BoxSelection : public vrui::coMenuListener, public vrui::coMenuItem
{

public:
    BoxSelection(vrui::coRowMenu *, const char *name, const char *label);
    ~BoxSelection();
    BoxSelection *instance();
    void registerInteractionFinishedCallback(void (*interactionFinished)());
    void unregisterInteractionFinishedCallback();
    void setMenuListener(vrui::coMenuListener *);
    vrui::coMenuListener *getMenuListener() const;
    vrui::coSubMenuItem *getSubMenu() const;
    vrui::coCheckboxMenuItem *getCheckbox() const;
    bool getCheckboxState() const;

    void getBox(float &xmin, float &ymin, float &zmin, float &xmax, float &ymax, float &zmax);
    void update(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax);

private:
    static void (*s_interactionFinished)();
    vrui::coMenu *m_pinboard;
    vrui::coSubMenuItem *m_infoSubMenuItem;
    vrui::coCheckboxMenuItem *m_useBoxSelection;
    vrui::coLabelMenuItem *m_xminItem, *m_yminItem, *m_zminItem;
    vrui::coLabelMenuItem *m_xmaxItem, *m_ymaxItem, *m_zmaxItem;
    vrui::coRowMenu *m_selectionSubMenu;
    vrui::coMenuListener *m_parentListener;
    static float s_xmin;
    static float s_ymin;
    static float s_zmin;
    static float s_xmax;
    static float s_ymax;
    static float s_zmax;

    static opencover::BoxSelectionInteractor *s_boxSelectionInteractor;
    void menuEvent(coMenuItem *);

    void createSubMenu();
    void deleteSubMenu();

    static void interactionFinished();
    static void interactionRunning();

    std::string stringify(float);
};
}
#endif
