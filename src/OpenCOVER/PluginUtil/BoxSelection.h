/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BOX_SELECTION_
#define _BOX_SELECTION_

#include <util/coExport.h>
#ifdef VRUI
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coMenu;
class coRowMenu;
class coCheckboxMenuItem;
class coLabelMenuItem;
}
#else
#include <cover/ui/Owner.h>
namespace opencover
{
namespace ui
{
class Menu;
class Button;
class Label;
}
}
#endif

namespace opencover
{
class BoxSelectionInteractor;

class PLUGIN_UTILEXPORT BoxSelection
#ifdef VRUI
    : public vrui::coMenuListener, public vrui::coMenuItem
#else
    : public ui::Owner
#endif
{

public:
    BoxSelection(opencover::ui::Menu *, const char *name, const char *label);
    ~BoxSelection();
    BoxSelection *instance();
    void registerInteractionFinishedCallback(void (*interactionFinished)());
    void unregisterInteractionFinishedCallback();
#ifdef VRUI
    void setMenuListener(vrui::coMenuListener *);
    vrui::coMenuListener *getMenuListener() const;
    vrui::coSubMenuItem *getSubMenu() const;
    vrui::coCheckboxMenuItem *getCheckbox() const;
#else
    ui::Button *getButton() const;
#endif
    bool getCheckboxState() const;

    void getBox(float &xmin, float &ymin, float &zmin, float &xmax, float &ymax, float &zmax);
    void update(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax);

private:
    static void (*s_interactionFinished)();
#ifdef VRUI
    vrui::coMenu *m_pinboard;
    vrui::coSubMenuItem *m_infoSubMenuItem;
    vrui::coCheckboxMenuItem *m_useBoxSelection;
    vrui::coLabelMenuItem *m_xminItem, *m_yminItem, *m_zminItem;
    vrui::coLabelMenuItem *m_xmaxItem, *m_ymaxItem, *m_zmaxItem;
    vrui::coRowMenu *m_selectionSubMenu;
    vrui::coMenuListener *m_parentListener;

    void menuEvent(coMenuItem *);
#else
    ui::Menu *m_pinboard = nullptr;
    ui::Button *m_useBoxSelection = nullptr;
    ui::Label *m_xminItem = nullptr, *m_yminItem = nullptr, *m_zminItem = nullptr;
    ui::Label *m_xmaxItem = nullptr, *m_ymaxItem = nullptr, *m_zmaxItem = nullptr;
    ui::Menu *m_selectionSubMenu = nullptr;
#endif
    static float s_xmin;
    static float s_ymin;
    static float s_zmin;
    static float s_xmax;
    static float s_ymax;
    static float s_zmax;

    static opencover::BoxSelectionInteractor *s_boxSelectionInteractor;

    void createSubMenu();
    void deleteSubMenu();

    static void interactionFinished();
    static void interactionRunning();

    std::string stringify(float);
};
}
#endif
