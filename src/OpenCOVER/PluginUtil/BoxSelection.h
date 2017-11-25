/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BOX_SELECTION_
#define _BOX_SELECTION_

#include <util/coExport.h>
#include <cover/ui/Owner.h>

namespace opencover {
namespace ui {
class Menu;
class Button;
class Label;
}
}

namespace opencover
{
class BoxSelectionInteractor;

class PLUGIN_UTILEXPORT BoxSelection
    : public ui::Owner
{

public:
    BoxSelection(opencover::ui::Menu *, const char *name, const char *label);
    ~BoxSelection();
    BoxSelection *instance();
    void registerInteractionFinishedCallback(void (*interactionFinished)());
    void unregisterInteractionFinishedCallback();
    ui::Button *getButton() const;
    bool getCheckboxState() const;

    void getBox(float &xmin, float &ymin, float &zmin, float &xmax, float &ymax, float &zmax);

private:
    static void (*s_interactionFinished)();
    ui::Menu *m_pinboard = nullptr;
    ui::Button *m_useBoxSelection = nullptr;
    ui::Menu *m_selectionSubMenu = nullptr;

    static float s_xmin;
    static float s_ymin;
    static float s_zmin;
    static float s_xmax;
    static float s_ymax;
    static float s_zmax;

    static opencover::BoxSelectionInteractor *s_boxSelectionInteractor;

    static void interactionFinished();
    static void interactionRunning();

    std::string stringify(float);
};
}
#endif
