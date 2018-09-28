/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/input/VRKeys.h>
#include "BoxSelection.h"
#include "BoxSelectionInteractor.h"

#include <util/common.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <cover/ui/Button.h>
#include <cover/ui/Button.h>

using namespace opencover;
using vrui::coInteraction;
using vrui::coInteractionManager;

float BoxSelection::s_xmin = 0.;
float BoxSelection::s_ymin = 0.;
float BoxSelection::s_zmin = 0.;
float BoxSelection::s_xmax = 0.;
float BoxSelection::s_ymax = 0.;
float BoxSelection::s_zmax = 0.;
BoxSelectionInteractor *BoxSelection::s_boxSelectionInteractor = NULL;
void (*BoxSelection::s_interactionFinished)() = NULL;

BoxSelection::BoxSelection(ui::Menu *parentMenu, const char *name, const char *title)
    : ui::Owner(std::string("BoxSelection")+name, cover->ui)
{
    m_selectionSubMenu = new ui::Menu(title, this);
    parentMenu->add(m_selectionSubMenu);

    s_boxSelectionInteractor = new BoxSelectionInteractor(
        coInteraction::ButtonA, "BoxSelection", coInteraction::High);
    s_boxSelectionInteractor->registerInteractionFinishedCallback(interactionFinished);
    s_boxSelectionInteractor->registerInteractionRunningCallback(interactionRunning);

    m_useBoxSelection = new ui::Button("BoxSelection", this, cover->navGroup());
    m_useBoxSelection->setGroup(cover->navGroup(), coVRNavigationManager::NavOther);
    m_useBoxSelection->setCallback([this](bool state){
        if (state)
        {
            // enable interaction
            printf("register boxSelectionInteractor\n");
            VRSceneGraph::instance()->setPointerType(HAND_SPHERE);
            coInteractionManager::the()->registerInteraction(s_boxSelectionInteractor);
        }
        else
        {
            printf("unregister boxSelectionInteractor\n");
            coInteractionManager::the()->unregisterInteraction(s_boxSelectionInteractor);
        }
    });
}

BoxSelection::~BoxSelection()
{
    delete s_boxSelectionInteractor;
}

void BoxSelection::interactionRunning()
{
    s_boxSelectionInteractor->getBox(s_xmin, s_ymin, s_zmin, s_xmax, s_ymax, s_zmax);
}

inline std::string
BoxSelection::stringify(float x)
{
    std::ostringstream o;
    if (o << x)
        return o.str();
    else
        return "";
}

bool BoxSelection::getCheckboxState() const
{
    if (m_useBoxSelection)
    {
        return m_useBoxSelection->state();
    }
    else
    {
        return false;
    }
}

ui::Button *BoxSelection::getButton() const
{
    return m_useBoxSelection;
}

void BoxSelection::interactionFinished()
{
    interactionRunning();
    if (s_interactionFinished)
        s_interactionFinished();
}

void BoxSelection::getBox(float &xmin, float &ymin, float &zmin, float &xmax, float &ymax, float &zmax)
{
    xmin = s_xmin;
    ymin = s_ymin;
    zmin = s_zmin;
    xmax = s_xmax;
    ymax = s_ymax;
    zmax = s_zmax;
}

void BoxSelection::registerInteractionFinishedCallback(void (*interactionFinished)())
{
    s_interactionFinished = interactionFinished;
}

void BoxSelection::unregisterInteractionFinishedCallback()
{
    s_interactionFinished = NULL;
}
