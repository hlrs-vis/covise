/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coCheckboxMenuItem.h>

#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuContainer.h>

#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/util/vruiLog.h>

#include <OpenVRUI/coCombinedButtonInteraction.h>

using namespace std;

namespace vrui
{

/** Constructor.
  @param name           displayed menu item text
  @param defaultState   default checkbox state (true = checked)
  @param cbg            checkbox group this element is to be added to. Defaults to NULL for independent checkbox.
*/
coCheckboxMenuItem::coCheckboxMenuItem(const string &name, bool defaultState, coCheckboxGroup *cbg)
    : coRowMenuItem(name)
{
    group = cbg;
    if (group)
    {
        group->add(this);
        if (group->getAllowDeselect())
            checkBox = new coToggleButton(new coFlatButtonGeometry("UI/rund"), this);
        else
            checkBox = new coToggleButton(new coFlatButtonGeometry("UI/radio"), this);
        group->setState(this, defaultState);
    }
    else
    {
        checkBox = new coToggleButton(new coFlatButtonGeometry("UI/haken"), this);
        checkBox->setState(defaultState);
        myState = defaultState;
    }
    checkBox->setSize((float)LEFTMARGIN);
    container->addElement(checkBox);
    container->addElement(label);
    vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
}

/// Destructor.
coCheckboxMenuItem::~coCheckboxMenuItem()
{
    vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    if (group)
        group->remove(this);
    delete checkBox;
}

/** This method is called on intersections of the input device with the
  checkbox menu item.
  @return ACTION_CALL_ON_MISS
*/

int coCheckboxMenuItem::hit(vruiHit *hit)
{

    //VRUILOG("coCheckboxMenuItem::hit info: called")
    if (!vruiRendererInterface::the()->isRayActive())
        return ACTION_CALL_ON_MISS;

    // update coJoystickManager
    if (vruiRendererInterface::the()->isJoystickActive())
        vruiRendererInterface::the()->getJoystickManager()->selectItem(this, myMenu);

    Result preReturn = vruiRendererInterface::the()->hit(this, hit);
    if (preReturn != ACTION_UNDEF)
        return preReturn;

    background->setHighlighted(true);

    if (myMenu && myMenu->getInteraction()->wasStopped() && !vruiRendererInterface::the()->isJoystickActive())
    {
        if (group)
        {
            group->toggleCheckbox(this);
        }
        else
        {
            checkBox->setState(!myState);
            myState = !myState;
        }

        if (listener)
            listener->menuEvent(this);
    }

    return ACTION_CALL_ON_MISS;
}

/// Called when input device leaves the element.
void coCheckboxMenuItem::miss()
{
    if (!vruiRendererInterface::the()->isRayActive())
        return;

    vruiRendererInterface::the()->miss(this);
    background->setHighlighted(false);
}

/**
  * highlight the item
  */
void coCheckboxMenuItem::selected(bool select)
{
    coRowMenuItem::selected(select);
}

/**
  * open or close Submenu
  */
void coCheckboxMenuItem::doActionRelease()
{
    if (group)
    {
        group->toggleCheckbox(this);
    }
    else
    {
        checkBox->setState(!myState);
        myState = !myState;
    }

    if (listener)
        listener->menuEvent(this);
}

// just for the sake of implementing it
void coCheckboxMenuItem::buttonEvent(coButton *button)
{
    (void)button;

    myState = !myState;

    if (listener)
        listener->menuEvent(this);
}

/** Set new checkbox state.
  @param newState true = checked, false = unchecked
  @param generateEvent if true, a menuEvent is generated
  @param updateGroup if true, checkbox group members are updated accordingly, default false
*/
void coCheckboxMenuItem::setState(bool newState, bool generateEvent, bool updateGroup)
{
    myState = newState;
    if (generateEvent || checkBox->getState() != newState) // neu, events werden jetzt generiert, auch wenn sich der Status nicht geaendert hat, in der Hoffnung, dass das nicht zu Problemen führt
    // falls doch, dann muss das || generateEvent wieder raus und bei coVRModuleSupport::setBuiltInFunctionState explizit die Callbacks aufgerufen werden, sonst tut das Setzen der Navigationmodes von VRML aus nicht immer.
    {
        if (group && updateGroup)
            group->setState(this, newState, generateEvent);
        else
            checkBox->setState(newState);
        if (listener && generateEvent)
            listener->menuEvent(this);
    }
    /*else // dies wurde geaendert da sonst evtl. checkboxes deaktiviert wurden ohne einen Event zu generieren (Uwe)
   {
   if (group&&updateGroup)
   group->setState(this, newState);
   else
   checkBox->setState(newState);
   }*/
}

/** Get checkbox state.
  @return checkbox state (true = checked, false = unchecked)
*/
bool coCheckboxMenuItem::getState() const
{
    return checkBox->getState();
}

/** Get group this checkbox belongs to
  @return pointer to checkbox group
*/
coCheckboxGroup *coCheckboxMenuItem::getGroup()
{
    return group;
}

const char *coCheckboxMenuItem::getClassName() const
{
    return "coCheckboxMenuItem";
}

bool coCheckboxMenuItem::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coRowMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

//set if item is active
void coCheckboxMenuItem::setActive(bool a)
{
    // if item is activated add background to intersector
    if (!active_ && a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->add(background->getDCS(), this);
        checkBox->setActive(a);
    }
    // if item is deactivated remove background from intersector
    else if (active_ && !a)
    {
        vruiIntersection::getIntersectorForAction("coAction")->remove(this);
        checkBox->setActive(a);
    }
    coRowMenuItem::setActive(a);
}

}
