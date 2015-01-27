/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coCheckboxGroup.h>

#include <OpenVRUI/coCheckboxMenuItem.h>

#include <OpenVRUI/util/vruiLog.h>

#include <algorithm>

using namespace std;

namespace vrui
{

/** Constructor.
  @param ad true if all checkboxes may be deselected (optional, defaults to false)
*/
coCheckboxGroup::coCheckboxGroup(bool ad)
{
    selected = 0;
    allowDeselect = ad;
}

/** Returns a pointer to the item which selected most recently.
  In case no item is seleted, it returns NULL.
*/
coCheckboxMenuItem *coCheckboxGroup::getSelectedCheckbox()
{
    return selected;
}

/** Change selection state of a coCheckboxMenuItem.
  @param selection coCheckboxMenuItem which was clicked on and thus should change its selection state
*/
void coCheckboxGroup::toggleCheckbox(coCheckboxMenuItem *selection)
{

    selected = selection;

    for (list<coCheckboxMenuItem *>::iterator currentItem = itemList.begin();
         currentItem != itemList.end();
         ++currentItem)
    {

        if ((*currentItem) == selected)
        {
            if (allowDeselect)
            {
                //VRUILOG("coCheckboxGroup::toggleCheckbox info: deselect")
                (*currentItem)->setState(!((*currentItem)->getState()));
            }
            else
            {
                //VRUILOG("coCheckboxGroup::toggleCheckbox info: deselect blocked")
                (*currentItem)->setState(true, true);
            }
        }
        else
        {
            //VRUILOG("coCheckboxGroup::toggleCheckbox info: select")
            (*currentItem)->setState(false, true);
        }
    }
}

/** Change selection state of a coCheckboxMenuItem.
  @param selection coCheckboxMenuItem which was clicked on and thus should change its state
  @param newState new state for selection
  @param generateEvent if true, a menuEvent is generated
*/
void coCheckboxGroup::setState(coCheckboxMenuItem *selection, bool newState, bool generateEvent)
{

    selected = selection;

    bool nothingSelected = true;
    for (list<coCheckboxMenuItem *>::iterator currentItem = itemList.begin();
         currentItem != itemList.end();
         ++currentItem)
    {
        if (*currentItem != selected)
        {
            if (newState && (*currentItem)->getState())
                (*currentItem)->setState(false, generateEvent);
        }

        if ((*currentItem)->getState())
            nothingSelected = false;
    }
    if (!allowDeselect && nothingSelected)
        newState = true;

    if (selected)
        selected->setState(newState, generateEvent);
}

/** Add a coCheckboxMenuItem to the radio button group.
  @param item item to add
*/
void coCheckboxGroup::add(coCheckboxMenuItem *item)
{
    if (find(itemList.begin(), itemList.end(), item) == itemList.end())
    {
        itemList.push_back(item);
    }
    else
    {
        VRUILOG("coCheckboxMenuItem::add warn: did not add duplicate item")
    }
}

/** Remove a coCheckboxMenuItem from the radio button group.
  @param item item to remove
*/
void coCheckboxGroup::remove(coCheckboxMenuItem *item)
{
    itemList.remove(item);
}

/** Returns true if all menu items of the group may be deselected at once,
  otherwise it returns false.
*/
bool coCheckboxGroup::getAllowDeselect() const
{
    return allowDeselect;
}
}
