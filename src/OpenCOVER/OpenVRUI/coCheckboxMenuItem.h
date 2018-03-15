/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_CHECKBOXMENUITEM_H
#define CO_CHECKBOXMENUITEM_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coRowMenuItem.h>
#include <OpenVRUI/coButton.h>
#include <string>

namespace vrui
{

class coCheckboxGroup;

/** Menu item which is usable as a toggle button, a radio button,
    or a group of toggle buttons.
    The usage type is detemined by:<UL>
    <LI>for toggle buttons: no coCheckboxGroup assigned</LI>
    <LI>for radio buttons: assign a coCheckboxGroup without an argument to
        its constructor</LI>
    <LI>for grouped toggle buttons: assign a coCheckboxGroup with <I>true</I>
        as the argument to the constructor</LI></UL>
*/
class OPENVRUIEXPORT coCheckboxMenuItem : public coRowMenuItem, public coButtonActor, public coAction
{
protected:
    coButton *checkBox; ///< actual button which is used for interaction
    coCheckboxGroup *group; ///< checkbox group this checkbox belongs to
    bool myState;

    virtual void buttonEvent(coButton *source);

public:
    coCheckboxMenuItem(const std::string &name, bool on, coCheckboxGroup *group = 0);
    virtual ~coCheckboxMenuItem();
    void setState(bool newState, bool generateEvent, bool updateGroup = false);
    void setState(bool newState)
    {
        setState(newState, false);
    }
    bool getState() const;
    //       void setGroup(coCheckboxGroup * cbg);
    coCheckboxGroup *getGroup();
    int hit(vruiHit *hit);
    void miss();

    virtual void selected(bool select); ///< MenuItem is selected via joystick
    virtual void doActionRelease(); ///< Action is called via joystick

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// functions activates or deactivates the item
    virtual void setActive(bool a);
};
}
#endif
