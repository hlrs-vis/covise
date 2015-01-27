/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_ICONSUBMENU_TOOLBOXITEM_H
#define CO_ICONSUBMENU_TOOLBOXITEM_H

#include <OpenVRUI/coSubMenuToolboxItem.h>
#include <OpenVRUI/coButton.h>

/** Menu item which can be used to open and close submenus
    menu events are generated when the user opens or closes the submenu
    In difference to a normal Menu subwindow, orientation (attachement) changes
    have to be handled properly.
*/
namespace vrui
{

class OPENVRUIEXPORT coIconSubMenuToolboxItem
    : public coSubMenuToolboxItem,
      public coButtonActor
{
protected:
    coToggleButton *myButton;

public:
    coIconSubMenuToolboxItem(const std::string &name);
    virtual ~coIconSubMenuToolboxItem();

    // not really used but there because of coButton Constructor :-/
    virtual void buttonEvent(coButton *)
    {
    }

    // only for not shadowing the coRotButton buttonEvent
    virtual void buttonEvent(coRotButton *b)
    {
        coSubMenuToolboxItem::buttonEvent(b);
    }

    virtual void setAttachment(int);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *name) const;

private:
    float smallGap, largeGap;
};
}
#endif
