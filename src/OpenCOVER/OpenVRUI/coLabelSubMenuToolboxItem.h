/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_LABELSUBMENU_TOOLBOXITEM_H
#define CO_LABELSUBMENU_TOOLBOXITEM_H

#include <OpenVRUI/coSubMenuToolboxItem.h>
#include <string>

namespace vrui
{

class coLabel;

/** Menu item which can be used to open and close submenus
    menu events are generated when the user opens or closes the submenu
    In difference to a normal Menu subwindow, orientation (attachement) changes
    have to be handled properly.
*/

class OPENVRUIEXPORT coLabelSubMenuToolboxItem
    : public coSubMenuToolboxItem
{
protected:
    coLabel *label;

public:
    coLabelSubMenuToolboxItem(const std::string &name);
    virtual ~coLabelSubMenuToolboxItem();

    virtual void setAttachment(int);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *name) const;

    /// change the label text
    virtual void setLabel(const std::string &labelString);

    virtual void buttonEvent(coRotButton *)
    {
    }
};
}
#endif
