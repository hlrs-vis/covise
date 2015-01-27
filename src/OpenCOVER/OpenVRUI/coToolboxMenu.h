/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TOOLBOX_MENU_H
#define CO_TOOLBOX_MENU_H

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coRowContainer.h>
#include <OpenVRUI/coUIElement.h>

#include <string>

//class coButton;

namespace vrui
{

class coMenuItem;
class coTexturedBackground;
class coBackground;
class coToolboxMenuHandle;
class coFrame;
class coSubMenuToolboxItem;
class coIconButtonToolboxItem;

/// A Toolbox Menu contains Toolbox Menu Item and
/// offers different organization modes
/// (horizontal or vertical)
/// The Handle, Items and submenus are organized respectively
/// to this setting.
class OPENVRUIEXPORT coToolboxMenu
    : public coMenu,
      public coButtonActor,
      public coMenuListener
{
protected:
    coRowContainer *itemsContainer; ///< menu items (anything below title bar)

    coFrame *itemsFrame; /// surrounding frame
    // coBackground* itemsBackground;            /// global background

    coToolboxMenuHandle *handle; /// handle

public:
    coToolboxMenu(const std::string &name = "Toolbox",
                  coMenu *parent = 0,
                  coRowContainer::Orientation orientation = coRowContainer::HORIZONTAL,
                  int attachment = coUIElement::BOTTOM, int maxItems = 0);

    virtual ~coToolboxMenu();

    void add(coMenuItem *);
    void insert(coMenuItem *, int);
    void remove(coMenuItem *);
    void removeAll();
    int getItemCount() const;
    virtual void setScale(float s); ///< Sets the menu size by applying a scaling factor to the default size.
    virtual float getScale() const; ///< Gets the current scaling factor.

    virtual void setTransformMatrix(vruiMatrix *mat);
    virtual void setTransformMatrix(vruiMatrix *mat, float scale);

    virtual void setVisible(bool newState);

    vruiTransformNode *getDCS();
    virtual coUIElement *getUIElement();

    virtual void buttonEvent(coButton *)
    {
    }

    bool update();

    // Functions to Show and Hide the Toolbar
    void show();
    void hide();

    // set the container Attachment
    virtual void setAttachment(int);
    virtual int getAttachment() const;

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    void fixPos(bool doFix); ///< fix mmy menu's position

    virtual void selected(bool select); ///< Menu is selected via joystick
    virtual void makeVisible(coMenuItem *item); ///< makes the item visible for joystick interaction

private:
    int attachment;
    int isHidden;
    int stateChangeRunning;
    int showOperation;

    double t_start, t_end, t_now;
    double stateDelay;

    coIconButtonToolboxItem *upItem_, *downItem_;
    int maxItems_;
    int startPos_;
    virtual void menuEvent(coMenuItem *item);
};
}
#endif
