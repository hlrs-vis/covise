/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_ROW_MENU_H
#define CO_ROW_MENU_H

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coButton.h>

#include <OpenVRUI/sginterface/vruiMatrix.h>

namespace vrui
{

class coRowContainer;
class coRowMenuItem;
class coMenuItem;
class coButton;
class coTexturedBackground;
class coRowMenuHandle;
class coFrame;
class coIconButtonToolboxItem;

class coGenericSubMenuItem;

/** This class provides a simple menu for use in 3D space.
   The MenuItems are layed out in a row. It has a title bar (coMenuHandle)
   that shows the menu's name and can be used to reposition and scale
   the menu. There are buttons in the Menu to show, hide and close the menu.
*/
class OPENVRUIEXPORT coRowMenu : public coMenu, public coButtonActor, public coMenuListener
{
protected:
    coRowContainer *itemsContainer; ///< menu items (anything below title bar)
    coFrame *itemsFrame; ///< all menu items are framed by this frame
    coRowMenuHandle *handle; ///< the titlebar
    coGenericSubMenuItem *myMenuItem; ///< parent submenu, NULL if this is the topmost menu

public:
    coRowMenu(const char *title, coMenu *parent = 0, int maxItems = 0, bool inScene = false);
    virtual ~coRowMenu();
    virtual void add(coMenuItem *item);
    virtual void insert(coMenuItem *item, int position);
    virtual void remove(coMenuItem *item);

    virtual int hit(vruiHit *hit);
    virtual void miss();

    virtual void selected(bool select); ///< Menu is selected via joystick
    virtual void makeVisible(coMenuItem *item); ///< makes the item visible for joystick interaction

    virtual void setTransformMatrix(vruiMatrix *matrix);
    virtual void setTransformMatrix(vruiMatrix *matrix, float scalefactor);
    vruiTransformNode *getDCS();
    bool update();
    virtual coUIElement *getUIElement();
    virtual void setVisible(bool visible);
    virtual void setScale(float scale); ///< Sets the menu size by applying a scaling factor to the default size.
    virtual float getScale() const; ///< Gets the current scaling factor.

    // Functions to Show and Hide the Menu
    void show();
    void hide();

    //        coGenericSubMenuItem* getSubMenuItem();
    //        void setSubMenuItem(coGenericSubMenuItem*);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// update the Title bar
    virtual void updateTitle(const char *newTitle);

    virtual int getMaxItems()
    {
        return maxItems_;
    };
    virtual int getStartPos()
    {
        return startPos_;
    };

protected:
    virtual void buttonEvent(coButton *source);

private:
    bool isHidden; ///< visibility state
    bool stateChangeRunning; ///< true, if the menu is currently opened or closed
    bool showOperation; ///< true, if the current state change is a show operation

    double t_start; ///< time at the beginning of a show/hide operation
    double t_end; ///< time at the end of a show/hide operation
    double t_now; ///< current time of a show/hide operation
    double stateDelay; ///< time for a state change

    bool inScene_;

    coCombinedButtonInteraction *interactionB;

    coIconButtonToolboxItem *upItem_, *downItem_;
    int maxItems_;
    int startPos_;
    virtual void menuEvent(coMenuItem *item);
};
}
#endif
