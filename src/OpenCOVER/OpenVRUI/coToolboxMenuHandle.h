/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TOOLBOXMENUHANDLE_H
#define CO_TOOLBOXMENUHANDLE_H

#include <OpenVRUI/coRowContainer.h>
#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coUpdateManager.h>

#include <string>

namespace vrui
{

class vruiInteraction;
class vruiMatrix;

class coBackground;
class coFrame;
class coMenuContainer;
class coToolboxMenu;
class coCombinedButtonInteraction;

/// Offers a Handle for Toolboxes
class OPENVRUIEXPORT coToolboxMenuHandle
    : public coRowContainer,
      public coAction,
      public coUpdateable,
      public coButtonActor
{
public:
    coToolboxMenuHandle(const std::string &, coToolboxMenu *);
    virtual ~coToolboxMenuHandle();
    virtual void setTransformMatrix(vruiMatrix *mat);
    virtual void setTransformMatrix(vruiMatrix *mat, float scale);
    virtual void setScale(float s);
    virtual float getScale() const;
    virtual bool update();

    virtual void addCloseButton(); ///< adds the close button to the title bar

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    virtual int hit(vruiHit *hit);

    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    virtual void highlight(bool highlight);

    void setOrientation(coRowContainer::Orientation);

    /// get the Element's classname
    virtual const char *getClassName() const;

    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// fix this menu at current position
    void fixPos(bool doFix);

protected:
    virtual void buttonEvent(coButton *);

    coToolboxMenu *myMenu;

    coCombinedButtonInteraction *interactionA; ///< interaction for first button
    coCombinedButtonInteraction *interactionB; ///< interaction for second button
    coCombinedButtonInteraction *interactionC; ///< interaction for third button

    bool unregister;
    vruiMatrix *startPosition;
    coVector localPickPosition;
    coVector pickPosition;
    vruiMatrix *invStartHandTrans;
    float lastRoll;
    float myScale;
    bool fixedPos; ///< menu is fixed and may not be moved

private:
    coButton *closeButton;
    coButton *minmaxButton;
    coButton *cwrotButton;

    coBackground *titleBackground;
    coFrame *titleFrame;
    coMenuContainer *titleContainer; ///< toolbox handle bar

    bool minimized;
};
}
#endif
