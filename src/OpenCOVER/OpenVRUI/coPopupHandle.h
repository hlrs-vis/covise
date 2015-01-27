/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_POPUPHANDLE_H
#define CO_POPUPHANDLE_H

#include <OpenVRUI/coRowContainer.h>
#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coUpdateManager.h>
#include <string>

namespace vrui
{

class coLabel;
class coButton;
class coBackground;
class coFrame;
class coMenuContainer;
class coCombinedButtonInteraction;
class vruiHit;
class vruiMatrix;

/// Popup Handler
class OPENVRUIEXPORT coPopupHandle
    : public coRowContainer,
      public coAction,
      public coUpdateable,
      public coButtonActor
{
public:
    coPopupHandle(const std::string &title);
    virtual ~coPopupHandle();
    virtual void setTransformMatrix(vruiMatrix *matrix);
    virtual void setTransformMatrix(vruiMatrix *matrix, float scale);
    virtual void setScale(float scale);
    virtual float getScale() const;
    virtual bool update();

    virtual void resizeToParent(float, float, float, bool shrink = true);
    virtual void shrinkToMin();

    virtual void createGeometry()
    {
    }
    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    virtual int hit(vruiHit *hit);

    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    virtual void buttonEvent(coButton *source);
    /// true, if the pointer could be grabbed for an interaction

    coCombinedButtonInteraction *interactionA; ///< interaction for first button
    coCombinedButtonInteraction *interactionB; ///< interaction for second button
    coCombinedButtonInteraction *interactionC; ///< interaction for third button

    bool unregister;
    /// transformation matrix at the beginning of an interaction
    vruiMatrix *startPosition;
    /// point in local menu coordinates where the titlebar was picked
    coVector localPickPosition;
    /// point in world coordinates where the titlebar was picked
    coVector pickPosition;
    /// inverse transformation of the pointer at the beginning of an interaction
    vruiMatrix *invStartHandTrans;
    /// roll angle during the last frame
    float lastRoll;
    /// the size
    float myScale;

private:
    //
    coLabel *titleLabel; ///< the title
    coButton *closeButton; ///< the close button
    coButton *minmaxButton; ///< the min/max button
    coBackground *titleBackground; ///< background behind the titlebar
    coMenuContainer *titleContainer; ///< container that holds the title bar elements
    coFrame *titleFrame; ///< frame around the title bar
    std::string title; ///< the tilte string
    bool minimized; ///< true, if the menu is minimized
};
}
#endif
