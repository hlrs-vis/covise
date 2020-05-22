/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** \mainpage OpenVRUI
  OpenVRUI is an object oriented virtual reality user interface API.
  Its design is scene graph independent, OpenSG and OpenSceneGraph
  backends are implemented, but only the latter is fully functional.
  Its syntax resembles GUIs like Java AWT or QT.<P>
  The base class of all GUI elements is vruiUIElement.<P>
  Events are processed by Actors (e.g., coButtonActor, coSliderActor,
  coValuePotiActor), which are called on user events. User interface
  elements defined by the API developer need to be derived from the respective
  actors in order to receive events.
*/

#ifndef CO_UI_ELEMENT_H
#define CO_UI_ELEMENT_H

#ifndef coMax
#define coMax(v1, v2) ((v1) > (v2) ? (v1) : (v2))
#endif

#include <util/coTypes.h>

#include <OpenVRUI/sginterface/vruiMatrix.h>
#include <util/coTypes.h>

#include <string>

namespace vrui
{

class coUIContainer;
class coUIUserData;

class vruiTransformNode;
class vruiUIElementProvider;

/**
 * Basic VRUI GUI element.
 * This class provides functionality for all VRUI elements like position,
 * size, font, visibility, availability, parent, etc.<BR>
 * At least this class should be subclassed for any new GUI element types.<BR>
 * All inheritable functions are defined virtual so that they can be overwritten
 * by subclasses.
 */

class OPENVRUIEXPORT coUIElement
{
private:
    coUIContainer *parentContainer; ///< info about parent container, needed by layout managers and destructors
    coUIUserData *userData; ///< userdata that can be attached to any UI Element

    std::string Unique_Name;

public:
    // it is the callers' responsibility to delete the returned matrix by calling vruiRendererInterface::the()->deleteMatrix(matrix);
    static vruiMatrix *getMatrixFromPositionHprScale(float x, float y, float z, float h, float p, float r, float scale);

    /// Color definitions, to be used whenever a material is needed.
    enum Material
    {
        RED = 0,
        GREEN,
        BLUE,
        YELLOW,
        GREY,
        WHITE,
        BLACK,
        DARK_YELLOW,
        WHITE_NL, ///< self illuminated white (NL = no lighting)
        ITEM_BACKGROUND_NORMAL,
        ITEM_BACKGROUND_HIGHLIGHTED,
        ITEM_BACKGROUND_DISABLED,
        HANDLE_BACKGROUND_NORMAL,
        HANDLE_BACKGROUND_HIGHLIGHTED,
        HANDLE_BACKGROUND_DISABLED,
        NUM_MATERIALS ///< this entry must always be the last one in the list
    };

    coUIElement();
    virtual ~coUIElement();

    virtual void createGeometry();

    virtual void setParent(coUIContainer *);
    virtual coUIContainer *getParent();

    virtual void setEnabled(bool enabled);
    virtual void setHighlighted(bool highlighted);
    virtual void setVisible(bool visible);

    virtual bool isEnabled() const;
    virtual bool isHighlighted() const;
    virtual bool isVisible() const;

    virtual float getWidth() const = 0; ///< Returns element width
    virtual float getHeight() const = 0; ///< Returns element height
    virtual float getDepth() const;
    virtual float getXpos() const = 0; ///< Returns element x position
    virtual float getYpos() const = 0; ///< Returns element y position
    virtual float getZpos() const;

    virtual void childResized(bool shrink = true);
    virtual void resizeToParent(float, float, float, bool shrink = true);
    virtual void shrinkToMin();

    ///< Set element location in space.
    virtual void setPos(float, float, float) = 0;
    virtual void setSize(float, float, float);
    virtual void setSize(float);
    virtual float getResizePref() ///< Do we like to be resized - 0.0=never resize, 1=take 1 share of free space
    {
        return 0.0f;
    }
    virtual void setUserData(coUIUserData *);
    virtual coUIUserData *getUserData() const;

    virtual vruiTransformNode *getDCS();

    virtual vruiUIElementProvider *getUIElementProvider() const
    {
        return uiElementProvider;
    }

    /// attachment order is counterclockwise like this and describes
    /// the side/border which the item is attached to!
    enum
    {
        LEFT = 0,
        TOP,
        RIGHT,
        BOTTOM,
        REPLACE
    } Attachments;

    virtual void setAttachment(int) /// sets the attachment border
    {
    }
    virtual int getAttachment() const /// returns the attachment border
    {
        return LEFT;
    }

    void setUniqueName(const char *);
    const char *getUniqueName() const;

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    virtual void resizeGeometry();

    float xScaleFactor; ///< UI element x axis scale factor
    float yScaleFactor; ///< UI element y axis scale factor
    float zScaleFactor; ///< UI element z axis scale factor
    bool enabled; ///< true if UI element is enabled, false if UI element cannot be used
    bool highlighted; ///< true if highlighted
    bool visible; ///< true if UI element is visible, false if not visible but still present in scene tree

    virtual const vruiMatrix *getTransformMatrix();

    vruiUIElementProvider *uiElementProvider;

private:
    vruiMatrix *transformMatrix;
};
}
#endif
