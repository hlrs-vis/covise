/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SLIDER_TOOLBOX_ITEM_H
#define CO_SLIDER_TOOLBOX_ITEM_H

#include <OpenVRUI/coToolboxMenuItem.h>
#include <OpenVRUI/coSlider.h>
#include <OpenVRUI/coAction.h>

#include <string>

/** This class defines a menu item which provides a slider to manipulate a scalar value.
  The slider UI element is defined in coSlider, events are processed
  by coSliderActor and coAction.
*/
namespace vrui
{

class OPENVRUIEXPORT coSliderToolboxItem
    : public coToolboxMenuItem,
      public coSliderActor,
      public coAction
{
protected:
    coSlider *slider; ///< slider UI element
    coLabel *minLabel;
    coLabel *maxLabel;
    coLabel *label;

public:
    coSliderToolboxItem(const std::string &, float, float, float);
    virtual ~coSliderToolboxItem();
    virtual int hit(vruiHit *);
    virtual void miss();
    void setValue(float);
    float getValue() const;
    void setMin(float);
    float getMin() const;
    void setMax(float);
    float getMax() const;
    void setNumTicks(float);
    float getNumTicks() const;
    void setPrecision(int);
    int getPrecision() const;
    void setInteger(bool);
    bool isInteger() const;
    void setLabel(const std::string &labelstr);
    void sliderEvent(coSlider *);
    void sliderReleasedEvent(coSlider *);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// functions activates or deactivates the item
    virtual void setActive(bool a);

    virtual void selected(bool select); ///< MenuItem is selected via joystick
    virtual void doActionPress();
    virtual void doSecondActionPress();
};
}
#endif
