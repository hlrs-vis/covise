/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SLIDER_MENU_ITEM_H
#define CO_SLIDER_MENU_ITEM_H

#include <OpenVRUI/coSlider.h>
#include <OpenVRUI/coRowMenuItem.h>
#include <OpenVRUI/coAction.h>
#include <string>

namespace vrui
{

class vruiHit;

/** This class defines a menu item which provides a slider to manipulate a scalar value.
  The slider UI element is defined in coSlider, events are processed
  by coSliderActor and coAction.
*/
class OPENVRUIEXPORT coSliderMenuItem
    : public coRowMenuItem,
      public coSliderActor,
      public coAction
{
protected:
    coSlider *slider; ///< slider UI element

public:
    coSliderMenuItem(const std::string &name, float min, float max, float init);
    virtual ~coSliderMenuItem();
    int hit(vruiHit *hit);
    void miss();
    void setValue(float value);
    float getValue() const;
    void setMin(float min);
    float getMin() const;
    void setMax(float max);
    float getMax() const;
    void setNumTicks(float ticks);
    float getNumTicks() const;
    void setPrecision(int precision);
    int getPrecision() const;
    bool isInteger() const;
    void setInteger(bool on);
    void sliderEvent(coSlider *slider);
    void sliderReleasedEvent(coSlider *);

    virtual void selected(bool select); ///< MenuItem is selected via joystick
    virtual void doActionPress(); ///< Action is called via joystick
    virtual void doActionRelease(); ///< Action is called via joystick
    virtual void doSecondActionPress(); ///< second Action for Item
    virtual void doSecondActionRelease(); ///< second Action for Item

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    /// functions activates or deactivates the item
    virtual void setActive(bool a);
};
}
#endif
