/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SLIDER_H
#define CO_SLIDER_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/coUpdateManager.h>

namespace vrui
{

class coSlider;
class vruiHit;
class vruiTransformNode;
class coCombinedButtonInteraction;

/// Action listener for events triggered by coSlider.
class OPENVRUIEXPORT coSliderActor
{
public:
    virtual ~coSliderActor()
    {
    }
    virtual void sliderEvent(coSlider *slider);
    virtual void sliderReleasedEvent(coSlider *slider);
};

/** This class provides a basic 3D slider, which is based on a texture mapped
  tickmark field and a round slider position indicator.
*/
class OPENVRUIEXPORT coSlider : public coAction, public coUIElement, public coUpdateable
{
public:
    coSlider(coSliderActor *actor, bool showValue = true);
    virtual ~coSlider();

    // hit is called whenever the button
    // with this action is intersected
    // return ACTION_CALL_ON_MISS if you want miss to be called
    // otherwise return ACTION_DONE
    virtual int hit(vruiHit *hit);

    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    void joystickUp(); ///< increment slidervalue
    void joystickDown(); ///< decrementslidervalue
    void resetLastPressAction(); ///< no last press action

    void setValue(float val, bool generateEvent = false);
    float getValue() const;
    void setMin(float mi);
    float getMin() const;
    void setMax(float ma);
    float getMax() const;
    void setNumTicks(float nt);
    float getNumTicks() const;
    void setPrecision(int nt);
    int getPrecision() const;
    void setInteger(bool on);
    bool isInteger() const;
    float getDialSize() const;
    void setPos(float x, float y, float z = 0.0f);

    virtual void setHighlighted(bool highlighted);
    virtual void setSize(float x, float y, float z);
    virtual void setSize(float size);
    virtual float getWidth() const
    {
        return myWidth;
    }
    virtual float getHeight() const
    {
        return myHeight;
    }
    virtual float getXpos() const
    {
        return myX;
    }
    virtual float getYpos() const
    {
        return myY;
    }
    virtual float getZpos() const
    {
        return myZ;
    }
    virtual void setActive(bool a);
    virtual bool getActive()
    {
        return active_;
    };

    bool getShowValue() const;

    static void adjustSlider(float &mini, float &maxi, float & /*value*/, float &step, int &digits);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    void clamp() const;

    bool integer; ///< true = slider processes only integer values
    bool showValue; ///< true = the slider value is displayed
    coCombinedButtonInteraction *interactionA; ///< interaction for first button
    ///< interaction for wheel
    coCombinedButtonInteraction *interactionWheel[2];
    bool unregister; ///< true if the interaction should be unregistered
    float dialSize; ///< size of slider dial
    coSliderActor *myActor; ///< action listener for slider events
    float myX; ///< slider element x position in object space [mm]
    float myY; ///< slider element y position in object space [mm]
    float myZ; ///< slider element z position in object space [mm]
    float myWidth; ///< slider width [mm]
    float myHeight; ///< slider height [mm]
    float myDepth; ///< slider depth [mm]
    float minVal; ///< minimum slider value
    float maxVal; ///< maximum slider value
    mutable float value; ///< current slider value
    float numTicks; ///< number of tickmarks on slider dial
    int precision; ///< precision of slider value display: number of decimals

    bool valueChanged; ///< indicates a value change since last readout
    bool active_; ///< flag if slider is active

    virtual bool update();

    long lastPressAction;
};
}
#endif
