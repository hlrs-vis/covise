/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VALUEPOTI_H
#define CO_VALUEPOTI_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coButtonGeometry.h>
#include <OpenVRUI/coUIElement.h>
#include <OpenVRUI/coUpdateManager.h>

#include <OpenVRUI/sginterface/vruiCollabInterface.h>

#include <string>

namespace vrui
{

class coValuePoti;
class coCombinedButtonInteraction;
class vruiTransformNode;
class vruiNode;

/** Action listener for coValuePoti events.
 */
class OPENVRUIEXPORT coValuePotiActor
{
public:
    virtual ~coValuePotiActor()
    {
    }
    /** Called whenever poti value was changed.
          @param oldValue previous poti value
          @param newValue poti value which is to be set
          @param poti     poti by which event was triggered
          @param context  remote context for collaborative use (optional)
      */
    virtual void potiValueChanged(float oldValue, float newValue, coValuePoti *poti, int context = -1) = 0;
    virtual void potiPressed(coValuePoti *poti, int context = -1);
    virtual void potiReleased(coValuePoti *poti, int context = -1);
    /** Get the current context, if one poti is used in different contexts (for collaborative use).
          @return collaborative context
      */
    virtual int getContext()
    {
        return -1;
    }
};

/** This class offers a poti GUI element which can be used to set integer or
  floating point values. To change the value, the user clicks on the poti and twists
  the hand to turn the poti. Action events are processed by coValuePotiActor and coAction.
  @see coUIElement
  @see vruiCollabInterface
  @see coUpdateable
*/
class OPENVRUIEXPORT coValuePoti
    : public coAction,
      public coUIElement,
      public vruiCollabInterface,
      public coUpdateable
{
public:
    enum
    {
        MAX_SLOPE = 1000
    };

    coValuePoti(const std::string &buttonText, coValuePotiActor *actor,
                const std::string &backgroundTexture, vruiCOIM *cInterfaceManager = 0,
                const std::string &interfaceName = "");

    virtual ~coValuePoti();

    virtual float getWidth() const // see superclass for comment
    {
        return 67.0f * xScaleFactor;
    }
    virtual float getHeight() const // see superclass for comment
    {
        return 60.0f * yScaleFactor;
    }
    virtual float getXpos() const // see superclass for comment
    {
        return myX;
    }
    virtual float getYpos() const // see superclass for comment
    {
        return myY;
    }

    virtual bool update();
    virtual void setValue(float value);
    virtual void setMin(float min);
    virtual void setMax(float max);
    virtual float getValue() const;
    virtual int hit(vruiHit *hit);
    // miss is called once after a hit, if the button is not intersected
    // anymore
    virtual void miss();

    void joystickUp();
    void joystickDown();

    void setPos(float x, float y, float z = 0.0f);
    void setSize(float size);
    void setSize(float width, float height, float depth);
    void setInteger(bool on);
    void setIncrement(float increment);
    void setLogarithmic(bool on);

    void setState(float min, float max, float value,
                  bool isInt, float inc = 0.0f);

    virtual float getMin() const;
    virtual float getMax() const;
    virtual bool isInteger() const;
    virtual bool isDiscrete() const;
    virtual float getIncrement() const;
    virtual bool isLogarithmic() const;

    virtual float getXSize() const;
    virtual float getYSize() const;
    virtual float getZSize() const;

    virtual void setLabelVisible(bool);
    virtual bool isLabelVisible();

    const std::string &getBackgroundTexture() const
    {
        return backgroundTexture;
    }
    const std::string &getButtonText() const
    {
        return buttonText;
    }

    float discreteValue(float value) const;

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

protected:
    float myX; ///< poti x location
    float myY; ///< poti y location
    float minValue; ///< poti minimum value
    float maxValue; ///< poti maximum value
    float value; ///< current value represented by this poti
    coCombinedButtonInteraction *interactionA; ///< interaction for first button
    coCombinedButtonInteraction *interactionB; ///< interaction for second button (fine Tune)
    coCombinedButtonInteraction *interactionW; ///< wheel interaction
    bool unregister; ///< true if the interaction should be unregistered
    float lastRoll; ///< previous mouse orientation
    std::string buttonText; ///< value display text
    std::string baseButtonText;
    std::string backgroundTexture;
    coValuePotiActor *myActor; ///< action listener
    bool integer; ///< true if adjustable value is an integer
    float increment; ///< step size this poti, 0.0 if continuous
    bool discrete;
    bool logarithmic = false; ///< true if scale is logarithmic
    bool labelVisible; ///< true if label is visible

    virtual void remoteLock(const char *message);
    virtual void remoteOngoing(const char *message);
    virtual void releaseRemoteLock(const char *message);
    virtual void setEnabled(bool on);

    virtual void setText(const std::string &buttonText);

    void displayValue(float value);
};

/** A special rotary knob to adjust slope values.
  The angle of the rotary knob represents the selected slope value.
  @see coValuePoti
*/
class OPENVRUIEXPORT coSlopePoti : public coValuePoti
{
public:
    coSlopePoti(const std::string &buttonText, coValuePotiActor *actor,
                const std::string &backgroundTexture, vruiCOIM *cInterfaceManager = 0,
                const std::string &interfaceName = "");

    virtual void setValue(float);
    virtual bool update();

    virtual void remoteOngoing(const char *message);
    virtual void setABS(bool on);
    virtual bool getABS() const;
    virtual float getValue() const;

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    float convertSlopeToLinear(float) const;
    float convertLinearToSlope(float) const;

protected:
    bool positive; ///< true=positive values only, false=negative values are allowed

    void setLinearValue(float value);
};
}
#endif
