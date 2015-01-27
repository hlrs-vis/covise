/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_DIAL_H_
#define _CUI_DIAL_H_

#include "Widget.h"
#include "Card.h"
#include "Calculator.h"
#include "FloatOMeter.h"

namespace cui
{
class Interaction;
class DialChangeListener;

/** 
      This is the implementation of a push button, which triggers an action.
   */
class CUIEXPORT Dial : public Card, public CalculatorListener, public FloatOMeterListener
{
public:
    enum AdvancedInput
    {
        FLOATOMETER,
        CALCULATOR,
        NO
    };

    Dial(Interaction *, AdvancedInput input = NO);
    virtual ~Dial(){};
    virtual void setValue(double, bool = false);
    virtual void setValue(int, bool = false);
    virtual double getValue();
    virtual void setInteger(bool);
    virtual void setKnobRange(float = -1.0f);
    virtual void setMin(double);
    virtual void setMax(double);
    virtual void turn(osg::Matrix &, osg::Matrix &);
    virtual void addDialChangeListener(DialChangeListener *d);
    FloatOMeter *getFloatOMeter()
    {
        return _floatOMeter;
    }
    Calculator *getCalculator()
    {
        return _calc;
    }
    void setAdvancedInput(AdvancedInput);

    virtual void setLogFile(LogFile *);

protected:
    osgText::Text *_valueString;
    osg::Vec3 _stringPos;

    AdvancedInput _input;
    Calculator *_calc;
    osg::MatrixTransform *_calcTrans;
    FloatOMeter *_floatOMeter;
    osg::MatrixTransform *_floatOMeterTrans;

    double _value; ///< current dial value

    double _min, _max; ///< lower/upper bounds for value range
    double _knobRange; ///< value range of 360 degrees dial rotation
    double _turnAngle;
    bool _hasMin, _hasMax; ///< true = value range limited at lower and/or upper end
    bool _isInteger; ///< true = value is integer
    std::list<DialChangeListener *> _dialListeners;

    void createFloatOMeter();
    void createCalculator();
    virtual void createValueString();

    void updateValue(double, bool = false);
    virtual void updateValueString();

    bool isCursorInArea(osg::Vec3);

    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void wheelEvent(InputDevice *, int);

    virtual void valueSet(double);
    virtual void calculatorClosed(){};
    virtual void calculatorOpened(){};

    virtual void floatOMeterValueChanged(double);
    virtual void floatOMeterOpened(){};
    virtual void floatOMeterClosed(){};
};

/** Derive your class that uses a dial from this class to
    capture dial events.
  */

class CUIEXPORT DialChangeListener
{
public:
    virtual ~DialChangeListener()
    {
    }
    virtual void dialValueChanged(Dial *, float) = 0;
};
}
#endif
