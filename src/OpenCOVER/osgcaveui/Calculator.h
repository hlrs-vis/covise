/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_CALCULATOR_H
#define _CUI_CALCULATOR_H

#include "Widget.h"
#include "DigitLabel.h"

namespace cui
{
class Interaction;
class CalculatorListener;
class DigitPanel;

class CUIEXPORT Calculator : public Widget, public Events, public DigitListener
{
public:
    Calculator(Interaction *);
    ~Calculator();

    void setValue(double, bool = false);
    void addListener(CalculatorListener *);
    int getDecimalPlaces()
    {
        return _decimalPlace;
    }
    int getIntegerPlaces()
    {
        return _integerPlace;
    }
    void setMaxIntegerPlaces(int max)
    {
        _maxIntegerPlaces = max - 1;
    }
    double getValue()
    {
        return _value;
    }

    virtual void setVisible(bool);
    virtual void setLogFile(LogFile *);

private:
    static const int INTEGER_PLACES;
    static const int DECIMAL_PLACES;
    static const int DISPLAY_LENGTH;
    static const float SCALE;

    Interaction *_interaction;

    DigitPanel *_panel;
    DigitLabel *_zero;
    DigitLabel *_one;
    DigitLabel *_two;
    DigitLabel *_three;
    DigitLabel *_four;
    DigitLabel *_five;
    DigitLabel *_six;
    DigitLabel *_seven;
    DigitLabel *_eight;
    DigitLabel *_nine;
    DigitLabel *_set;
    DigitLabel *_restore;
    DigitLabel *_clear;
    DigitLabel *_dot;
    DigitLabel *_sign;
    DigitLabel *_undo;

    osgText::Text *_display;

    std::list<CalculatorListener *> _listeners;
    std::list<CalculatorListener *>::iterator _iter;

    double _value, _savedValue;
    bool _isDot, _savedIsDot;
    bool _isSign, _savedIsSign;
    int _decimalPlace, _savedDecimalPlace;
    int _integerPlace, _savedIntegerPlace;
    int _length, _savedLength;
    int _maxIntegerPlaces;

    void init();
    void createPanel();
    void createDisplay();
    void updateDisplay();
    void undo();
    void close();

    virtual void digitValueUpdate(DigitLabel *);
    virtual void digitLabelUpdate(DigitLabel *);
    virtual void digitMarked(DigitLabel *);
    virtual bool fallBelowMin(DigitLabel *);
    virtual bool passOverMax(DigitLabel *);

    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *);
    virtual void wheelEvent(InputDevice *, int);
};

class CUIEXPORT CalculatorListener
{
public:
    virtual ~CalculatorListener()
    {
    }
    virtual void valueSet(double) = 0;
    virtual void calculatorOpened() = 0;
    virtual void calculatorClosed() = 0;
};
}

#endif
