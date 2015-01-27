/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_FLOATOMETER_H_
#define _CUI_FLOATOMETER_H_

#include "Widget.h"
#include "DigitLabel.h"

namespace cui
{
class Interaction;
class FloatOMeterListener;
class DigitPanel;

class CUIEXPORT FloatOMeter : public Widget, public Events, public DigitListener
{
public:
    enum Side
    {
        LEFT,
        RIGHT
    };

    FloatOMeter(Interaction *);
    ~FloatOMeter();

    void updateDigits(bool clear = false, bool setValue = false);
    void setValue(double, bool = false);
    void addListener(FloatOMeterListener *);
    bool isDigitMarked()
    {
        return _isDigitMarked;
    }
    DigitLabel *getMarkedDigit()
    {
        return _markedDigit;
    }
    int getMarkedPlace()
    {
        return _markedDigitPlace;
    }
    int getDecimalPlaces()
    {
        return _decimalPlace;
    }
    float getHeight()
    {
        return _point->getHeight();
    }
    void setMaxIntegerPlaces(int max)
    {
        _maxIntegerPlaces = max - 1;
    }
    void setMinValue(double min)
    {
        _minValue = min;
    }
    void setMaxValue(double max)
    {
        _maxValue = max;
    }

    virtual void setVisible(bool);
    virtual void setLogFile(LogFile *);

private:
    static const int INTEGER_PLACES;
    static const int DECIMAL_PLACES;
    static const float SCALE;

    Interaction *_interaction;

    double _value;
    double _minValue, _maxValue;
    std::vector<double> _savedValues;
    int _indexSavedValues;

    int _integerPlace, _savedIntegerPlace, _decimalPlace, _savedDecimalPlace;
    int _maxIntegerPlaces;
    int _markedDigitPlace;

    bool _sign, _savedSign;
    bool _isDigitMarked;

    std::list<FloatOMeterListener *> _listeners;
    std::list<FloatOMeterListener *>::iterator _iter;

    DigitPanel *_digitPanel;

    std::list<DigitLabel *> _digits;
    DigitLabel *_leftArrow;
    DigitLabel *_rightArrow;
    DigitLabel *_signLabel;
    DigitLabel *_markedDigit;
    DigitLabel *_point;
    DigitLabel *_set;
    DigitLabel *_restore;
    DigitLabel *_clear;

    void createPanel();

    bool addDigitToPanel(Side);
    bool removeDigitFromPanel(Side);

    void getMinMaxPlaces(bool, bool, int &, int &);

    virtual void digitValueUpdate(DigitLabel *);
    virtual void digitLabelUpdate(DigitLabel *);
    virtual void digitMarked(DigitLabel *);
    virtual bool fallBelowMin(DigitLabel *);
    virtual bool passOverMax(DigitLabel *);

    virtual void cursorEnter(InputDevice *){};
    virtual void cursorUpdate(InputDevice *){};
    virtual void cursorLeave(InputDevice *){};
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *){};
    virtual void wheelEvent(InputDevice *, int){};
};

class CUIEXPORT FloatOMeterListener
{
public:
    virtual ~FloatOMeterListener()
    {
    }
    virtual void floatOMeterValueChanged(double) = 0;
    virtual void floatOMeterOpened() = 0;
    virtual void floatOMeterClosed() = 0;
};
}
#endif
