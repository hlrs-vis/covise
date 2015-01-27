/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++:
#include <iostream>
#include <fstream>
#include <stdio.h>

// OSG:
#include <osg/Geode>
#include <osgText/Text>
#include <osgDB/ReadFile>

// Local:
#include "CUI.h"
#include "Dial.h"
#include "Interaction.h"

using namespace osg;
using namespace cui;
using namespace std;

Dial::Dial(Interaction *interaction, AdvancedInput input)
    : Card(interaction)
    , CalculatorListener()
    , FloatOMeterListener()
{
    _input = input;

    _value = 0.0;
    _hasMin = _hasMax = false;
    _knobRange = 1.0f;
    _isInteger = false;
    _icon->setImage(osgDB::readImageFile(_resourcePath + "dial.tif"));

    this->setFont(osgText::readFontFile("arial.ttf"));
    _valueString = new osgText::Text();
    _valueString->setDataVariance(Object::DYNAMIC);
    createValueString();

    _floatOMeter = 0;
    _floatOMeterTrans = 0;
    _calc = 0;
    _calcTrans = 0;

    if (_input == FLOATOMETER)
        createFloatOMeter();
    else if (_input == CALCULATOR)
        createCalculator();

    updateValueString();
}

void Dial::createValueString()
{
    Geode *geode;

    // Create drawable:
    _valueString->setFont(_font);
    _valueString->setColor(COL_BLACK);
    _valueString->setFontResolution(20, 20);
    _stringPos.set(0.0, -DEFAULT_CARD_HEIGHT / 2.0 + (DEFAULT_CARD_HEIGHT - DEFAULT_CARD_WIDTH + DEFAULT_CARD_WIDTH / 2.0), 2 * EPSILON_Z);
    _valueString->setPosition(_stringPos);
    _valueString->setCharacterSize(DEFAULT_FONT_SIZE * 0.3 / 0.25);
    _valueString->setMaximumWidth(2.0);
    _valueString->setMaximumHeight(1.0);
    _valueString->setAlignment(osgText::Text::CENTER_CENTER);

    // Create geode:
    geode = new Geode();
    geode->addDrawable(_valueString);
    // ignore in intersection test
    geode->setNodeMask(~1);

    _magnifyXF->addChild(geode);
}

void Dial::createFloatOMeter()
{
    Matrix mat;

    if (_floatOMeter)
        return;

    if (!_floatOMeterTrans)
    {
        _floatOMeter = new FloatOMeter(_interaction);
        _floatOMeter->addListener(this);
        _floatOMeter->setVisible(false);

        mat.setTrans(Vec3(0.0, DEFAULT_CARD_HEIGHT / 2.0 + _floatOMeter->getHeight() / 2.0, 0.0));
        _floatOMeterTrans = new MatrixTransform();
        _floatOMeterTrans->setMatrix(mat);

        _floatOMeterTrans->addChild(_floatOMeter->getNode());
        _floatOMeterTrans->ref();
    }

    _magnifyXF->addChild(_floatOMeterTrans);

    _turnAngle = 0.0;
    _knobRange = 1.0;

    if (_logFile)
        _floatOMeter->setLogFile(_logFile);
}

void Dial::createCalculator()
{
    Matrix mat;

    if (_calc)
        return;

    if (!_calcTrans)
    {
        _calc = new Calculator(_interaction);
        _calc->addListener(this);
        _calc->setVisible(false);

        mat.setTrans(Vec3(0.0, 0.0, 0.25));
        _calcTrans = new MatrixTransform();
        _calcTrans->setMatrix(mat);

        _calcTrans->addChild(_calc->getNode());
        _calcTrans->ref();
    }

    _magnifyXF->addChild(_calcTrans);

    _turnAngle = 0.0;
    _knobRange = 1.0;

    if (_logFile)
        _calc->setLogFile(_logFile);
}

void Dial::setLogFile(LogFile *lf)
{
    Widget::setLogFile(lf);

    if (_input == FLOATOMETER)
        _floatOMeter->setLogFile(_logFile);
    else if (_input == CALCULATOR)
        _calc->setLogFile(_logFile);
}

void Dial::setAdvancedInput(AdvancedInput input)
{
    if (input == _input)
        return;

    if (_input == FLOATOMETER)
    {
        _floatOMeter->setVisible(false);
        //       if (_magnifyXF->removeChild(_floatOMeterTrans))
        // 	std::cerr << "floatOMeter removed" << endl;
        //       else
        // 	std::cerr << "floatOMeter not removed" << endl;
    }
    else if (_input == CALCULATOR)
    {
        _calc->setVisible(false);
        //       if (_magnifyXF->removeChild(_calcTrans))
        // 	std::cerr << "calculator removed" << endl;
        //       else
        // 	std::cerr << "calculator not removed" << endl;
    }

    _input = input;

    if (_input == FLOATOMETER)
        createFloatOMeter();
    else if (_input == CALCULATOR)
        createCalculator();
}

void Dial::updateValueString()
{
    char buf1[64], buf2[64], buf[64];
    char *tmp1, *tmp2;
    int min = 0;

    if (_isInteger)
    {
        sprintf(buf, "%d", int(floor(_value)));
    }
    else
    {
        if (_input != NO)
        {
            if (_input == FLOATOMETER)
                min = _floatOMeter->getDecimalPlaces();
            else if (_input == CALCULATOR)
                min = _calc->getDecimalPlaces();

            sprintf(buf1, "%1.2e", _value);
            sprintf(buf2, "%.*lf", abs(min), _value);

            setTipText(buf2, true);

            tmp1 = strtok(buf1, "e");
            tmp2 = strtok(NULL, "");

            if ((strlen(buf2) > 6) && (strlen(tmp1) < 6) && (strlen(tmp2) < 6))
            {
                if (tmp2 == 0)
                    sprintf(buf, "%s", tmp1);
                else
                    sprintf(buf, "%s\ne%s", tmp1, tmp2);

                setTipVisibility(true);
            }
            else
            {
                setTipVisibility(false);
                strcpy(buf, buf2);
            }
        }
        else
        {
            sprintf(buf, "%2.1lf", _value);
        }
    }

    _valueString->setText(buf);
}

void Dial::setValue(double newValue, bool triggerEvent)
{
    updateValue(newValue, triggerEvent);
    if (_input == FLOATOMETER)
        _floatOMeter->setValue(_value, true);
    else if (_input == CALCULATOR)
        _calc->setValue(_value, true);
}

void Dial::setValue(int newValue, bool triggerEvent)
{
    setValue(double(newValue), triggerEvent);
}

void Dial::updateValue(double newValue, bool triggerEvent)
{
    // Constrain value to valid range:
    if (_hasMin && newValue < _min)
        _value = _min;
    else if (_hasMax && newValue > _max)
        _value = _max;
    else
        // Update value:
        _value = newValue;

    updateValueString();

    if (_logFile)
    {
        sprintf(_logBuf, "Dial value updated:\t%lf", _value);
        _logFile->addLog(_logBuf);
    }

    if (triggerEvent)
    {
        std::list<DialChangeListener *>::iterator iter;
        for (iter = _dialListeners.begin(); iter != _dialListeners.end(); ++iter)
        {
            (*iter)->dialValueChanged(this, _value);
        }
    }
}

double Dial::getValue()
{
    if (_isInteger)
        return floor(_value);
    else
        return _value;
}

void Dial::setInteger(bool newValue)
{
    _isInteger = newValue;
}

/** @param newRange if <0 set reasonable knob range according to min and max
 */
void Dial::setKnobRange(float newRange)
{
    if (newRange < 0.0f)
    {
        if (_hasMin && _hasMax)
        {
            _knobRange = (_max - _min) * 3.0f;
        }
        else
            _knobRange = 10.0f;
    }
    else
        _knobRange = newRange;
}

void Dial::turn(Matrix &lastWand2w, Matrix &newWand2w)
{
    double diffAngle = angleDiff(lastWand2w, newWand2w, Widget::Z);
    double tmp;

    if ((_input == FLOATOMETER) && _floatOMeter->isDigitMarked())
    {
        _turnAngle += diffAngle;

        while ((_turnAngle > DigitLabel::STEP_ANGLE) || (_turnAngle < -DigitLabel::STEP_ANGLE))
        {
            if (_turnAngle > 0)
            {
                tmp = _value + pow(10.0, _floatOMeter->getMarkedPlace());
                _turnAngle -= DigitLabel::STEP_ANGLE;
            }
            else
            {
                tmp = _value - pow(10.0, _floatOMeter->getMarkedPlace());
                _turnAngle += DigitLabel::STEP_ANGLE;
            }

            updateValue(tmp, true);

            _floatOMeter->setValue(_value, false);
        }
    }
    else if (_input == CALCULATOR)
    {
        _turnAngle += diffAngle;

        while ((_turnAngle > DigitLabel::STEP_ANGLE) || (_turnAngle < -DigitLabel::STEP_ANGLE))
        {
            if (_turnAngle > 0)
            {
                tmp = _value + pow(10.0, _calc->getDecimalPlaces());
                _turnAngle -= DigitLabel::STEP_ANGLE;
            }
            else
            {
                tmp = _value - pow(10.0, _calc->getDecimalPlaces());
                _turnAngle += DigitLabel::STEP_ANGLE;
            }

            updateValue(tmp, true);

            _calc->setValue(_value, false);
        }
    }
    else
    {
        tmp = _value + diffAngle / 360.0f * _knobRange;

        updateValue(tmp, true);
    }
}

void Dial::cursorEnter(InputDevice *dev)
{
    Card::cursorEnter(dev);

    if (_logFile)
        _logFile->addLog("Dial cursor entered");
}

void Dial::cursorUpdate(InputDevice *dev)
{
    Matrix lastI2W, I2W;

    Card::cursorUpdate(dev);

    if (dev->getButtonState(0) == 1)
    {
        lastI2W = dev->getLastI2W();
        I2W = dev->getI2W();
        turn(lastI2W, I2W);
    }
}

void Dial::cursorLeave(InputDevice *dev)
{
    Card::cursorLeave(dev);

    if (_logFile)
        _logFile->addLog("Dial cursor left");
}

bool Dial::isCursorInArea(Vec3 isect)
{
    Vec3 local;
    Matrix w2l; // world to local matrix
    Matrix l2w; // local to world matrix

    l2w = CUI::computeLocal2Root(getNode());
    w2l = Matrix::inverse(l2w);

    local = isect * w2l;

    if (local[1] > getHeight() / 3.0)
        return true;
    else
        return false;
}

void Dial::setMin(double min)
{
    if (_input == NO)
    {
        _hasMin = true;
        _min = min;
    }
    //   else if (_input == FLOATOMETER)
    //     {
    //       _hasMin = true;
    //       _min = min;
    //       _floatOMeter->setMinValue(_min);
    //     }
    else
        cerr << "no min value possible for the dial - still to do" << endl;
}

void Dial::setMax(double max)
{
    if (_input == NO)
    {
        _hasMax = true;
        _max = max;
    }
    //   else if (_input == FLOATOMETER)
    //     {
    //       _hasMax = true;
    //       _max = max;
    //       _floatOMeter->setMaxValue(_max);
    //     }
    else
        std::cerr << "no max value possible for the dial - still to do" << endl;
}

void Dial::addDialChangeListener(DialChangeListener *d)
{
    _dialListeners.push_back(d);
}

void Dial::buttonEvent(InputDevice *dev, int button)
{
    std::list<CardListener *>::iterator iter;
    if (button == 0)
    {
        for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
        {
            (*iter)->cardButtonEvent(this, 0, dev->getButtonState(0));
        }
    }

    if ((button == 2) && (dev->getButtonState(button) == 1))
    {
        if (_input == CALCULATOR)
        {
            if (!_calc->isVisible())
            {
                _calc->setValue(_value);
                _calc->setVisible(true);
            }
        }
        else if (_input == FLOATOMETER)
        {
            if (!_floatOMeter->isVisible())
            {
                _floatOMeter->setValue(_value, false);
                _floatOMeter->setVisible(true);
            }
        }
    }
}

void Dial::wheelEvent(InputDevice *, int dir)
{
    updateValue(_value + float(dir) / 30.0f * _knobRange);
    std::list<DialChangeListener *>::iterator iter;
    for (iter = _dialListeners.begin(); iter != _dialListeners.end(); ++iter)
    {
        (*iter)->dialValueChanged(this, _value);
    }
}

void Dial::valueSet(double value)
{
    updateValue(value, true);
}

void Dial::floatOMeterValueChanged(double value)
{
    updateValue(value, true);
}
