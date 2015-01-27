/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++
#include <iostream>

// Local:
#include "FloatOMeter.h"
#include "Interaction.h"
#include "DigitPanel.h"

using namespace cui;
using namespace osg;
using namespace std;

const int FloatOMeter::INTEGER_PLACES = 7;
const int FloatOMeter::DECIMAL_PLACES = -8;
const float FloatOMeter::SCALE = 0.75;

FloatOMeter::FloatOMeter(Interaction *interaction)
    : Widget()
    , Events()
    , DigitListener()
{
    _interaction = interaction;

    _value = 0.0;
    _savedValues.push_back(0.0);
    _indexSavedValues = 0;

    _minValue = -DBL_MAX;
    _maxValue = DBL_MAX;

    _integerPlace = _savedIntegerPlace = 0;
    _decimalPlace = _savedDecimalPlace = 0;

    _maxIntegerPlaces = INTEGER_PLACES;

    setGroupID(getNewGroupID());

    createPanel();

    _interaction->addAnyButtonListener(this, this);
}

FloatOMeter::~FloatOMeter()
{
    _listeners.clear();

    _interaction->removeAnyButtonListener(this);
}

void FloatOMeter::createPanel()
{
    DigitLabel *tmp;
    int pos = 0;

    _digitPanel = new DigitPanel(_interaction, Panel::STATIC, Panel::NON_MOVABLE);
    _node->addChild(_digitPanel->getNode());

    pos = -(_integerPlace + 3);

    // create DigitLabel for left arrow
    _leftArrow = new DigitLabel(_interaction, SCALE);
    _leftArrow->addDigitListener(this);
    _digitPanel->addDigit(_leftArrow, pos, 0);
    _leftArrow->setDigitText("<");
    _leftArrow->enableInteraction(false);
    _leftArrow->setGroupID(getGroupID());
    pos++;

    // create DigitLabel for sign
    _sign = _savedSign = false;
    _signLabel = new DigitLabel(_interaction, SCALE);
    _signLabel->addDigitListener(this);
    _digitPanel->addDigit(_signLabel, pos, 0);
    _signLabel->setDigitText("+");
    _signLabel->enableInteraction(false);
    _signLabel->setGroupID(getGroupID());
    pos++;

    // create DigitLabels for digits left to decimal point
    for (int i = _integerPlace; i > -1; i--, pos++)
    {
        tmp = new DigitLabel(_interaction, SCALE);
        tmp->addDigitListener(this);
        tmp->setGroupID(getGroupID());
        _digitPanel->addDigit(tmp, pos, 0);
        _digits.push_back(tmp);
    }

    // initalize _markedDigit to the first digit left to the decimal point
    _markedDigit = *(--_digits.end());
    _markedDigit->highlight(true);
    _markedDigitPlace = 0;
    _isDigitMarked = true;

    // create DigitalLabel for decimal point
    _point = new DigitLabel(_interaction, SCALE);
    _point->addDigitListener(this);
    _digitPanel->addDigit(_point, 0, 0);
    _point->setDigitText(".");
    _point->enableInteraction(false);
    _point->setGroupID(getGroupID());
    pos++;

    // create DigitalLabels for digits right to decimal point
    for (int i = -1; i > (_decimalPlace - 1); i--, pos++)
    {
        tmp = new DigitLabel(_interaction, SCALE);
        tmp->addDigitListener(this);
        tmp->setGroupID(getGroupID());
        _digitPanel->addDigit(tmp, pos, 0);
        _digits.push_back(tmp);
    }

    // create DigitLabel for right arrow
    _rightArrow = new DigitLabel(_interaction, SCALE);
    _rightArrow->addDigitListener(this);
    _digitPanel->addDigit(_rightArrow, pos, 0);
    _rightArrow->setDigitText(">");
    _rightArrow->enableInteraction(false);
    _rightArrow->setGroupID(getGroupID());

    _set = new DigitLabel(_interaction, SCALE);
    _set->addDigitListener(this);
    _digitPanel->addDigit(_set, -1, -1);
    _set->setDigitText("S");
    _set->enableInteraction(false);
    _set->setVisible(true);
    _set->setGroupID(getGroupID());

    _restore = new DigitLabel(_interaction, SCALE);
    _restore->addDigitListener(this);
    _digitPanel->addDigit(_restore, 0, -1);
    _restore->setDigitText("R");
    _restore->enableInteraction(false);
    _restore->setVisible(true);
    _restore->setGroupID(getGroupID());

    _clear = new DigitLabel(_interaction, SCALE);
    _clear->addDigitListener(this);
    _digitPanel->addDigit(_clear, 1, -1);
    _clear->setDigitText("C");
    _clear->enableInteraction(false);
    _clear->setVisible(true);
    _clear->setGroupID(getGroupID());
}

void FloatOMeter::setLogFile(LogFile *lf)
{
    Widget::setLogFile(lf);

    list<DigitLabel *>::iterator iter;

    for (iter = _digits.begin(); iter != _digits.end(); iter++)
        (*iter)->setLogFile(_logFile);

    _leftArrow->setLogFile(_logFile);
    _rightArrow->setLogFile(_logFile);
    _signLabel->setLogFile(_logFile);
    _set->setLogFile(_logFile);
    _restore->setLogFile(_logFile);
    _clear->setLogFile(_logFile);
}

void FloatOMeter::addListener(FloatOMeterListener *listener)
{
    _listeners.push_back(listener);
}

bool FloatOMeter::addDigitToPanel(Side side)
{
    DigitLabel *tmp;

    switch (side)
    {
    case LEFT:
        if (_integerPlace == _maxIntegerPlaces)
            return false;

        _integerPlace += 1;

        if (_logFile)
        {
            sprintf(_logBuf, "add digit to place:\t%d", _integerPlace);
            _logFile->addLog(_logBuf);
        }

        _digitPanel->setDigitPos(_leftArrow, -(_integerPlace + 3), 0);
        _digitPanel->setDigitPos(_signLabel, -(_integerPlace + 2), 0);

        tmp = new DigitLabel(_interaction, SCALE);
        tmp->addDigitListener(this);
        tmp->setGroupID(getGroupID());
        tmp->setLogFile(_logFile);
        _digitPanel->addDigit(tmp, -(_integerPlace + 1), 0);
        _digits.push_front(tmp);
        break;
    case RIGHT:
        if (_decimalPlace == DECIMAL_PLACES)
            return false;

        _decimalPlace -= 1;

        if (_logFile)
        {
            sprintf(_logBuf, "add digit to place:\t%d", _decimalPlace);
            _logFile->addLog(_logBuf);
        }

        _digitPanel->setDigitPos(_rightArrow, abs(_decimalPlace) + 1, 0);

        tmp = new DigitLabel(_interaction, SCALE);
        tmp->addDigitListener(this);
        tmp->setGroupID(getGroupID());
        tmp->setLogFile(_logFile);
        _digitPanel->addDigit(tmp, abs(_decimalPlace), 0);
        _digits.push_back(tmp);
        break;
    }

    return true;
}

bool FloatOMeter::removeDigitFromPanel(Side side)
{
    DigitLabel *tmp;

    switch (side)
    {
    case LEFT:
        if (_integerPlace == 0)
            return false;

        tmp = _digits.front();

        if (tmp == _markedDigit)
            return false;

        if (_logFile)
        {
            sprintf(_logBuf, "remove digit from place:\t%d", _integerPlace);
            _logFile->addLog(_logBuf);
        }

        _integerPlace -= 1;
        _digitPanel->removeDigit(tmp);
        _digitPanel->setDigitPos(_leftArrow, -(_integerPlace + 3), 0);
        _digitPanel->setDigitPos(_signLabel, -(_integerPlace + 2), 0);
        _digits.pop_front();

        break;
    case RIGHT:
        if (_decimalPlace == 0)
            return false;

        tmp = _digits.back();

        if (tmp == _markedDigit)
            return false;

        if (_logFile)
        {
            sprintf(_logBuf, "remove digit from place:\t%d", _decimalPlace);
            _logFile->addLog(_logBuf);
        }

        _decimalPlace += 1;
        _digitPanel->removeDigit(tmp);
        _digitPanel->setDigitPos(_rightArrow, abs(_decimalPlace) + 1, 0);
        _digits.pop_back();
        break;
    }

    return true;
}

void FloatOMeter::updateDigits(bool clear, bool setValue)
{
    char buf[64];
    int i, max, min, digit; //, num
    list<DigitLabel *>::iterator iter;

    getMinMaxPlaces(clear, setValue, min, max);

    // set correct sign
    if (_sign)
        _signLabel->setDigitText("-");
    else
        _signLabel->setDigitText("+");

    // adjust number of digits before point
    if (max > _integerPlace)
    {
        for (i = _integerPlace; i < max; i++)
            if (!addDigitToPanel(LEFT))
                break;
    }
    else if (max < _integerPlace)
    {
        for (i = _integerPlace; i > max; i--)
            if (!removeDigitFromPanel(LEFT))
                break;
    }

    // adjust number of digits behind point
    if (min > _decimalPlace)
    {
        for (i = _decimalPlace; i < min; i++)
            if (!removeDigitFromPanel(RIGHT))
                break;
    }
    else if (min < _decimalPlace)
    {
        for (i = _decimalPlace; i > min; i--)
            if (!addDigitToPanel(RIGHT))
                break;
    }

    //num = sprintf(buf, "%1.*lf", abs(_decimalPlace), _value);

    // skip leading zero positions
    for (i = _markedDigitPlace, iter = _digits.begin(); (_isDigitMarked && (i > max)); i--, iter++)
        (*iter)->setDigit(0);

    // update DigitLabels
    for (i = 0; iter != _digits.end(); i++)
    {
        if ((buf[i] != '.') && (buf[i] != '-'))
        {
            digit = (int)(buf[i] - '0');
            (*iter)->setDigit(digit);

            iter++;
        }
    }

    if (_logFile)
    {
        sprintf(_logBuf, "digits updated - new value is:\t%1.*lf", abs(_decimalPlace), _value);
        _logFile->addLog(_logBuf);
    }

    for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
        (*_iter)->floatOMeterValueChanged(_value);
}

void FloatOMeter::digitLabelUpdate(DigitLabel *digit)
{
    char buf[64];
    double tmp;

    if (_leftArrow == digit)
    {
        addDigitToPanel(LEFT);
        sprintf(_logBuf, "floatOMeter label pressed:\t'left arrow'");
    }
    else if (_rightArrow == digit)
    {
        addDigitToPanel(RIGHT);
        sprintf(_logBuf, "floatOMeter label pressed:\t'right arrow'");
    }
    else if (_signLabel == digit)
    {
        sprintf(_logBuf, "floatOMeter label pressed:\t'sign'");

        if (_sign)
        {
            _sign = false;
            _signLabel->setDigitText("+");
        }
        else
        {
            _sign = true;
            _signLabel->setDigitText("-");
        }

        _value *= -1.0;

        for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
            (*_iter)->floatOMeterValueChanged(_value);
    }
    else if (_set == digit)
    {
        sprintf(_logBuf, "floatOMeter label pressed:\t'set'");

        sprintf(buf, "%.*lf", abs(_decimalPlace), _value);
        if (sscanf(buf, "%lf", &tmp) != 1)
        {
            cerr << "cui::FlotOMeter::updateDigits: sscanf failed" << endl;
        }

        _savedValues.push_back(tmp);
        _indexSavedValues = _savedValues.size() - 1;

        _savedIntegerPlace = _integerPlace;
        _savedDecimalPlace = _decimalPlace;
        _savedSign = _sign;
    }
    else if (_restore == digit)
    {
        sprintf(_logBuf, "floatOMeter label pressed:\t'restore'");

        _value = _savedValues[_indexSavedValues];
        updateDigits(false, true);
    }
    else if (_clear == digit)
    {
        sprintf(_logBuf, "floatOMeter label pressed:\t'clear'");

        if (0.0 < _minValue)
            _value = _minValue;
        else if (0.0 > _maxValue)
            _value = _maxValue;
        else
            _value = 0.0;

        updateDigits(false, true);
    }
    else
        return;

    if (_logFile)
        _logFile->addLog(_logBuf);
}

void FloatOMeter::digitValueUpdate(DigitLabel *digit)
{
    double tmp = 0.0;
    int place;
    std::list<DigitLabel *>::iterator iter;

    for (iter = _digits.begin(), place = _integerPlace; iter != _digits.end(); iter++, place--)
    {
        tmp += (*iter)->getDigit() * pow(10.0f, place);
        if (_logFile && ((*iter) == digit))
        {
            sprintf(_logBuf, "floatOMeter: value of place\t%d\t has changed - new digit is:\t%d", place, (*iter)->getDigit());
            _logFile->addLog(_logBuf);
        }
    }

    if (_sign)
        tmp *= -1;

    if ((tmp >= _minValue) && (tmp <= _maxValue))
        _value = tmp;
    else
        updateDigits(false, false);

    for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
        (*_iter)->floatOMeterValueChanged(_value);
}

void FloatOMeter::digitMarked(DigitLabel *digit)
{
    int place;
    list<DigitLabel *>::iterator iter;

    for (iter = _digits.begin(), place = _integerPlace; iter != _digits.end(); iter++, place--)
    {
        if ((*iter) == digit)
        {
            if (_markedDigit == digit)
            {
                sprintf(_logBuf, "floatOMeter digit at place\t%d\t unmarked", place);

                _markedDigit->highlight(false);
                _markedDigit = 0;
                _isDigitMarked = false;
            }
            else
            {
                sprintf(_logBuf, "floatOMeter digit at place\t%d\t marked", place);

                if (_markedDigit != 0)
                    _markedDigit->highlight(false);
                _markedDigit = digit;
                _markedDigitPlace = place;
                _markedDigit->highlight(true);
                _isDigitMarked = true;
            }

            if (_logFile)
                _logFile->addLog(_logBuf);

            return;
        }
    }
}

bool FloatOMeter::fallBelowMin(DigitLabel *digit)
{
    int place;
    list<DigitLabel *>::iterator iter;

    if (digit == _restore)
    {
        if (_indexSavedValues > 0)
        {
            _indexSavedValues--;
            _value = _savedValues[_indexSavedValues];
            updateDigits();
            return true;
        }
        else
            return false;
    }

    for (iter = _digits.begin(), place = _integerPlace; iter != _digits.end(); iter++, place--)
    {
        if ((*iter) == digit)
            break;
    }

    if ((iter == _digits.end()) || (iter == _digits.begin()))
        return false;

    if (_logFile)
    {
        sprintf(_logBuf, "digit at place\t%d\t fell below 0", place);
        _logFile->addLog(_logBuf);
    }

    iter--;

    if ((*iter)->decreaseValue())
    {
        if (_logFile)
        {
            sprintf(_logBuf, "digit at place\t%d\t decreased", place + 1);
            _logFile->addLog(_logBuf);
        }

        return true;
    }
    else
        return false;
}

bool FloatOMeter::passOverMax(DigitLabel *digit)
{
    int place;
    list<DigitLabel *>::iterator iter;

    if (digit == _restore)
    {
        if (_indexSavedValues < (((int)_savedValues.size()) - 1))
        {
            _indexSavedValues++;
            _value = _savedValues[_indexSavedValues];
            updateDigits();
            return true;
        }
        else
            return false;
    }

    for (iter = _digits.begin(), place = _integerPlace; iter != _digits.end(); iter++, place--)
    {
        if ((*iter) == digit)
            break;
    }

    if (iter == _digits.end())
        return false;

    if (_logFile)
    {
        sprintf(_logBuf, "digit at place\t%d\t pass over 9", place);
        _logFile->addLog(_logBuf);
    }

    if (iter == _digits.begin())
    {
        if (addDigitToPanel(LEFT))
        {
            if ((*_digits.begin())->increaseValue())
            {
                if (_logFile)
                {
                    sprintf(_logBuf, "digit at place\t%d\t increased", place + 1);
                    _logFile->addLog(_logBuf);
                }

                return true;
            }
            else
                return false;
        }
        else
            return false;
    }

    iter--;

    if ((*iter)->increaseValue())
    {
        if (_logFile)
        {
            sprintf(_logBuf, "digit at place\t%d\t increased", place + 1);
            _logFile->addLog(_logBuf);
        }

        return true;
    }
    else
        return false;
}

void FloatOMeter::getMinMaxPlaces(bool clear, bool setValue, int &min, int &max)
{
    char buf[64];
    int i, num, lastDigit;

    if (setValue)
        num = sprintf(buf, "%1.*lf", abs(DECIMAL_PLACES), _value);
    else
        num = sprintf(buf, "%1.*lf", abs(_decimalPlace), _value);

    _sign = false;

    // get digits in front of decimal point
    for (i = 0, max = -1; i < num; i++)
    {
        if (buf[i] == '.')
            break;
        else if (buf[i] == '-')
            _sign = true;
        else
            max++;
    }

    if (clear || setValue)
    {
        // get digits behind decimal point
        for (i++, lastDigit = 0, min = 0; i < num; i++)
        {
            min--;
            if (buf[i] != '0')
                lastDigit = min;
        }
        min = lastDigit;
    }
    else
        min = _decimalPlace;
}

void FloatOMeter::setValue(double value, bool setValue)
{
    if ((value >= _minValue) && (value <= _maxValue))
        _value = value;
    else
        return;

    updateDigits(false, setValue);
}

void FloatOMeter::setVisible(bool flag)
{
    Widget::setVisible(flag);

    if (flag)
        for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
            (*_iter)->floatOMeterOpened();
    else
        for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
            (*_iter)->floatOMeterClosed();
}

void FloatOMeter::buttonEvent(InputDevice *dev, int button)
{
    if (isVisible() && (dev->getButtonState(button) == 1) && ((dev->getIsectWidget() == 0) || (dev->getIsectWidget()->getGroupID() != getGroupID())))
    {
        setVisible(false);
        updateDigits(true);
    }
}
