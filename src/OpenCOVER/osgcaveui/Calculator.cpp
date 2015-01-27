/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Calculator.h"

// C++
#include <iostream>

// OSG
#include <osg/Geometry>
#include <osg/Geode>

#include "DigitPanel.h"

using namespace cui;
using namespace osg;
using namespace std;

const int Calculator::INTEGER_PLACES = 7;
const int Calculator::DECIMAL_PLACES = -8;
const int Calculator::DISPLAY_LENGTH = 16;
const float Calculator::SCALE = 0.75;

Calculator::Calculator(Interaction *interaction)
    : Widget()
    , Events()
    , DigitListener()
{
    _interaction = interaction;

    init();

    _interaction->addAnyButtonListener(this, this);
}

Calculator::~Calculator()
{
    _listeners.clear();

    _interaction->removeAnyButtonListener(this);
}

void Calculator::init()
{
    _value = _savedValue = 0;
    _isDot = _savedIsDot = false;
    _isSign = _savedIsSign = false;
    _integerPlace = _savedIntegerPlace = -1;
    _decimalPlace = _savedDecimalPlace = 0;
    _length = _savedLength = 0;
    _maxIntegerPlaces = INTEGER_PLACES;

    setGroupID(getNewGroupID());

    createPanel();
    createDisplay();
    updateDisplay();
}

void Calculator::setLogFile(LogFile *lf)
{
    Widget::setLogFile(lf);

    _zero->setLogFile(_logFile);
    _one->setLogFile(_logFile);
    _two->setLogFile(_logFile);
    _three->setLogFile(_logFile);
    _four->setLogFile(_logFile);
    _five->setLogFile(_logFile);
    _six->setLogFile(_logFile);
    _seven->setLogFile(_logFile);
    _eight->setLogFile(_logFile);
    _nine->setLogFile(_logFile);
    _set->setLogFile(_logFile);
    _restore->setLogFile(_logFile);
    _clear->setLogFile(_logFile);
    _dot->setLogFile(_logFile);
    _sign->setLogFile(_logFile);
    _undo->setLogFile(_logFile);
}

void Calculator::addListener(CalculatorListener *listener)
{
    _listeners.push_back(listener);
}

void Calculator::createPanel()
{
    MatrixTransform *trans;
    Matrix mat;

    _panel = new DigitPanel(_interaction, Panel::STATIC, Panel::NON_MOVABLE);

    _zero = new DigitLabel(_interaction, SCALE);
    _zero->addDigitListener(this);
    _zero->setDigitText("0");
    _zero->enableInteraction(false);
    _zero->setGroupID(getGroupID());
    _panel->addDigit(_zero, 1, 4);

    _dot = new DigitLabel(_interaction, SCALE);
    _dot->addDigitListener(this);
    _dot->setDigitText(".");
    _dot->enableInteraction(false);
    _dot->setGroupID(getGroupID());
    _panel->addDigit(_dot, 0, 4);

    _sign = new DigitLabel(_interaction, SCALE);
    _sign->addDigitListener(this);
    _sign->enableInteraction(false);
    _sign->setDigitText("+/-");
    _sign->setGroupID(getGroupID());
    _panel->addDigit(_sign, 2, 4);

    _undo = new DigitLabel(_interaction, SCALE);
    _undo->addDigitListener(this);
    _undo->enableInteraction(false);
    _undo->setDigitText("<");
    _undo->setGroupID(getGroupID());
    _panel->addDigit(_undo, 3, 2);

    _one = new DigitLabel(_interaction, SCALE);
    _one->addDigitListener(this);
    _one->setDigitText("1");
    _one->enableInteraction(false);
    _one->setGroupID(getGroupID());
    _panel->addDigit(_one, 0, 3);

    _two = new DigitLabel(_interaction, SCALE);
    _two->addDigitListener(this);
    _two->setDigitText("2");
    _two->enableInteraction(false);
    _two->setGroupID(getGroupID());
    _panel->addDigit(_two, 1, 3);

    _three = new DigitLabel(_interaction, SCALE);
    _three->addDigitListener(this);
    _three->setDigitText("3");
    _three->enableInteraction(false);
    _three->setGroupID(getGroupID());
    _panel->addDigit(_three, 2, 3);

    _four = new DigitLabel(_interaction, SCALE);
    _four->addDigitListener(this);
    _four->setDigitText("4");
    _four->enableInteraction(false);
    _four->setGroupID(getGroupID());
    _panel->addDigit(_four, 0, 2);

    _five = new DigitLabel(_interaction, SCALE);
    _five->addDigitListener(this);
    _five->setDigitText("5");
    _five->enableInteraction(false);
    _five->setGroupID(getGroupID());
    _panel->addDigit(_five, 1, 2);

    _six = new DigitLabel(_interaction, SCALE);
    _six->addDigitListener(this);
    _six->setDigitText("6");
    _six->enableInteraction(false);
    _six->setGroupID(getGroupID());
    _panel->addDigit(_six, 2, 2);

    _seven = new DigitLabel(_interaction, SCALE);
    _seven->addDigitListener(this);
    _seven->setDigitText("7");
    _seven->enableInteraction(false);
    _seven->setGroupID(getGroupID());
    _panel->addDigit(_seven, 0, 1);

    _eight = new DigitLabel(_interaction, SCALE);
    _eight->addDigitListener(this);
    _eight->setDigitText("8");
    _eight->enableInteraction(false);
    _eight->setGroupID(getGroupID());
    _panel->addDigit(_eight, 1, 1);

    _nine = new DigitLabel(_interaction, SCALE);
    _nine->addDigitListener(this);
    _nine->setDigitText("9");
    _nine->enableInteraction(false);
    _nine->setGroupID(getGroupID());
    _panel->addDigit(_nine, 2, 1);

    _set = new DigitLabel(_interaction, SCALE);
    _set->addDigitListener(this);
    _set->setDigitText("S");
    _set->enableInteraction(false);
    _set->setGroupID(getGroupID());
    _panel->addDigit(_set, 0, 0);

    _restore = new DigitLabel(_interaction, SCALE);
    _restore->addDigitListener(this);
    _restore->setDigitText("R");
    _restore->enableInteraction(false);
    _restore->setGroupID(getGroupID());
    _panel->addDigit(_restore, 1, 0);

    _clear = new DigitLabel(_interaction, SCALE);
    _clear->addDigitListener(this);
    _clear->setDigitText("C");
    _clear->enableInteraction(false);
    _clear->setGroupID(getGroupID());
    _panel->addDigit(_clear, 2, 0);

    trans = new MatrixTransform();
    mat = trans->getMatrix();
    mat.setTrans(Vec3(-(_panel->getWidth() - _one->getWidth()) / 2.0 + _one->getWidth() / 2.0,
                      _panel->getHeight() / 2.0 - _one->getHeight() / 2.0, 0.0));
    trans->setMatrix(mat);

    _node->addChild(trans);
    trans->addChild(_panel->getNode());
}

void Calculator::createDisplay()
{
    Geometry *geom, *frame;
    Geode *geode;
    Vec3Array *vertices, *frameVertices;
    Vec4Array *color, *frameColor;
    Vec4 displayColor;
    Vec3 displayPosition;
    StateSet *state;
    MatrixTransform *trans;
    Matrix mat;
    float width2, height2;

    geode = new Geode();
    geode->setNodeMask(~1);
    geom = new Geometry();
    geode->addDrawable(geom);

    width2 = 3;
    height2 = _one->getHeight() / 2.0;

    vertices = new Vec3Array(4);
    (*vertices)[0].set(-width2, -height2, 0.0);
    (*vertices)[1].set(width2, -height2, 0.0);
    (*vertices)[2].set(width2, height2, 0.0);
    (*vertices)[3].set(-width2, height2, 0.0);
    geom->setVertexArray(vertices);

    geom->addPrimitiveSet(new DrawArrays(GL_QUADS, 0, 4));
    //  geom->setUseDisplayList(false);

    color = new Vec4Array(1);
    (*color)[0].set(1.0, 1.0, 1.0, 1.0);
    geom->setColorArray(color);
    geom->setColorBinding(Geometry::BIND_OVERALL);

    state = geom->getOrCreateStateSet();
    state->setMode(GL_LIGHTING, StateAttribute::OFF);

    frame = new Geometry();
    geode->addDrawable(frame);

    frameVertices = new Vec3Array(5);
    (*frameVertices)[0].set(-width2, -height2, EPSILON_Z);
    (*frameVertices)[1].set(width2, -height2, EPSILON_Z);
    (*frameVertices)[2].set(width2, height2, EPSILON_Z);
    (*frameVertices)[3].set(-width2, height2, EPSILON_Z);
    (*frameVertices)[4].set(-width2, -height2, EPSILON_Z);
    frame->setVertexArray(frameVertices);

    frame->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, 5));
    frame->setUseDisplayList(false);

    frameColor = new Vec4Array(1);
    (*frameColor)[0].set(0.0, 0.0, 0.0, 1.0);
    frame->setColorArray(frameColor);
    frame->setColorBinding(Geometry::BIND_OVERALL);

    state = frame->getOrCreateStateSet();
    state->setMode(GL_LIGHTING, StateAttribute::OFF);

    // create display
    _display = new osgText::Text();
    _display->setDataVariance(Object::DYNAMIC);
    this->setFont(osgText::readFontFile("arial.ttf"));
    _display->setFont(_font);
    displayColor = COL_BLACK;
    _display->setColor(displayColor);
    _display->setFontResolution(20, 20);
    displayPosition.set(0.0, 0.0, EPSILON_Z);
    _display->setPosition(displayPosition);
    _display->setCharacterSize(0.5);
    _display->setMaximumWidth(2 * width2);
    _display->setMaximumHeight(2 * height2);
    _display->setAlignment(osgText::Text::CENTER_CENTER);

    geode->addDrawable(_display);

    trans = new MatrixTransform();
    mat = trans->getMatrix();
    mat.setTrans(Vec3(0.0, _panel->getHeight() / 2.0 + _one->getHeight() / 2.0, 2 * EPSILON_Z));
    trans->setMatrix(mat);

    geode->setNodeMask(~1);
    trans->addChild(geode);
    _node->addChild(trans);
}

void Calculator::updateDisplay()
{
    char buf[64];

    //   if (_isSign)
    //     strcpy(buf, "-");
    //   else
    //     strcpy(buf, "");

    sprintf(buf, "%.*lf", abs(_decimalPlace), _value);

    if (_isDot && (_decimalPlace == 0))
        strcat(buf, ".");

    _display->setText(buf);

    for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
        (*_iter)->valueSet(_value);
}

void Calculator::digitValueUpdate(DigitLabel *)
{
}

void Calculator::digitLabelUpdate(DigitLabel *label)
{
    if (label == _dot)
    {
        sprintf(_logBuf, "calculator label pressed:\t'dot'");

        if (_isDot)
            return;
        else
            _isDot = true;
    }
    else if (label == _sign)
    {
        sprintf(_logBuf, "calculator label pressed:\t'sign'");

        if (_isSign)
            _isSign = false;
        else
            _isSign = true;

        _value *= -1.0;
    }
    else if (label == _clear)
    {
        sprintf(_logBuf, "calculator label pressed:\t'clear'");

        _value = 0;
        _integerPlace = -1;
        _decimalPlace = 0;
        _isDot = false;
        _isSign = false;
        _length = 0;
    }
    else if (label == _set)
    {
        sprintf(_logBuf, "calculator label pressed:\t'set'");

        _savedValue = _value;
        _savedIntegerPlace = _integerPlace;
        _savedDecimalPlace = _decimalPlace;
        _savedIsDot = _isDot;
        _savedIsSign = _isSign;
        _savedLength = _length;
    }
    else if (label == _restore)
    {
        sprintf(_logBuf, "calculator label pressed:\t'restore'");

        _value = _savedValue;
        _integerPlace = _savedIntegerPlace;
        _decimalPlace = _savedDecimalPlace;
        _isDot = _savedIsDot;
        _isSign = _savedIsSign;
        _length = _savedLength;
    }
    else if (label == _undo)
    {
        sprintf(_logBuf, "calculator label pressed:\t'undo'");
        undo();
    }
    else if (_length >= DISPLAY_LENGTH)
        return;
    else
    {
        sprintf(_logBuf, "calculator label pressed:\t%d", label->getDigit());
        if ((!_isDot) && (_integerPlace < _maxIntegerPlaces))
        {
            if ((_integerPlace == -1) && (label == _zero))
                return;

            _length++;
            _integerPlace++;

            if (_isSign)
                _value *= -10.0;
            else
                _value *= 10.0;

            if (label == _one)
                _value += 1.0;
            else if (label == _two)
                _value += 2.0;
            else if (label == _three)
                _value += 3.0;
            else if (label == _four)
                _value += 4.0;
            else if (label == _five)
                _value += 5.0;
            else if (label == _six)
                _value += 6.0;
            else if (label == _seven)
                _value += 7.0;
            else if (label == _eight)
                _value += 8.0;
            else if (label == _nine)
                _value += 9.0;

            if (_isSign)
                _value *= -1.0;
        }
        else if (_isDot && (_decimalPlace > DECIMAL_PLACES))
        {
            _length++;
            _decimalPlace--;

            if (_isSign)
                _value *= -1.0;

            if (label == _one)
                _value += 1.0 * pow(10.0f, _decimalPlace);
            else if (label == _two)
                _value += 2.0 * pow(10.0f, _decimalPlace);
            else if (label == _three)
                _value += 3.0 * pow(10.0f, _decimalPlace);
            else if (label == _four)
                _value += 4.0 * pow(10.0f, _decimalPlace);
            else if (label == _five)
                _value += 5.0 * pow(10.0f, _decimalPlace);
            else if (label == _six)
                _value += 6.0 * pow(10.0f, _decimalPlace);
            else if (label == _seven)
                _value += 7.0 * pow(10.0f, _decimalPlace);
            else if (label == _eight)
                _value += 8.0 * pow(10.0f, _decimalPlace);
            else if (label == _nine)
                _value += 9.0 * pow(10.0f, _decimalPlace);

            if (_isSign)
                _value *= -1.0;
        }
    }

    if (_logFile)
        _logFile->addLog(_logBuf);

    updateDisplay();
}

void Calculator::undo()
{
    char buf[64];
    int num;

    if (_isDot && (_decimalPlace == 0))
    {
        _isDot = false;
        return;
    }

    if (_length == 0)
        return;

    if (_isSign)
        _value *= -1.0;

    num = sprintf(buf, "%.*lf", abs(_decimalPlace), _value);

    if (_isDot)
    {
        _value -= double(((int)(buf[num - 1] - '0')) * pow(10.0, _decimalPlace));
        _decimalPlace++;
    }
    else
    {
        _value -= double(((int)(buf[num - 1] - '0')));
        _value /= 10.0;
        _integerPlace--;
    }

    _length--;

    if (_isSign)
        _value *= -1.0;
}

void Calculator::setValue(double value, bool setValue)
{
    char buf[64];
    int i, num, lastDigit;

    if (setValue)
    {
        if (value == 0)
        {
            _value = 0;
            _integerPlace = -1;
            _decimalPlace = 0;
            _isDot = false;
            _isSign = false;
            _length = 0;
        }
        else
        {
            num = sprintf(buf, "%.*lf", abs(DECIMAL_PLACES), value);

            _isSign = false;

            // get _integerPlace
            for (i = 0, _integerPlace = -1, _length = 0; i < num; i++)
            {
                if (buf[i] == '.')
                {
                    _isDot = true;
                    break;
                }
                else if (buf[i] == '-')
                    _isSign = true;
                else
                {
                    _integerPlace++;
                    _length++;
                }
            }

            // get _decimalPlace
            for (i++, lastDigit = 0, _decimalPlace = 0; i < num; i++)
            {
                _decimalPlace--;

                if (buf[i] != '0')
                    lastDigit = _decimalPlace;
            }
            _decimalPlace = lastDigit;

            _length += abs(_decimalPlace);

            _value = value;
        }
    }
    else
    {
        if (value == 0)
        {
            _value = 0;
            _integerPlace = -1;
            _length = abs(_decimalPlace);
            _isSign = false;
        }
        else
        {
            num = sprintf(buf, "%.*lf", abs(_decimalPlace), value);

            _isSign = false;

            // get _integerPlace
            for (i = 0, _integerPlace = -1, _length = 0; i < num; i++)
            {
                if (buf[i] == '.')
                {
                    _isDot = true;
                    break;
                }
                else if (buf[i] == '-')
                    _isSign = true;
                else
                {
                    _integerPlace++;
                    _length++;
                }
            }

            _length += abs(_decimalPlace);

            if (sscanf(buf, "%lf", &_value) != 1)
            {
                cerr << "cui::Calculator::setValue: sscanf failed" << endl;
            }
        }
    }

    updateDisplay();
}

void Calculator::digitMarked(DigitLabel *)
{
}

bool Calculator::fallBelowMin(DigitLabel *)
{
    return false;
}

bool Calculator::passOverMax(DigitLabel *)
{
    return false;
}

void Calculator::close()
{
    setVisible(false);
    for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
        (*_iter)->valueSet(_value);
}

void Calculator::cursorEnter(InputDevice *)
{
}

void Calculator::cursorUpdate(InputDevice *)
{
}

void Calculator::cursorLeave(InputDevice *)
{
}

void Calculator::buttonEvent(InputDevice *dev, int button)
{
    if (isVisible() && (dev->getButtonState(button) == 1) && ((dev->getIsectWidget() == 0) || (dev->getIsectWidget()->getGroupID() != getGroupID())))
        close();
}

void Calculator::joystickEvent(InputDevice *)
{
}

void Calculator::wheelEvent(InputDevice *, int)
{
}

void Calculator::setVisible(bool flag)
{
    Widget::setVisible(flag);

    if (flag)
        for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
            (*_iter)->calculatorOpened();
    else
        for (_iter = _listeners.begin(); _iter != _listeners.end(); _iter++)
            (*_iter)->calculatorClosed();
}
