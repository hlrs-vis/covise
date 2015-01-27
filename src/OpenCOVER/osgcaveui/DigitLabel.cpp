/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DigitLabel.h"
#include "Interaction.h"

using namespace osg;
using namespace cui;

const float DigitLabel::DEFAULT_LABEL_WIDTH = 1.0;
const float DigitLabel::DEFAULT_LABEL_HEIGHT = 1.0;
const float DigitLabel::DEFAULT_DIGIT_HEIGHT = 0.8;
const float DigitLabel::STEP_ANGLE = 8.0;
const float DigitLabel::MAX_ANGLE = (10 * STEP_ANGLE) - 1;

DigitLabel::DigitLabel(Interaction *interaction, float scale)
    : Widget()
    , Events()
{
    _interaction = interaction;
    _scale = scale;

    _geode = new Geode();

    _geode->addDrawable(createGeometry());
    _geode->addDrawable(createFrame());

    createDigit();
    _geode->addDrawable(_digitText);

    _node->addChild(_geode);

    _interactionOn = true;
    _isHighlighted = false;
    _isActive = false;

    _interaction->addListener(this, this);
}

DigitLabel::~DigitLabel()
{
    _interaction->removeListener(this);
    _listener.clear();
}

Geometry *DigitLabel::createGeometry()
{
    float width2, height2;

    Geometry *geom = new Geometry();

    width2 = DEFAULT_LABEL_WIDTH / 2.0 * _scale;
    height2 = DEFAULT_LABEL_HEIGHT / 2.0 * _scale;

    Vec3Array *vertices = new Vec3Array(4);
    (*vertices)[0].set(-width2, -height2, 0);
    (*vertices)[1].set(width2, -height2, 0);
    (*vertices)[2].set(width2, height2, 0);
    (*vertices)[3].set(-width2, height2, 0);
    geom->setVertexArray(vertices);

    Vec3Array *normal = new Vec3Array(1);
    (*normal)[0].set(0.0, 0.0, 1.0);
    geom->setNormalArray(normal);
    geom->setNormalBinding(Geometry::BIND_OVERALL);

    _geomColor = new Vec4Array(1);
    (*_geomColor)[0].set(1.0, 1.0, 1.0, 1.0);
    geom->setColorArray(_geomColor);
    geom->setColorBinding(Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new DrawArrays(GL_QUADS, 0, 4));
    geom->setUseDisplayList(false);

    StateSet *stateSet = geom->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);

    return geom;
}

Geometry *DigitLabel::createFrame()
{
    float width2, height2;

    Geometry *geom = new Geometry();

    width2 = DEFAULT_LABEL_WIDTH / 2.0 * _scale;
    height2 = DEFAULT_LABEL_HEIGHT / 2.0 * _scale;

    Vec3Array *vertices = new Vec3Array(5);
    (*vertices)[0].set(-width2, -height2, EPSILON_Z);
    (*vertices)[1].set(width2, -height2, EPSILON_Z);
    (*vertices)[2].set(width2, height2, EPSILON_Z);
    (*vertices)[3].set(-width2, height2, EPSILON_Z);
    (*vertices)[4].set(-width2, -height2, EPSILON_Z);
    geom->setVertexArray(vertices);

    Vec3Array *normal = new Vec3Array(1);
    (*normal)[0].set(0.0, 0.0, 1.0);
    geom->setNormalArray(normal);
    geom->setNormalBinding(Geometry::BIND_OVERALL);

    _frameColor = new Vec4Array(1);
    (*_frameColor)[0].set(0.0, 0.0, 0.0, 1.0);
    geom->setColorArray(_frameColor);
    geom->setColorBinding(Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINE_STRIP, 0, 5));
    geom->setUseDisplayList(false);

    StateSet *stateSet = geom->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);

    return geom;
}

void DigitLabel::createDigit()
{
    _digit = 0;
    _value = 0.0;

    _digitText = new osgText::Text();
    _digitText->setDataVariance(Object::DYNAMIC);

    _digitText->setFont(osgText::readFontFile("arial.ttf"));
    _digitText->setColor(COL_BLACK);
    _digitText->setFontResolution(20, 20);
    _digitText->setPosition(Vec3(0.0, 0.0, EPSILON_Z));
    _digitText->setCharacterSize(DEFAULT_DIGIT_HEIGHT * _scale);
    _digitText->setMaximumWidth(DEFAULT_LABEL_WIDTH * _scale);
    _digitText->setMaximumHeight(DEFAULT_LABEL_HEIGHT * _scale);
    _digitText->setAlignment(osgText::Text::CENTER_CENTER);
    _digitText->setUseDisplayList(false);

    setDigit();

    StateSet *stateSet = _digitText->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);
}

void DigitLabel::highlight(bool flag)
{
    if (_isHighlighted == flag)
        return;

    if (_isHighlighted)
    {
        _isHighlighted = false;
        if (_isActive)
            (*_geomColor)[0].set(0.0, 0.0, 0.0, 1.0);
        else
            (*_geomColor)[0].set(1.0, 1.0, 1.0, 1.0);
    }
    else
    {
        _isHighlighted = true;
        (*_geomColor)[0].set(0.7, 0.7, 0.7, 1.0);
    }
}

void DigitLabel::enableInteraction(bool flag)
{
    _interactionOn = flag;
}

void DigitLabel::setDigitText(const std::string &text)
{
    _digitText->setText(text);
}

void DigitLabel::setDigit()
{
    char text[4];
    std::list<DigitListener *>::iterator iter;

    sprintf(text, "%d", _digit);
    setDigitText(text);

    for (iter = _listener.begin(); iter != _listener.end(); iter++)
    {
        (*iter)->digitValueUpdate(this);
    }
}

void DigitLabel::setDigit(int digit)
{
    char text[4];

    if (digit > 9)
        _digit = 9;
    else if (digit < 0)
        _digit = 0;
    else
        _digit = digit;

    _value = _digit * STEP_ANGLE + STEP_ANGLE / 2.0;

    sprintf(text, "%d", _digit);
    setDigitText(text);
}

void DigitLabel::changeValue(Matrix &lastWand2W, Matrix &newWand2w)
{
    float diffAngle, tmp;
    bool flag = false;
    std::list<DigitListener *>::iterator iter;

    diffAngle = angleDiff(lastWand2W, newWand2w, Widget::Z);

    if (!_interactionOn)
        diffAngle *= STEP_ANGLE;

    tmp = _value + diffAngle;

    if (tmp > MAX_ANGLE)
    {
        for (iter = _listener.begin(); iter != _listener.end(); iter++)
        {
            if ((*iter)->passOverMax(this))
                flag = true;
        }

        if (flag)
            _value = tmp - MAX_ANGLE;
        else
            _value = MAX_ANGLE;
    }
    else if (tmp < 0.0f)
    {
        for (iter = _listener.begin(); iter != _listener.end(); iter++)
        {
            if ((*iter)->fallBelowMin(this))
                flag = true;
        }

        if (flag)
            _value = tmp + MAX_ANGLE;
        else
            _value = 0.0f;
    }
    else
        _value = tmp;

    if (_interactionOn)
    {
        _digit = (int)floor(_value / STEP_ANGLE);
        setDigit();
    }

    //   _digit = (int) floor(_value / STEP_ANGLE);
    //   setDigit();
}

bool DigitLabel::decreaseValue()
{
    bool flag = false;
    std::list<DigitListener *>::iterator iter;

    if (_digit > 0)
    {
        _digit = _digit - 1;
        _value = _value - STEP_ANGLE;
    }
    else
    {
        for (iter = _listener.begin(); iter != _listener.end(); iter++)
        {
            if ((*iter)->fallBelowMin(this))
                flag = true;
        }

        if (flag)
        {
            _digit = 9;
            _value = MAX_ANGLE;
        }
        else
            return false;
    }

    setDigit();
    return true;
}

bool DigitLabel::increaseValue()
{
    bool flag = false;
    std::list<DigitListener *>::iterator iter;

    if (_digit < 9)
    {
        _digit = _digit + 1;
        _value = _value + STEP_ANGLE;
    }
    else
    {
        for (iter = _listener.begin(); iter != _listener.end(); iter++)
        {
            if ((*iter)->passOverMax(this))
                flag = true;
        }

        if (flag)
        {
            _digit = 0;
            _value = 0.0f;
        }
        else
            return false;
    }

    setDigit();
    return true;
}

void DigitLabel::cursorEnter(InputDevice *)
{
    if (_logFile)
    {
        sprintf(_logBuf, "cursor entered digit label with digit:\t%s", _digitText->getText().createUTF8EncodedString().c_str());
        _logFile->addLog(_logBuf);
    }

    _isActive = true;
    (*_geomColor)[0].set(0.0, 0.0, 0.0, 1.0);
    (*_frameColor)[0].set(1.0, 1.0, 1.0, 1.0);
    _digitText->setColor(COL_WHITE);
}

void DigitLabel::cursorUpdate(InputDevice *input)
{
    Matrix last, curr;

    last = input->getLastI2W();
    curr = input->getI2W();

    if (input->getButtonState(0) == 1)
        changeValue(last, curr);

    //   if ((input->getButtonState(0) == 1) && _interactionOn)
    //     {
    //       changeValue(last, curr);
    //     }
}

void DigitLabel::cursorLeave(InputDevice *)
{
    if (_logFile)
    {
        sprintf(_logBuf, "cursor left digit label with digit:\t%s", _digitText->getText().createUTF8EncodedString().c_str());
        _logFile->addLog(_logBuf);
    }

    _isActive = false;
    if (_isHighlighted)
        (*_geomColor)[0].set(0.7, 0.7, 0.7, 1.0);
    else
        (*_geomColor)[0].set(1.0, 1.0, 1.0, 1.0);
    (*_frameColor)[0].set(0.0, 0.0, 0.0, 1.0);
    _digitText->setColor(COL_BLACK);
}

void DigitLabel::buttonEvent(InputDevice *input, int)
{
    std::list<DigitListener *>::iterator iter;

    if (input->buttonJustPressed(0))
    {
        for (iter = _listener.begin(); iter != _listener.end(); iter++)
        {
            (*iter)->digitLabelUpdate(this);
        }
    }

    if (input->buttonJustPressed(2))
    {
        for (iter = _listener.begin(); iter != _listener.end(); iter++)
        {
            (*iter)->digitMarked(this);
        }
    }
}

void DigitLabel::joystickEvent(InputDevice *)
{
}

void DigitLabel::wheelEvent(InputDevice *, int)
{
}

void DigitLabel::addDigitListener(DigitListener *listener)
{
    _listener.push_back(listener);
}
