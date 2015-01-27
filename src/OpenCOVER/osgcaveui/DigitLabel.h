/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_DIGITLABEL_WIDGET_H_
#define _CUI_DIGITLABEL_WIDGET_H_

// OSG:
#include <osg/Geometry>
#include <osg/Matrix>
#include <osgText/Text>

// Local:
#include "Widget.h"
#include "Events.h"

namespace cui
{
class Interaction;
class DigitListener;

/**
     This is a class for a label widget which shows
     a single digit.
  */
class CUIEXPORT DigitLabel : public Widget, public Events
{

public:
    static const float STEP_ANGLE;
    static const float MAX_ANGLE;
    static const float DEFAULT_LABEL_WIDTH;
    static const float DEFAULT_LABEL_HEIGHT;
    static const float DEFAULT_DIGIT_HEIGHT;

    DigitLabel(Interaction *, float scale = 1.0);
    ~DigitLabel();
    void addDigitListener(DigitListener *);
    float getWidth()
    {
        return DEFAULT_LABEL_WIDTH * _scale;
    }
    float getHeight()
    {
        return DEFAULT_LABEL_HEIGHT * _scale;
    }
    void setDigitText(const std::string &);
    void setDigit(int);
    void enableInteraction(bool);
    bool decreaseValue();
    bool increaseValue();
    int getDigit()
    {
        return _digit;
    }
    void highlight(bool);

protected:
    void cursorEnter(InputDevice *);
    void cursorUpdate(InputDevice *);
    void cursorLeave(InputDevice *);
    void buttonEvent(InputDevice *, int);
    void joystickEvent(InputDevice *);
    void wheelEvent(InputDevice *, int);

private:
    osg::Geode *_geode;
    osgText::Text *_digitText;
    Interaction *_interaction;
    std::list<DigitListener *> _listener;
    int _digit;
    float _value;
    float _scale;
    bool _interactionOn;
    bool _isActive;
    bool _isHighlighted;
    osg::Vec4Array *_geomColor;
    osg::Vec4Array *_frameColor;

    osg::Geometry *createGeometry();
    osg::Geometry *createFrame();
    void createDigit();
    void changeValue(osg::Matrix &, osg::Matrix &);
    void setDigit();
};

class CUIEXPORT DigitListener
{
public:
    virtual ~DigitListener()
    {
    }
    virtual void digitValueUpdate(DigitLabel *) = 0;
    virtual void digitLabelUpdate(DigitLabel *) = 0;
    virtual void digitMarked(DigitLabel *) = 0;
    virtual bool fallBelowMin(DigitLabel *) = 0;
    virtual bool passOverMax(DigitLabel *) = 0;
};
}
#endif
