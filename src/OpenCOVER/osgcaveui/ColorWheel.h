/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COLOR_WHEEL_H_
#define _COLOR_WHEEL_H_

// OSG:
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osgText/Text>

// CUI:
#include "Widget.h"
#include "CheckBox.h"
#include "Interaction.h"
#include "Dial.h"
#include "Events.h"
#include "Card.h"
#include "TextureWidget.h"
#include "Panel.h"

namespace cui
{
class CUIEXPORT ColorWheelListener;

class CUIEXPORT ColorWheel : public cui::TextureWidget
{

public:
    ColorWheel(Interaction *, float, float);
    virtual ~ColorWheel();
    static const int HEIGHT = 2048;
    static const int WIDTH = 2048;
    static const float BRIGHTNESS;
    virtual void addColorWheelListener(ColorWheelListener *);
    virtual void cursorEnter(cui::InputDevice *);
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void cursorLeave(cui::InputDevice *);
    virtual void buttonEvent(cui::InputDevice *, int);
    osg::Vec4 colorAtPoint(osg::Vec3);
    list<ColorWheelListener *> _colorListeners;
    osg::Vec4 getSelectedColor();

protected:
    unsigned char *_texture;
    osg::Image *_image;
    osgText::Text *_satText;
    osgText::Text *_hueText;
    osg::Vec4 _selected;

    cui::Button *_okButton;
    cui::TextureWidget *_textureWidget;
    bool _selectColor;
};

class CUIEXPORT ColorWheelListener
{
public:
    virtual ~ColorWheelListener()
    {
    }
    virtual bool colorWheelCursorUpdate(ColorWheel *, InputDevice *) = 0;
    virtual bool colorWheelButtonUpdate(ColorWheel *, int, int) = 0;
};
}
#endif
