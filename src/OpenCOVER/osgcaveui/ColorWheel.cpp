/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++:
#include <fstream>
#include <iostream>

// OSG:
#include <osg/Math>
#include <osg/Geode>
#include <osg/Vec3>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/BlendFunc>
#include <osg/LineSegment>
#include <osg/LineWidth>
#include <osgDB/ReadFile>
#include <osgUtil/Optimizer>

// Virvo:
#include <virvo/vvtoolshed.h>

// Local:
#include "ColorWheel.h"
#include "CUI.h"

using namespace osg;
using namespace cui;

const float ColorWheel::BRIGHTNESS = 1.0f;

ColorWheel::ColorWheel(Interaction *interaction, float height, float width)
    : TextureWidget(interaction, height, width)
{
    _texture = new uchar[WIDTH * HEIGHT * 4];
    vvToolshed::makeColorBoardTexture(WIDTH, HEIGHT, BRIGHTNESS, _texture);

    _image = new Image();
    _image->setImage(WIDTH, HEIGHT, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, _texture, Image::USE_NEW_DELETE);
    this->setImage(0, _image);
    this->showTexture(0);

    _selectColor = false;
}
ColorWheel::~ColorWheel()
{
}
Vec4 ColorWheel::getSelectedColor()
{
    return _selected;
}
void ColorWheel::addColorWheelListener(ColorWheelListener *listener)
{
    _colorListeners.push_back(listener);
}
Vec4 ColorWheel::colorAtPoint(Vec3)
{
    return Vec4(0, 0, 0, 0);
}
void ColorWheel::cursorEnter(cui::InputDevice *dev)
{
    TextureWidget::cursorEnter(dev);
}

void ColorWheel::cursorUpdate(cui::InputDevice *dev)
{
    TextureWidget::cursorUpdate(dev);

    if (_selectColor)
    {
        Matrix o2w = CUI::computeLocal2Root(getNode());

        Matrix w2o = Matrix::inverse(o2w);

        // Get the location of the intersection
        Vec3 wPoint = dev->getIsectPoint();
        Vec3 oPoint = wPoint * w2o;

        // Need to convert this to the color
        float dx = getWidth() / _image->s();
        float dy = getHeight() / _image->t();

        //Get color at the pixel
        int x = (int)((oPoint[0] + getWidth() / 2) / dx);
        int y = _image->t() - (int)((-oPoint[1] + getHeight() / 2) / dy);

        Vec4 col(_texture[4 * (y * _image->s() + x)], _texture[4 * (y * _image->s() + x) + 1],
                 _texture[4 * (y * _image->s() + x) + 2], _texture[4 * (y * _image->s() + x) + 3]);

        // Set current selected to the new color
        _selected = col;

        std::list<ColorWheelListener *>::iterator iter;

        for (iter = _colorListeners.begin(); iter != _colorListeners.end(); ++iter)
        {
            (*iter)->colorWheelCursorUpdate(this, dev);
        }
    }
}

void ColorWheel::cursorLeave(cui::InputDevice *dev)
{
    TextureWidget::cursorLeave(dev);
}

void ColorWheel::buttonEvent(cui::InputDevice *dev, int button)
{
    TextureWidget::buttonEvent(dev, button);
    if (button == 0)
    {
        if (dev->getButtonState(button) == 1)
        {
            _selectColor = true;
        }
        else if (dev->getButtonState(button) == 0)
        {
            _selectColor = false;
        }
    }

    // PHILIP
    //else if (button==1)
    //{
    //  if (dev->getButtonState(button))
    //  {
    std::list<ColorWheelListener *>::iterator iter;

    for (iter = _colorListeners.begin(); iter != _colorListeners.end(); ++iter)
    {
        (*iter)->colorWheelButtonUpdate(this, button, dev->getButtonState(button));
    }
    //  }
    //}
}
