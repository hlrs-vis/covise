/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_TEXTURE_WIDGET_H_
#define _CUI_TEXTURE_WIDGET_H_

// C++:
#include <string>

// OSG:
#include <osg/Geometry>
#include <osgText/Text>

// Local:
#include "Widget.h"
#include "Events.h"

namespace cui
{
class Interaction;
class TextureListener;
class InputDevice;

/** This is a class for texture widgets.
   */
class CUIEXPORT TextureWidget : public Widget, public Events
{
public:
    static const float DEFAULT_TEXTURE_WIDTH;
    static const float DEFAULT_TEXTURE_HEIGHT;
    static const float DEFAULT_LABEL_HEIGHT;
    static const float DEFAULT_FONT_SIZE;
    TextureWidget(Interaction *, float width = DEFAULT_TEXTURE_WIDTH, float height = DEFAULT_TEXTURE_HEIGHT);
    ~TextureWidget();
    bool loadImage(const std::string &);
    void setImage(int, osg::Image *);
    void showTexture(int);
    void addTextureListener(TextureListener *);
    bool isInside();
    float getWidth();
    float getHeight();
    void setLabelText(int, const std::string &);
    void addGeomToLeft(osg::Drawable *);
    void addGeomToRight(osg::Drawable *);

protected:
    osg::ref_ptr<osg::Switch> _swTexture;
    osg::Geode *_texture[2];
    osg::Geometry *_geom;
    osg::Geode *_leftGeode;
    osg::Geode *_rightGeode;
    osg::Geometry *_texGeom0;
    osg::Geometry *_texGeom1;
    osg::Texture2D *_tex0;
    osg::Texture2D *_tex1;
    osg::Image *_defaultTexImage;
    osg::Vec3Array *_vertices;
    osgText::Text *_label0;
    osgText::Text *_label1;
    float _width;
    float _height;
    Interaction *_interaction;
    std::list<TextureListener *> _listeners;
    bool _cursorInside;

    void createBackground();
    void createTexturesGeometry();
    void initLabels();
    void setLabelPos(osg::Vec3);
    void initTextures();
    void createDefaultTexImage();
    void initVertices();
    void setVertices();
    void cursorEnter(InputDevice *);
    void cursorUpdate(InputDevice *);
    void cursorLeave(InputDevice *);
    void buttonEvent(InputDevice *, int);
    void joystickEvent(InputDevice *);
    void wheelEvent(InputDevice *, int);
};

class CUIEXPORT TextureListener
{
public:
    virtual ~TextureListener()
    {
    }
    virtual bool texButtonEvent(TextureWidget *, int, int) = 0;
    virtual bool texCursorUpdate(TextureWidget *, InputDevice *) = 0;
};
}
#endif
