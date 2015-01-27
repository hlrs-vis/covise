/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COLOR_BOX_H_
#define _COLOR_BOX_H_

// C++:
#include <string>

// OSG:
#include <osg/Geometry>
#include <osgText/Text>

// CUI:
#include <Widget.h>
#include <Events.h>
#include <Interaction.h>
#include <InputDevice.h>

// Virvo:
#include <vvtransfunc.h>

// Local:
#include "TFColorBar.h"

class TFColorBox : public cui::Widget, public cui::Events
{
public:
    static const float DEFAULT_TEXTURE_WIDTH;
    static const float DEFAULT_TEXTURE_HEIGHT;
    static const float DEFAULT_LABEL_HEIGHT;
    static const float DEFAULT_FONT_SIZE;
    TFColorBox(cui::Interaction *, float width = DEFAULT_TEXTURE_WIDTH, float height = DEFAULT_TEXTURE_HEIGHT, vvTransFunc * = NULL);
    ~TFColorBox();
    bool loadImage(const std::string &);
    void setImage(int, osg::Image *);
    void showTexture(int);
    bool isInside();
    float getWidth();
    float getHeight();
    void setLabelText(int, const std::string &);
    void addGeomToLeft(osg::Drawable *);
    void addGeomToRight(osg::Drawable *);
    void addBarListener(BarListener *);

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
    cui::Interaction *_interaction;
    bool _cursorInside;
    bool _moveWidget;
    vvTransFunc *_transFunc;
    std::list<BarListener *> _listeners;

    void createBackground();
    void createTexturesGeometry();
    void initLabels();
    void setLabelPos(osg::Vec3);
    void initTextures();
    void createDefaultTexImage();
    void initVertices();
    void setVertices();
    void cursorEnter(cui::InputDevice *);
    void cursorUpdate(cui::InputDevice *);
    void cursorLeave(cui::InputDevice *);
    void buttonEvent(cui::InputDevice *, int);
    void joystickEvent(cui::InputDevice *);
    void wheelEvent(cui::InputDevice *, int);
    void selectWidget(vvTFWidget *);
};

#endif
