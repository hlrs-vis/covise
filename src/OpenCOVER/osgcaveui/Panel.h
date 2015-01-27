/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PANEL_H_
#define _PANEL_H_

// OSG:
#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osgText/Text>

// CUI:
#include "CheckBox.h"
#include "Interaction.h"
#include "Dial.h"
#include "Events.h"
#include "Card.h"
#include "TextureWidget.h"

namespace cui
{

class CUIEXPORT PanelCard
{
public:
    cui::Card *_card;
    int _pos[2]; ///< grid position of widget (x, y), y is from top to bottom

    PanelCard(cui::Card *, int, int);
    ~PanelCard();
};

class CUIEXPORT PanelTexture
{
public:
    TextureWidget *_tex;
    float _pos[2]; ///< grid position of widget (x, y), y is from top to bottom

    PanelTexture(TextureWidget *, float, float);
    ~PanelTexture();
};

class CUIEXPORT Panel : public cui::Widget, public cui::Events, public cui::CardListener
{
public:
    enum Movability
    { // free movable and 'watchs' to the viewer
        FREE_MOVABLE,
        FIXED_ORIENTATION, // free movable, but panel is fixed oriented
        SCROLL, // scrolling in both the x- and y-direction
        SCROLL_X,
        SCROLL_Y,
        NON_MOVABLE
    };

    enum Appearance
    { // panel is shown always for Wall Appearance
        WALL_PANEL,
        STATIC, // panel is shown always
        POPUP // panel pops up und down
    };

    Panel(Interaction *, Appearance, Movability = NON_MOVABLE);
    Panel(Interaction *, Appearance, Movability, int widgetsPerRow);
    Panel(Interaction *, osg::Vec3, Appearance, Movability);
    virtual ~Panel();
    void initObject();
    void reset();
    virtual void initGraphics();
    virtual void setVertices();
    osg::Geometry *createGeometry();
    void updateGeometry();
    void setObjectBlendState(osg::Geode *);
    virtual void setVisible(bool);
    osg::Geode *getPanelGeode()
    {
        return _panelGeode;
    }
    Movability getMovability()
    {
        return _moveMode;
    }
    int getRows()
    {
        return _numRows;
    }
    int getCols()
    {
        return _numCols;
    }
    float getScale()
    {
        return _scaleFact;
    }
    void setWidth(float);
    float getWidth()
    {
        return _width;
    }
    float getHeight()
    {
        return _height;
    }
    void setHeight(float);
    void setSpacing(float);
    void setDimension(int, int);
    void setBGColor();
    osg::Vec4 getBGColor()
    {
        return (*_BGcolor)[0];
    }
    void setBGTexture(unsigned char *, int, int);
    void setMoveMode(Movability);
    void setPopupMode(int);
    void addCard(cui::Card *);
    void addCard(cui::Card *, int, int);
    void addTexture(cui::TextureWidget *, float, float);
    void removeWidget(cui::Widget *);
    void move(osg::Matrix &, osg::Matrix &);
    bool moveThresholdTest(osg::Matrix &, osg::Matrix &);
    void scroll(int, float);
    virtual void cursorEnter(cui::InputDevice *);
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void cursorLeave(cui::InputDevice *);
    virtual void buttonEvent(cui::InputDevice *, int);
    virtual void joystickEvent(cui::InputDevice *);
    virtual void wheelEvent(cui::InputDevice *, int);
    bool cardButtonEvent(cui::Card *c, int, int);
    bool cardCursorUpdate(cui::Card *, cui::InputDevice *);
    void setRotation(double, osg::Vec3);
    void setPosition(osg::Vec3);
    virtual void layout();

protected:
    Panel();

    std::list<PanelCard *> _cards;
    std::list<PanelTexture *> _textures;
    Interaction *_interaction;
    osg::MatrixTransform *_objNode;
    osg::Geode *_panelGeode;
    osg::Geometry *_geom;
    osg::Vec3Array *_panelVertices;
    osg::Vec3Array *_normals;
    osg::Vec4Array *_BGcolor;
    unsigned char *_img;
    osg::Matrix _initWand2w;
    osg::Vec3 _topLeft;
    bool _setPos;
    int _numRows;
    int _numCols;
    float _height;
    float _width;
    float _borderSizeX;
    float _borderSizeY;
    float _spacingX;
    float _spacingY;
    float _scrollAmount;
    float _angle;
    float _scaleFact;
    int _maxWidgets;
    Movability _moveMode;
    Appearance _appearance;
    bool _scrollAllDir;
    int _buttonPressed;
    bool _isDown;
    bool _justEntered;
    bool _cardPressed;
    bool _moveThresholdReached;
    osg::Matrix _buttonPressedI2W;
    osg::Matrix _lastWand2w; ///< wand matrix from previous run
    osg::Vec3 _initPoint;
    osg::Vec3 _lastPoint;
    osg::Vec3 _location;
    osg::Vec3 _initWand;
    osg::Vec3 _newWand;
    osg::Image *_panelImage;
    int _widgetsPerRow; ///< max. widgets per row in auto-layout mode
};
}
#endif
