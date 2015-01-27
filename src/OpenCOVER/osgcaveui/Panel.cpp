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
#include "Panel.h"
#include "CUI.h"

using namespace osg;
using namespace cui;

Panel::Panel(Interaction *interaction, Appearance appMode, Movability moveMode)
    : Widget()
    , Events()
    , CardListener()
{
    _interaction = interaction;
    _setPos = false;
    _topLeft.set(0, 0, 0);
    _moveMode = moveMode;
    _appearance = appMode;
    _panelGeode = 0;
    _widgetsPerRow = 9;

    initObject();
}

Panel::Panel(Interaction *interaction, Appearance appMode, Movability moveMode, int widgetsPerRow)
    : Widget()
    , Events()
    , CardListener()
{
    _interaction = interaction;
    _setPos = false;
    _topLeft.set(0, 0, 0);
    _moveMode = moveMode;
    _appearance = appMode;
    _panelGeode = 0;
    _widgetsPerRow = widgetsPerRow;

    initObject();
}

Panel::Panel(Interaction *interaction, Vec3 topLeft, Appearance appMode, Movability moveMode)
    : Widget()
    , Events()
    , CardListener()
{
    _interaction = interaction;
    _topLeft = topLeft;
    _setPos = true;
    _moveMode = moveMode;
    _appearance = appMode;
    _panelGeode = 0;
    _widgetsPerRow = 9;

    initObject();
}

Panel::Panel()
{
}

void Panel::initObject()
{
    reset();
    _justEntered = false;
    //_widgetsPerRow = 9;
    _scrollAmount = 0;
    _isDown = false;
    _cardPressed = false;
    _objNode = new MatrixTransform();

    initGraphics();
    _panelGeode->setNodeMask(1);
    _node->addChild(_panelGeode);
    _scrollAllDir = true;
    _moveThresholdReached = false;
    _interaction->addListener(this, this);
}

Panel::~Panel()
{
}

void Panel::reset()
{
    _numRows = 0;
    _numCols = 0;
    _height = 0.0;
    _width = 0.0;

    list<PanelCard *>::const_iterator iterCard;
    for (iterCard = _cards.begin(); iterCard != _cards.end(); iterCard++)
    {
        removeWidget((*iterCard)->_card);
    }
    list<PanelTexture *>::const_iterator iterTex;
    for (iterTex = _textures.begin(); iterTex != _textures.end(); iterTex++)
    {
        removeWidget((*iterTex)->_tex);
    }

    _cards.clear();
    _textures.clear();
}

void Panel::initGraphics()
{
    // Wall behind widgets:
    _panelGeode = new Geode();

    _height = 1.0f;
    _width = 1.0f;
    _scaleFact = 100.0f;

    _panelGeode->addDrawable(createGeometry());

    switch (_moveMode)
    {
    case NON_MOVABLE:
        _borderSizeX = 0.35;
        _borderSizeY = 0.35;
        _spacingX = 0.1;
        _spacingY = 0.5;
        break;
    default:
        _borderSizeX = 0.35;
        _borderSizeY = 0.35;
        _spacingX = 0.3;
        _spacingY = 0.3;
    }

    switch (_appearance)
    {
    case WALL_PANEL:
    {
        Vec3 pos(0.0f, 6.0f, 0.0f);
        setPosition(pos);

        _scaleFact *= 0.7f;
        break;
    }
    case STATIC:
    case POPUP:
    {
        Vec3 rotAxis(1.0f, 0.0f, 0.0f);
        setRotation(M_PI, rotAxis);

        switch (CUI::_display)
        {
        case CUI::DESKTOP:
        case CUI::CAVE:
        {
            Vec3 pos(-1.0f, -1.0f, 2.0f);
            setPosition(pos);
            _scaleFact *= 0.55f;
            break;
        }
        case CUI::FISHTANK:
        {
            Vec3 pos(0.0f, 0.0f, 0.0f);
            setPosition(pos);
            _scaleFact *= 0.1f;
        }
        }
    }
    }

    Matrix m;
    Matrix m2 = _node->getMatrix();
    m.makeScale(Vec3(_scaleFact, _scaleFact, _scaleFact));
    m.postMult(m2);
    _node->setMatrix(m);
}

Geometry *Panel::createGeometry()
{
    _geom = new Geometry();
    _panelVertices = new Vec3Array(4);

    setVertices();
    _geom->setVertexArray(_panelVertices);

    // Create normals:
    _normals = new Vec3Array(1);
    (*_normals)[0].set(0.0f, 0.0f, 1.0f);
    _geom->setNormalArray(_normals);
    _geom->setNormalBinding(Geometry::BIND_OVERALL);

    // Create colors:
    _BGcolor = new Vec4Array(1);
    setBGColor();
    _geom->setColorArray(_BGcolor);
    _geom->setColorBinding(Geometry::BIND_OVERALL);

    // Create quad:
    _geom->addPrimitiveSet(new DrawArrays(GL_QUADS, 0, 4));
    _geom->setUseDisplayList(false);

    StateSet *stateSet = _geom->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);

    return _geom;
}

void Panel::setVertices()
{
    Vec3 tl, bl, br, tr; // b=bottom, t=top, l=left, r=right

    if (!_setPos)
    {
        _topLeft.set(0.0f, 0.0f, 0.0f);
        _setPos = true;
    }

    tl = _topLeft;
    bl = tl;
    bl[1] = tl[1] - _height;
    br = bl;
    br[0] = bl[0] + _width;
    tr = br;
    tr[1] = tl[1];

    // Create vertices:
    (*_panelVertices)[0] = tl;
    (*_panelVertices)[1] = bl;
    (*_panelVertices)[2] = br;
    (*_panelVertices)[3] = tr;
}

void Panel::updateGeometry()
{
    setVertices();
    //  _geom->setVertexArray(_panelVertices);
    _geom->dirtyBound();
}

void Panel::setObjectBlendState(Geode *geodeCurrent)
{
    // retrieve or create a StateSet
    StateSet *stateBlend = geodeCurrent->getOrCreateStateSet();

    // create a new blend function using GL_SRC_ALPHA and GL_ONE
    BlendFunc *bf = new BlendFunc(GL_SRC_ALPHA, GL_ONE);

    // turn depth testing off
    // stateBlend->setMode(GL_DEPTH_TEST,StateAttribute::OFF);

    // turn standard OpenGL lighting on
    // stateBlend->setMode(GL_LIGHTING,StateAttribute::ON);

    // turn blending on
    stateBlend->setMode(GL_BLEND, StateAttribute::ON);

    // add rendering hint
    stateBlend->setRenderingHint(StateSet::TRANSPARENT_BIN);
    stateBlend->setNestRenderBins(false);

    // add the blend function to the StateSet
    stateBlend->setAttribute(bf);

    // set the StateSet of the Geode to the one that was just created
    geodeCurrent->setStateSet(stateBlend);
}

void Panel::setBGColor()
{
    Vec4 col(0.85f, 0.85f, 0.85f, 1.0f);
    if (_appearance == WALL_PANEL)
        col[3] = 0.0f; // set opacity to 0
    else
        col[3] = 0.7f;
    (*_BGcolor)[0] = col;
}

void Panel::setBGTexture(unsigned char *img, int width, int height)
{
    //_texWidth = width;
    //_texHeight = height;

    Texture2D *texture;

    // Initialize panel texture:
    texture = new Texture2D();
    _panelImage = new Image();
    _panelImage->setImage(width, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, img, Image::USE_NEW_DELETE);
    texture->setImage(_panelImage);

    Vec2Array *texCoords = new Vec2Array(4);
    (*texCoords)[0].set(0.0, 0.0);
    (*texCoords)[1].set(1.0, 0.0);
    (*texCoords)[2].set(1.0, 1.0);
    (*texCoords)[3].set(0.0, 1.0);
    _geom->setTexCoordArray(0, texCoords);

    // Texture:
    StateSet *stateset = _geom->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    stateset->setTextureAttributeAndModes(0, texture, StateAttribute::ON);
    stateset->setNestRenderBins(false);

    _panelImage->dirty();
}

void Panel::setVisible(bool canSee)
{
    Widget::setVisible(canSee);
    /*
  if (_appearance==POPUP)
  {
  Matrix rot, scale, trans, shift, mat;
  //int offset = _numWidgets;

  Matrix head2w = _osgObj->getHeadRoot()->getMatrix();
  const double* headMat = head2w.ptr();
  Matrix osgHeadMat(headMat);

  Vec3 headDir(headMat[8], headMat[9], headMat[10]);
  headDir.normalize();
  Vec3 headLoc = head2w.getTrans();

  Vec3 tmp(headDir[0],0,headDir[2]);
  tmp.normalize();
  trans.setTrans(Vec3(headLoc[0],0,headLoc[2]) + tmp*4);

  rot.makeRotate(M_PI, 1.0f, 0.0f, 0.0f);

  if(_numWidgets >= _maxWidgets)
  offset = _maxWidgets;
  //	cerr << "numwidgets: " << offset << endl;
  shift.makeTranslate(Vec3(-offset*(_widgetWidth+_spacing)/2.0,0,0));

  float angle = vectorAngle(Vec3(0,0,1), headDir, 1);
  Matrix matRot;
  matRot.makeRotate(angle,Vec3(0,1,0));

  mat = shift * matRot * scale * rot * trans;

  _node->setMatrix(mat);
  }*/
}

/** Adds a card to the next empty space (auto-layout mode).
 */
void Panel::addCard(Card *card)
{
    int row, col;
    int numCards = _cards.size();
    col = numCards % _widgetsPerRow;
    row = numCards / _widgetsPerRow;
    addCard(card, col, row);
}

/** Place a card at given location in panel.
  @param col column 0 is at left [0..numColumns-1]
  @param row row 0 is at top [0..numRows-1]
*/
void Panel::addCard(Card *card, int col, int row)
{
    card->addCardListener(this);
    _node->addChild(card->getNode());
    _cards.push_back(new PanelCard(card, col, row));

    // Update panel size:
    layout();
}

void Panel::addTexture(TextureWidget *tex, float col, float row)
{
    _node->addChild(tex->getNode());
    _textures.push_back(new PanelTexture(tex, col, row));

    // update layout:
    layout();
}

/** cards will only be removed. no adjustment of panel height and width
 */
void Panel::removeWidget(Widget *elem)
{
    //  cerr << "remove Card: " << card->getText() << endl;
    _node->removeChild(elem->getNode());

    //  cerr << "deleted: " << card->getNode() << endl;
    _interaction->widgetDeleted(elem->getNode());
    //   delete *iter;		// FIXME: should be deleted
    //   *iter = NULL;
}

void Panel::setMoveMode(Movability moveMode)
{
    _moveMode = moveMode;
}

void Panel::move(Matrix &lastWand2w, Matrix &wand2w)
{
    // Compute difference matrix between last and current wand:
    Matrix invLastWand2w = Matrix::inverse(lastWand2w);
    Matrix wDiff = invLastWand2w * wand2w;

    // Volume follows wand movement:
    Matrix box2w = _node->getMatrix();
    _node->setMatrix(box2w * wDiff);
}

/** @return true if matrices are different enough that one can assume
  the wand was moved on purpose.
*/
bool Panel::moveThresholdTest(Matrix &m1, Matrix &m2)
{
    // Compare translational part:
    Vec3 m1trans = m1.getTrans();
    Vec3 m2trans = m2.getTrans();
    Vec3 diff = m1trans - m2trans;
    float len = diff.length();
    if (len > 0.015f) // this is an empirical value!
    {
        return true;
    }
    else
    {
        // Compare rotational part:
        return false;
    }
}

void Panel::scroll(int axis, float amount)
{
    Matrix diffMat;
    Matrix nodeMat;
    Vec3 scrolling(0, 0, 0);

    scrolling[axis] = amount;
    diffMat.makeTranslate(scrolling);
    nodeMat = _node->getMatrix();
    nodeMat = nodeMat * diffMat;
    _node->setMatrix(nodeMat);
}

void Panel::cursorEnter(InputDevice *)
{
    //cerr << "Panel::cursorEnter" << endl;
    _justEntered = true;
}

void Panel::cursorLeave(InputDevice *)
{
    //cerr << "Panel::cursorLeave" << endl;
}

void Panel::cursorUpdate(InputDevice *evt)
{
    Vec3 newPoint = evt->getIsectPoint();

    if (!_justEntered && _isDown && _buttonPressed == 0)
    {
        switch (_moveMode)
        {
        case FREE_MOVABLE:
        {
            if (!_moveThresholdReached)
            {
                Matrix i2w = evt->getI2W();
                _lastWand2w = i2w;
                _moveThresholdReached = moveThresholdTest(_buttonPressedI2W, i2w);
            }
            if (_moveThresholdReached)
            {
                Matrix i2w = evt->getI2W();
                move(_lastWand2w, i2w);
                _lastWand2w = i2w;
            }
            break;
        }
        case FIXED_ORIENTATION:
        {
            Vec3 diffVec = newPoint - _lastPoint;
            float buf = diffVec[1];
            diffVec[1] = -diffVec[2];
            diffVec[2] = buf;
            Matrix diffMat;
            diffMat.makeTranslate(diffVec);
            Matrix nodeMat = _node->getMatrix();
            nodeMat = nodeMat * diffMat;
            _node->setMatrix(nodeMat);
            break;
        }
        case SCROLL:
        {
            float dx = newPoint[0] - _lastPoint[0];
            float dy = newPoint[1] - _lastPoint[1];
            if (absolute(dx) > absolute(dy))
                _moveMode = SCROLL_X;
            else if (absolute(dx) < absolute(dy))
                _moveMode = SCROLL_Y;
            break;
        }
        case SCROLL_X:
        {
            int scrollAxis = 0;
            scroll(scrollAxis, newPoint[scrollAxis] - _lastPoint[scrollAxis]);
            break;
        }
        case SCROLL_Y:
        {
            int scrollAxis = 1;
            scroll(scrollAxis, newPoint[scrollAxis] - _lastPoint[scrollAxis]);
            break;
        }
        default:
            break;
        }
    }
    _lastPoint = newPoint;
    _justEntered = false;
}

void Panel::buttonEvent(InputDevice *evt, int b)
{
    _buttonPressed = b;
    _isDown = evt->getButtonState(b) != 0;
    if (_buttonPressed == 0)
    {
        if (_isDown == true)
        {
            if (_moveMode == SCROLL_X || _moveMode == SCROLL_Y)
                _scrollAllDir = false;
            else if (_moveMode == FREE_MOVABLE)
            {
                _buttonPressedI2W = evt->getPressedI2W(0);
                _moveThresholdReached = false;
            }
        }
        else
        {
            if (_scrollAllDir && (_moveMode == SCROLL_X || _moveMode == SCROLL_Y))
                _moveMode = SCROLL;
        }
    }
}

void Panel::joystickEvent(InputDevice *)
{
}

void Panel::wheelEvent(InputDevice *, int)
{
}

void Panel::setRotation(double deg, Vec3 axis)
{
    Matrix mat1;
    Matrix mat2;
    mat1 = _node->getMatrix();
    mat2.makeRotate(deg, axis);
    mat1.postMult(mat2);
    _node->setMatrix(mat1);
}

void Panel::setPosition(Vec3 pos)
{
    Matrix mat;
    mat = _node->getMatrix();
    mat.setTrans(pos);
    _node->setMatrix(mat);
}

bool Panel::cardButtonEvent(Card *, int button, int state)
{
    _buttonPressed = button;
    _isDown = state != 0;

    if (_moveMode == FREE_MOVABLE)
    {
        if (state == 1)
        {
            _cardPressed = true;
            _moveThresholdReached = false;
            _buttonPressedI2W = _interaction->_wandR->getPressedI2W(0);
            _lastWand2w = _buttonPressedI2W;
            return true;
        }
        else
        {
            _cardPressed = false;
            if (_moveThresholdReached)
            {
                return true;
            }
            else
            {
                if (_appearance == POPUP)
                {
                    setVisible(false);
                }
                return false;
            }
        }
    }
    else
    {
        if (_appearance == POPUP)
        {
            if (state == 0)
            {
                setVisible(false);
            }
        }
        return false;
    }
}

bool Panel::cardCursorUpdate(Card *c, InputDevice *dev)
{
    if (_cardPressed)
    {
        Matrix i2w = dev->getI2W();
        _moveThresholdReached = moveThresholdTest(_buttonPressedI2W, i2w);

        // dials shouldn't move panel, or else panel rotates while turning dial
        if (_moveThresholdReached && dynamic_cast<Button *>(c))
        {
            move(_lastWand2w, i2w);
            _lastWand2w = i2w;
        }
    }
    return true;
}

/** Layout cards on panel as a grid.
 */
void Panel::layout()
{
    Matrix trans, scale;
    float offset[2];
    float gridWidth, gridHeight;
    float sizeX, sizeY;

    if (_cards.empty() && _textures.empty())
    {
        _numCols = _numRows = 0;
        _width = 2.0f * _borderSizeX;
        _height = 2.0f * _borderSizeY;
        updateGeometry();
    }
    else
    {
        gridWidth = Card::DEFAULT_CARD_WIDTH;
        gridHeight = Card::DEFAULT_CARD_HEIGHT;

        float dx = gridWidth + _spacingX;
        float dy = gridHeight + _spacingY;

        // calculate grid size:
        _numRows = _numCols = 0;
        list<PanelCard *>::const_iterator iterCard;
        for (iterCard = _cards.begin(); iterCard != _cards.end(); iterCard++)
        {
            _numCols = ts_max(_numCols, (*iterCard)->_pos[0] + 1);
            _numRows = ts_max(_numRows, (*iterCard)->_pos[1] + 1);
        }

        sizeX = sizeY = 0.0;
        list<PanelTexture *>::const_iterator iterTex;
        for (iterTex = _textures.begin(); iterTex != _textures.end(); iterTex++)
        {
            if (sizeX < ((*iterTex)->_pos[0] + (*iterTex)->_tex->getWidth()))
                sizeX = (*iterTex)->_pos[0] + (*iterTex)->_tex->getWidth();
            if (sizeY < ((*iterTex)->_pos[1] + (*iterTex)->_tex->getHeight()))
                sizeY = (*iterTex)->_pos[1] + (*iterTex)->_tex->getHeight();
        }

        // Update panel geometry:
        if (((float)(_numCols * dx - _spacingX)) > sizeX)
            _width = _numCols * dx + 2.0f * _borderSizeX - _spacingX;
        else
            _width = sizeX + 2.0f * _borderSizeX;

        if (((float)(_numRows * dy - _spacingY)) > sizeY)
            _height = _numRows * dy + 2.0f * _borderSizeY - _spacingY;
        else
            _height = sizeY + 2.0f * _borderSizeY;

        updateGeometry();

        // Place cards:
        float x, y;
        offset[0] = _topLeft[0] + _borderSizeX + gridWidth / 2.0f;
        offset[1] = _topLeft[1] - _borderSizeY - gridHeight / 2.0f;
        for (iterCard = _cards.begin(); iterCard != _cards.end(); iterCard++)
        {
            x = offset[0] + (*iterCard)->_pos[0] * dx;
            y = offset[1] - (*iterCard)->_pos[1] * dy;
            trans.makeTranslate(x, y, 2.0f * Widget::EPSILON_Z);
            //  scale.makeScale(Vec3(0.75f,0.75f,0.75f));
            (*iterCard)->_card->setMatrix(trans);
        }

        // Place textures:
        for (iterTex = _textures.begin(); iterTex != _textures.end(); iterTex++)
        {
            offset[0] = _topLeft[0] + _borderSizeX + (*iterTex)->_tex->getWidth() / 2.0f;
            offset[1] = _topLeft[1] - _borderSizeY - (*iterTex)->_tex->getHeight() / 2.0f;
            x = offset[0] + (*iterTex)->_pos[0];
            y = offset[1] - (*iterTex)->_pos[1];
            trans.makeTranslate(x, y, 2.0f * Widget::EPSILON_Z);
            (*iterTex)->_tex->setMatrix(trans);
        }
    }
}

void Panel::setWidth(float width)
{
    _width = width;
    updateGeometry();
}

void Panel::setHeight(float height)
{
    _height = height;
    updateGeometry();
}

PanelCard::PanelCard(Card *card, int col, int row)
{
    _card = card;
    _pos[0] = col;
    _pos[1] = row;
}

PanelCard::~PanelCard()
{
    delete _card;
}

PanelTexture::PanelTexture(TextureWidget *tex, float col, float row)
{
    _tex = tex;
    _pos[0] = col;
    _pos[1] = row;
}

PanelTexture::~PanelTexture()
{
    delete _tex;
}
