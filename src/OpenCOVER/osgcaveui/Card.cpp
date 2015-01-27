/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++:
#include <iostream>
#include <fstream>

// OSG:
#include <osg/Geode>
#include <osg/Switch>
#include <osg/TexEnv>
#include <osg/Depth>
#include <osg/LineWidth>
#include <osgText/Text>
#include <osgDB/ReadFile>

// Local:
#include "Card.h"
#include "Interaction.h"
#include "CUI.h"

#include <time.h>

using namespace osg;
using namespace cui;
using namespace std;

const float Card::DEFAULT_CARD_WIDTH = 1.0;
const float Card::DEFAULT_CARD_HEIGHT = 1.3;
const float Card::DEFAULT_FONT_SIZE = 0.25;
const float Card::TIP_WIDTH = 6;
const float Card::ICON_SIZE = 0.95;
bool Card::_magnification = false;

static bool getFont = true;

static osgText::Font *arialfont;

Card::Card(Interaction *interaction)
    : Widget()
    , Events()
{

    if (getFont)
    {
        arialfont = osgText::readFontFile("arial.ttf");
        getFont = false;
    }

    this->setFont(arialfont);

    _magnifyXF = new MatrixTransform();
    _swHighlight = new Switch();
    _swFocus = new Switch();
    _highlight[1] = _highlight[0] = NULL;
    _focus[1] = _focus[0] = NULL;
    for (int i = 0; i < 2; ++i)
    {
        _labelText[i] = new osgText::Text();
        _labelText[i]->setDataVariance(Object::DYNAMIC);
    }
    _icon = NULL;
    _magnifyOffset[2] = 0.1f;
    createGeometry();

    _interaction = interaction;

    createTip();

    _interaction->addListener(this, this);
}

Card::~Card()
{
}

void Card::setText(const std::string &text)
{
    for (int i = 0; i < 2; ++i)
    {
        _labelText[i]->setText(text);
    }
}

std::string Card::getText()
{
    osgText::String txt = _labelText[0]->getText();
    return txt.createUTF8EncodedString();
}

/** Create card geometry.
 */
void Card::createGeometry()
{
    int i;

    // Initialize icon texture:
    _icon = new Texture2D();
    Image *image = new Image();
    char *img = new char[2 * 2 * 4];
    memset(img, 0, 2 * 2 * 4);
    image->setImage(2, 2, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char *)img, Image::USE_NEW_DELETE);
    _icon->setImage(image);

    // Create geodes:
    for (i = 0; i < 2; ++i)
    {
        _highlight[i] = new Geode();
        _focus[i] = new Geode();
    }

    // Create highlight-dependent geometry:
    _highlight[0]->addDrawable(createBackground(COL_WHITE));
    _highlight[1]->addDrawable(createBackground(COL_BLACK));
    _highlight[0]->addDrawable(createLabel(COL_BLACK, 1));
    _highlight[1]->addDrawable(createLabel(COL_WHITE, 0));
    _highlight[0]->addDrawable(createIcon());
    _highlight[1]->addDrawable(createIcon());
    _highlight[0]->setNodeMask(1);
    _highlight[1]->setNodeMask(1);
    _swHighlight->addChild(_highlight[0]);
    _swHighlight->addChild(_highlight[1]);
    _swHighlight->setSingleChildOn(0);
    StateSet *hstateset = _swHighlight->getOrCreateStateSet();
    hstateset->setMode(GL_LIGHTING, StateAttribute::OFF);

    // Create focus-dependent geometry:
    _focus[0]->addDrawable(createFrame(COL_WHITE));
    _focus[1]->addDrawable(createFrame(COL_YELLOW));
    _swFocus->addChild(_focus[0]);
    _swFocus->addChild(_focus[1]);
    _swFocus->setSingleChildOn(0);

    // Thick lines and lighting off:
    LineWidth *lineWidth = new LineWidth();
    lineWidth->setWidth(3.0f);
    StateSet *stateset = _swFocus->getOrCreateStateSet();
    stateset->setAttribute(lineWidth);
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);

    _node->addChild(_magnifyXF.get());
    _magnifyXF->addChild(_swHighlight.get());
    _magnifyXF->addChild(_swFocus.get());
}

void Card::createTip()
{
    StateSet *stateSet;

    _useTipDelay = false;
    _tipDelay = 0.0;

    _tipWidth = 2 * DEFAULT_CARD_WIDTH;
    _tipHeight = 0.35 * DEFAULT_CARD_HEIGHT;

    _tipGeomVertices = new Vec3Array(4);
    _tipFrameVertices = new Vec3Array(5);
    setTipSize(_tipWidth, _tipHeight);

    // create tooltip background
    Geometry *tipGeom = new Geometry;

    tipGeom->setVertexArray(_tipGeomVertices);

    Vec4Array *color = new Vec4Array(1);
    (*color)[0].set(1.0, 1.0, 1.0, 1.0);
    tipGeom->setColorArray(color);
    tipGeom->setColorBinding(Geometry::BIND_OVERALL);

    tipGeom->addPrimitiveSet(new DrawArrays(GL_QUADS, 0, 4));
    tipGeom->setUseDisplayList(false);

    stateSet = tipGeom->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);

    // create tooltip text
    _tipString = new osgText::Text();
    _tipString->setDataVariance(Object::DYNAMIC);
    _tipString->setFont(_font);
    _tipString->setColor(COL_BLACK);
    _tipString->setFontResolution(20, 20);
    _tipStringPos.set(0.0, 0.0, EPSILON_Z);
    _tipString->setPosition(_tipStringPos);
    _tipString->setCharacterSize(DEFAULT_FONT_SIZE * 0.4 / 0.25);
    _tipString->setMaximumWidth(TIP_WIDTH);
    _tipString->setMaximumHeight(1.0);
    _tipString->setAlignment(osgText::Text::CENTER_CENTER);

    // create tooltip frame
    Geometry *tipFrameGeom = new Geometry;

    tipFrameGeom->setVertexArray(_tipFrameVertices);

    Vec4Array *tipColor = new Vec4Array(1);
    (*tipColor)[0].set(0.0, 0.0, 0.0, 1.0);
    tipFrameGeom->setColorArray(tipColor);
    tipFrameGeom->setColorBinding(Geometry::BIND_OVERALL);

    tipFrameGeom->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINE_STRIP, 0, 5));
    tipFrameGeom->setUseDisplayList(false);

    stateSet = tipFrameGeom->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);

    _tipGeode = new Geode();
    _tipGeode->addDrawable(tipGeom);
    _tipGeode->addDrawable(tipFrameGeom);
    _tipGeode->addDrawable(_tipString);
    _tipGeode->setNodeMask(0);
    _tipVisible = false;

    MatrixTransform *trans = new MatrixTransform;
    Matrix mat = trans->getMatrix();
    mat.setTrans(Vec3(0.0, -0.375 * DEFAULT_CARD_HEIGHT, 0.25));
    trans->setMatrix(mat);
    trans->addChild(_tipGeode);

    _magnifyXF->addChild(trans);
}

void Card::showTip(bool flag)
{
    if (flag)
        _tipGeode->setNodeMask(~1);
    else
        _tipGeode->setNodeMask(0);
}

void Card::setTipSize(float width, float height)
{
    _tipWidth = width;
    _tipHeight = height;

    if (_tipWidth > TIP_WIDTH)
        _tipString->setMaximumWidth(_tipWidth);

    (*_tipGeomVertices)[0].set(-_tipWidth / 2.0, -_tipHeight / 2.0, 0);
    (*_tipGeomVertices)[1].set(_tipWidth / 2.0, -_tipHeight / 2.0, 0);
    (*_tipGeomVertices)[2].set(_tipWidth / 2.0, _tipHeight / 2.0, 0);
    (*_tipGeomVertices)[3].set(-_tipWidth / 2.0, _tipHeight / 2.0, 0);

    (*_tipFrameVertices)[0].set(-_tipWidth / 2.0, -_tipHeight / 2.0, EPSILON_Z);
    (*_tipFrameVertices)[1].set(_tipWidth / 2.0, -_tipHeight / 2.0, EPSILON_Z);
    (*_tipFrameVertices)[2].set(_tipWidth / 2.0, _tipHeight / 2.0, EPSILON_Z);
    (*_tipFrameVertices)[3].set(-_tipWidth / 2.0, _tipHeight / 2.0, EPSILON_Z);
    (*_tipFrameVertices)[4].set(-_tipWidth / 2.0, -_tipHeight / 2.0, EPSILON_Z);
}

void Card::setTipText(char text[], bool calculateWidth)
{
    if (calculateWidth)
        setTipSize(strlen(text) * DEFAULT_FONT_SIZE, _tipHeight);

    _tipString->setText(text);
}

void Card::setTipVisibility(bool flag)
{
    _tipVisible = flag;
}

void Card::enableTipDelay(bool flag)
{
    _useTipDelay = flag;
}

void Card::setTipDelay(double delay)
{
    _tipDelay = delay;

    enableTipDelay(true);
}

void Card::cursorEnter(InputDevice *dev)
{
    _inside = true;
    if (dev == dev->_interaction->_head)
    {
        Card::setFocus(true);
    }
    Card::setHighlighted(true);
    if (_magnification)
    {
        setSize(1.5);
    }

    if (_useTipDelay)
        _enterTime = time(NULL);
    else
        showTip(_tipVisible);
}

void Card::cursorUpdate(InputDevice *dev)
{
    if ((_useTipDelay) && (difftime(time(NULL), _enterTime) > _tipDelay))
        showTip(_tipVisible);

    std::list<CardListener *>::iterator iter;
    for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
    {
        (*iter)->cardCursorUpdate(this, dev);
    }
}

void Card::cursorLeave(InputDevice *dev)
{
    _inside = false;
    if (dev == dev->_interaction->_head)
    {
        Card::setFocus(false);
    }
    Card::setHighlighted(false);
    if (_magnification)
    {
        setSize(1.0);
    }

    showTip(false);
}

/// Turn depth test on/off: force to foreground
void Card::setDepthTest(bool dt)
{
    StateSet *stateset = _node->getOrCreateStateSet();
    stateset->setMode(GL_DEPTH_TEST,
                      (dt) ? (osg::StateAttribute::ON) : (osg::StateAttribute::OFF));
    stateset->setRenderBinDetails(11, "RenderBin");
}

/** Creates the text on the card.
 */
osgText::Text *Card::createLabel(const Vec4 &color, int index)
{
    _labelText[index]->setFont(_font);
    _labelText[index]->setColor(color);
    _labelText[index]->setFontResolution(20, 20);
    Vec3 pos(0.0, -DEFAULT_CARD_HEIGHT / 2.0 + (DEFAULT_CARD_HEIGHT - DEFAULT_CARD_WIDTH) / 2.0, 2.0 * EPSILON_Z);
    _labelText[index]->setPosition(pos);
    _labelText[index]->setCharacterSize(DEFAULT_FONT_SIZE);
    _labelText[index]->setMaximumWidth(DEFAULT_CARD_WIDTH);
    _labelText[index]->setMaximumHeight(DEFAULT_CARD_HEIGHT - DEFAULT_CARD_WIDTH);
    _labelText[index]->setAlignment(osgText::Text::CENTER_CENTER);
    _labelText[index]->setUseDisplayList(false);

    // Turn off lighting:
    StateSet *stateset = _labelText[index]->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);

    return _labelText[index];
}

/** This creates the background rectangle, which is what the cursor
  intersects with. All other parts of the card should be excluded from
  intersection testing.
*/
Geometry *Card::createBackground(const Vec4 &color)
{
    Geometry *geom = new Geometry();
    Vec3Array *vertices;
    Vec3 myCoords[] = {
        // bottom left
        Vec3(-DEFAULT_CARD_WIDTH / 2.0, -DEFAULT_CARD_HEIGHT / 2.0, 0.0),
        // bottom right
        Vec3(DEFAULT_CARD_WIDTH / 2.0, -DEFAULT_CARD_HEIGHT / 2.0, 0.0),
        // top right
        Vec3(DEFAULT_CARD_WIDTH / 2.0, DEFAULT_CARD_HEIGHT / 2.0, 0.0),
        // top left
        Vec3(-DEFAULT_CARD_WIDTH / 2.0, DEFAULT_CARD_HEIGHT / 2.0, 0.0),
    };

    int numCoords = sizeof(myCoords) / sizeof(osg::Vec3);
    vertices = new Vec3Array(numCoords, myCoords);
    geom->setVertexArray(vertices);

    Vec4Array *colors = new osg::Vec4Array(1);
    (*colors)[0].set(color[0], color[1], color[2], color[3]);
    //  colors->push_back(color);
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec3Array *normals = new osg::Vec3Array(1); // six sides of the cube
    (*normals)[0].set(0.0f, 0.0f, 1.0f);
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new DrawArrays(osg::PrimitiveSet::QUADS, 0, numCoords));

    return geom;
}

/** This creates an empty icon texture.
 */
Geometry *Card::createIcon()
{
    Geometry *geom = new Geometry();

    Vec3Array *vertices = new Vec3Array(4);
    float marginX = (DEFAULT_CARD_WIDTH - ICON_SIZE * DEFAULT_CARD_WIDTH) / 2.0;
    float marginY = marginX;
    // bottom left
    (*vertices)[0].set(-DEFAULT_CARD_WIDTH / 2.0 + marginX, DEFAULT_CARD_HEIGHT / 2.0 - marginY - ICON_SIZE * DEFAULT_CARD_WIDTH, EPSILON_Z);
    // bottom right
    (*vertices)[1].set(DEFAULT_CARD_WIDTH / 2.0 - marginX, DEFAULT_CARD_HEIGHT / 2.0 - marginY - ICON_SIZE * DEFAULT_CARD_WIDTH, EPSILON_Z);
    // top right
    (*vertices)[2].set(DEFAULT_CARD_WIDTH / 2.0 - marginX, DEFAULT_CARD_HEIGHT / 2.0 - marginY, EPSILON_Z);
    // top left
    (*vertices)[3].set(-DEFAULT_CARD_WIDTH / 2.0 + marginX, DEFAULT_CARD_HEIGHT / 2.0 - marginY, EPSILON_Z);
    geom->setVertexArray(vertices);

    Vec2Array *texcoords = new Vec2Array(4);
    (*texcoords)[0].set(0.0, 0.0);
    (*texcoords)[1].set(1.0, 0.0);
    (*texcoords)[2].set(1.0, 1.0);
    (*texcoords)[3].set(0.0, 1.0);
    geom->setTexCoordArray(0, texcoords);

    Vec3Array *normals = new Vec3Array(1);
    (*normals)[0].set(0.0f, 0.0f, 1.0f);
    geom->setNormalArray(normals);
    geom->setNormalBinding(Geometry::BIND_OVERALL);

    Vec4Array *colors = new Vec4Array(1);
    (*colors)[0].set(1.0, 1.0, 1.0, 1.0);
    geom->setColorArray(colors);
    geom->setColorBinding(Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));

    // Texture:
    StateSet *stateset = geom->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    stateset->setNestRenderBins(false);
    stateset->setTextureAttributeAndModes(0, _icon, StateAttribute::ON);

    return geom;
}

Geometry *Card::createFrame(const Vec4 &color)
{
    Geometry *geom = new Geometry();

    // Create lines vertices:
    Vec3Array *vertices = (Vec3Array *)geom->getVertexArray();
    if (!vertices)
        vertices = new Vec3Array(8);

    Vec3 Vmin(-DEFAULT_CARD_WIDTH / 2.0, -DEFAULT_CARD_HEIGHT / 2.0, 0);
    Vec3 Vmax(DEFAULT_CARD_WIDTH / 2.0, DEFAULT_CARD_HEIGHT / 2.0, 0);

    (*vertices)[0].set(Vmin[0], Vmin[1], Vmax[2]);
    (*vertices)[1].set(Vmax[0], Vmin[1], Vmax[2]);
    (*vertices)[2].set(Vmin[0], Vmin[1], Vmax[2]);
    (*vertices)[3].set(Vmin[0], Vmax[1], Vmax[2]);
    (*vertices)[4].set(Vmin[0], Vmax[1], Vmax[2]);
    (*vertices)[5].set(Vmax[0], Vmax[1], Vmax[2]);
    (*vertices)[6].set(Vmax[0], Vmin[1], Vmax[2]);
    (*vertices)[7].set(Vmax[0], Vmax[1], Vmax[2]);

    // Pass the created vertex array to the points geometry object:
    geom->setVertexArray(vertices);

    // Set colors:
    Vec4Array *colors = new Vec4Array(1);
    colors->push_back(color);
    geom->setColorArray(colors);
    geom->setColorBinding(Geometry::BIND_OVERALL);

    // Set normals:
    Vec3Array *normals = new Vec3Array();
    normals->push_back(Vec3(0.0f, 0.0f, 1.0f));
    geom->setNormalArray(normals);
    geom->setNormalBinding(Geometry::BIND_OVERALL);

    // This time we simply use primitive, and hardwire the number of coords
    // to use since we know up front:
    geom->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 8));

    geom->setUseDisplayList(false); // allow dynamic changes

    return geom;
}

/** Loads and displays an image as a button icon.
  @return true if image loaded successfully, false otherwise
*/
bool Card::loadImage(const std::string &filename)
{
    Image *image = NULL;

    image = osgDB::readImageFile(filename);
    if (image)
    {
        _icon->setImage(image);
        return true;
    }
    else
    {
        //std::cerr << "Cannot load image file " << filename << std::endl;
        return false;
    }
}

/** Loads and displays an image as a button icon.
  The file will be searched in the resources folder under osgcaveui.
  @return true if image loaded successfully, false otherwise
*/
bool Card::loadCUIImage(const std::string &filename)
{
    return loadImage(_resourcePath + filename);
}

void Card::setIconImage(Image *image)
{
    _icon->setImage(image);
}

void Card::buttonEvent(InputDevice *dev, int button)
{
    std::list<CardListener *>::iterator iter;
    if (button == 0)
    {
        if (dev->getButtonState(button) == 0)
        {
            for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
            {
                (*iter)->cardButtonEvent(this, 0, 0);
            }
        }
        else if (dev->getButtonState(button) == 1)
        {
            for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
            {
                (*iter)->cardButtonEvent(this, 0, 1);
            }
        }
    }
}

void Card::joystickEvent(InputDevice *)
{
}

void Card::wheelEvent(InputDevice *, int)
{
}

void Card::addCardListener(CardListener *c)
{
    _listeners.push_back(c);
}

void Card::setFocus(bool focus)
{
    _swFocus->setSingleChildOn((focus) ? 1 : 0);
    Widget::setFocus(focus);
}

void Card::setHighlighted(bool highlight)
{
    _swHighlight->setSingleChildOn((highlight) ? 1 : 0);
    Widget::setHighlighted(highlight);
}

float Card::getWidth()
{
    return DEFAULT_CARD_WIDTH;
}

float Card::getHeight()
{
    return DEFAULT_CARD_HEIGHT;
}

/** @param size 1.0 is default
 */
void Card::setSize(float size)
{
    Matrix nodeMat, trans, scale;
    Vec3 popOut;

    scale.makeScale(size, size, size);
    if (size > 1.0)
        popOut = _magnifyOffset;
    trans.makeTranslate(popOut);
    nodeMat = scale * trans;
    _magnifyXF->setMatrix(nodeMat);
}

void Card::setMagnificationMode(bool mag)
{
    _magnification = mag;
}
