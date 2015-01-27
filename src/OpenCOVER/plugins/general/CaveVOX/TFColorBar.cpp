/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// OSG:
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Switch>
#include <osg/TexEnv>
#include <osg/Depth>
#include <osg/LineWidth>
#include <osgText/Text>
#include <osgDB/ReadFile>

// CUI:
#include "TFColorBar.H"
#include "Interaction.H"
#include "CUI.H"

using namespace osg;
using namespace cui;

const float TFColorBar::DEFAULT_TEXTURE_WIDTH = 2;
const float TFColorBar::DEFAULT_TEXTURE_HEIGHT = 2;
const float TFColorBar::DEFAULT_LABEL_HEIGHT = 0.5;
const float TFColorBar::DEFAULT_FONT_SIZE = 0.4;

TFColorBar::TFColorBar(Interaction *interaction, float width, float height, vvTransFunc *tf)
    : Widget()
    , Events()
{
    StateSet *stateSet;

    _transFunc = tf;
    _interaction = interaction;

    _width = width;
    _height = height;

    createBackground();
    createTexturesGeometry();
    initTextures();

    _texture[0] = new Geode();
    _texture[0]->addDrawable(_geom);
    _texture[0]->addDrawable(_texGeom0);
    stateSet = _texGeom0->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    stateSet->setTextureAttributeAndModes(StateAttribute::TEXTURE, _tex0, StateAttribute::ON);
    stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    _tex0->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);

    _texture[1] = new Geode();
    _texture[1]->addDrawable(_geom);
    _texture[1]->addDrawable(_texGeom0);
    stateSet = _texGeom0->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    stateSet->setTextureAttributeAndModes(StateAttribute::TEXTURE, _tex0, StateAttribute::ON);

    _swTexture = new Switch();
    _swTexture->addChild(_texture[0]);
    _swTexture->addChild(_texture[1]);
    _swTexture->setSingleChildOn(0);

    _node->addChild(_swTexture.get());

    _leftGeode = new Geode();
    _rightGeode = new Geode();

    MatrixTransform *transLeft = new MatrixTransform();
    MatrixTransform *transRight = new MatrixTransform();

    Matrix trans;

    trans.makeTranslate(Vec3(-_width / 2.0, -_height / 2.0 - DEFAULT_LABEL_HEIGHT / 2.0, 3 * EPSILON_Z));
    transLeft->setMatrix(trans);
    transLeft->addChild(_leftGeode);

    trans.makeTranslate(Vec3(+_width / 2.0, -_height / 2.0 - DEFAULT_LABEL_HEIGHT / 2.0, 3 * EPSILON_Z));
    transRight->setMatrix(trans);
    transRight->addChild(_rightGeode);

    _node->addChild(transLeft);
    _node->addChild(transRight);

    _moveWidget = false;

    _interaction->addListener(this, this);
}

TFColorBar::~TFColorBar()
{
    _defaultTexImage->unref();
}

/** 
    Create texture geometry.
*/
void TFColorBar::createBackground()
{
    _geom = new Geometry();

    initVertices();

    _geom->setVertexArray(_vertices);

    Vec3Array *normals = new Vec3Array(1);
    (*normals)[0].set(0.0f, 0.0f, 1.0f);
    _geom->setNormalArray(normals);
    _geom->setNormalBinding(Geometry::BIND_OVERALL);

    Vec4Array *colors = new Vec4Array(1);
    (*colors)[0].set(0.0, 0.0, 0.0, 1.0);
    _geom->setColorArray(colors);
    _geom->setColorBinding(Geometry::BIND_OVERALL);

    _geom->addPrimitiveSet(new DrawArrays(GL_QUADS, 0, 4));
    _geom->setUseDisplayList(false);
}

void TFColorBar::createTexturesGeometry()
{
    Vec3Array *vertices = new Vec3Array(4);
    (*vertices)[0].set(-_width / 2.0, -_height / 2.0 + DEFAULT_LABEL_HEIGHT / 2.0, EPSILON_Z); // bottom left
    (*vertices)[1].set(_width / 2.0, -_height / 2.0 + DEFAULT_LABEL_HEIGHT / 2.0, EPSILON_Z); // bottom right
    (*vertices)[2].set(_width / 2.0, _height / 2.0 + DEFAULT_LABEL_HEIGHT / 2.0, EPSILON_Z); // top right
    (*vertices)[3].set(-_width / 2.0, _height / 2.0 + DEFAULT_LABEL_HEIGHT / 2.0, EPSILON_Z); // top left

    Vec3Array *normals = new Vec3Array(1);
    (*normals)[0].set(0.0f, 0.0f, 1.0f);

    Vec4Array *colors = new Vec4Array(1);
    (*colors)[0].set(1.0, 1.0, 1.0, 1.0);

    Vec2Array *texCoords = new Vec2Array(4);
    (*texCoords)[0].set(0.0, 0.0);
    (*texCoords)[1].set(1.0, 0.0);
    (*texCoords)[2].set(1.0, 1.0);
    (*texCoords)[3].set(0.0, 1.0);

    _texGeom0 = new Geometry();

    _texGeom0->setVertexArray(vertices);

    _texGeom0->setNormalArray(normals);
    _texGeom0->setNormalBinding(Geometry::BIND_OVERALL);

    _texGeom0->setColorArray(colors);
    _texGeom0->setColorBinding(Geometry::BIND_OVERALL);

    _texGeom0->addPrimitiveSet(new DrawArrays(GL_QUADS, 0, 4));
    _texGeom0->setUseDisplayList(false);

    _texGeom0->setTexCoordArray(0, texCoords);

    _texGeom1 = new Geometry();

    _texGeom1->setVertexArray(vertices);

    _texGeom1->setNormalArray(normals);
    _texGeom1->setNormalBinding(Geometry::BIND_OVERALL);

    _texGeom1->setColorArray(colors);
    _texGeom1->setColorBinding(Geometry::BIND_OVERALL);

    _texGeom1->addPrimitiveSet(new DrawArrays(GL_QUADS, 0, 4));
    _texGeom1->setUseDisplayList(false);

    _texGeom1->setTexCoordArray(0, texCoords);
}

void TFColorBar::initVertices()
{
    _vertices = new Vec3Array(4);
    setVertices();
}

void TFColorBar::setVertices()
{
    (*_vertices)[0].set(-_width / 2.0, -_height / 2.0, 0.0); // bottom left
    (*_vertices)[1].set(_width / 2.0, -_height / 2.0, 0.0); // bottom right
    (*_vertices)[2].set(_width / 2.0, _height / 2.0, 0.0); // top right
    (*_vertices)[3].set(-_width / 2.0, _height / 2.0, 0.0); // top left
}

void TFColorBar::initTextures()
{
    createDefaultTexImage();

    _tex0 = new Texture2D();

    _tex0->setWrap(Texture::WRAP_S, Texture::CLAMP);
    _tex0->setWrap(Texture::WRAP_T, Texture::CLAMP);
    _tex0->setImage(_defaultTexImage);

    _tex1 = new Texture2D();

    _tex1->setWrap(Texture::WRAP_S, Texture::CLAMP);
    _tex1->setWrap(Texture::WRAP_T, Texture::CLAMP);
    _tex1->setImage(_defaultTexImage);
}

void TFColorBar::createDefaultTexImage()
{
    _defaultTexImage = new Image();

    // default texture is black:
    unsigned char *tex = new unsigned char[2 * 2 * 4];
    tex[0] = tex[4] = tex[8] = tex[12] = 0;
    tex[1] = tex[5] = tex[9] = tex[13] = 0;
    tex[2] = tex[6] = tex[10] = tex[14] = 0;
    tex[3] = tex[7] = tex[11] = tex[15] = 255;

    _defaultTexImage->setImage(2, 2, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, tex, Image::USE_NEW_DELETE);
}

void TFColorBar::addGeomToLeft(Drawable *geom)
{
    _leftGeode->addDrawable(geom);
}

void TFColorBar::addGeomToRight(Drawable *geom)
{
    _rightGeode->addDrawable(geom);
}

void TFColorBar::setImage(int num, Image *img)
{
    switch (num)
    {
    case 0:
        _tex0->setImage(img);
        break;
    case 1:
        _tex1->setImage(img);
        break;
    default:
        break;
    }
}

void TFColorBar::showTexture(int num)
{
    switch (num)
    {
    case 0:
        _swTexture->setSingleChildOn(0);
        break;
    case 1:
        _swTexture->setSingleChildOn(1);
        break;
    default:
        break;
    }
}

void TFColorBar::initLabels()
{
    StateSet *stateSet;
    Vec3 pos(0.0, -_height / 2.0, EPSILON_Z);

    _label0 = new osgText::Text();
    _label0->setFont(osgText::readFontFile("arial.ttf"));
    _label0->setColor(COL_WHITE);
    _label0->setFontResolution(20, 20);
    _label0->setPosition(pos);
    _label0->setCharacterSize(DEFAULT_FONT_SIZE);
    _label0->setMaximumWidth(_width);
    _label0->setMaximumHeight(DEFAULT_LABEL_HEIGHT);
    _label0->setAlignment(osgText::Text::CENTER_CENTER);
    _label0->setUseDisplayList(false);

    stateSet = _label0->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);

    _label1 = new osgText::Text();
    _label1->setFont(osgText::readFontFile("arial.ttf"));
    _label1->setColor(COL_WHITE);
    _label1->setFontResolution(20, 20);
    _label1->setPosition(pos);
    _label1->setCharacterSize(DEFAULT_FONT_SIZE);
    _label1->setMaximumWidth(_width);
    _label1->setMaximumHeight(DEFAULT_LABEL_HEIGHT);
    _label1->setAlignment(osgText::Text::CENTER_CENTER);
    _label1->setUseDisplayList(false);

    stateSet = _label1->getOrCreateStateSet();
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF);
}

void TFColorBar::setLabelText(int num, const std::string &text)
{
    switch (num)
    {
    case 0:
        _label0->setText(text, osgText::String::ENCODING_UTF8);
        break;
    case 1:
        _label1->setText(text, osgText::String::ENCODING_UTF8);
        break;
    default:
        break;
    }
}

float TFColorBar::getWidth()
{
    return _width;
}

float TFColorBar::getHeight()
{
    return (_height + DEFAULT_LABEL_HEIGHT);
}

void TFColorBar::cursorEnter(InputDevice *dev)
{
}

void TFColorBar::cursorLeave(InputDevice *dev)
{
}

bool TFColorBar::isInside()
{
    return false;
}

/** 
    Loads and displays an image.
    @return true if image loaded successfully, false otherwise
*/
bool TFColorBar::loadImage(const std::string &filename)
{
    return false;
}

void TFColorBar::buttonEvent(InputDevice *dev, int button)
{
    if (button == 0)
    {
        if (dev->getButtonState(button) == 1)
        {
            Matrix l2w;
            osgDrawObj::computeLocal2Root(getNode(), l2w);
            Matrix w2l = Matrix::inverse(l2w);

            Vec3 wPt = dev->getIsectPoint();

            Vec3 lPt = wPt * w2l;

            float x = (getWidth() / 2 + lPt[0]) / getWidth();

            bool widgetFound = false;

            // Find which Widget (if any) is there
            _transFunc->_widgets.first();
            cerr << _transFunc->_widgets.count() << endl;
            for (int i = 0; i <= _transFunc->_widgets.count(); ++i)
            {
                vvTFWidget *w = _transFunc->_widgets.getData();

                if (dynamic_cast<vvTFColor *>(w) != NULL)
                {
                    cerr << w->_pos[0] << " " << x << endl;
                    float dx = fabs(w->_pos[0] - x);
                    if (dx <= 0.05)
                    {
                        cerr << "I SHOULD BE SELECTING SOMETHING!" << endl;
                        selectWidget(w);
                        widgetFound = true;
                    }
                }
                _transFunc->_widgets.next();
            }
            if (widgetFound == false)
            {
                selectWidget(NULL);
                return;
            }
            _moveWidget = true;
        }
        else if (dev->getButtonState(button) == 0)
            _moveWidget = false;
    }
}

void TFColorBar::cursorUpdate(InputDevice *dev)
{
    if (_moveWidget == true)
    {
        std::list<BarListener *>::iterator iter;
        for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
        {
            Matrix l2w;
            osgDrawObj::computeLocal2Root(getNode(), l2w);
            Matrix w2l = Matrix::inverse(l2w);

            Vec3 wPt = dev->getIsectPoint();
            Vec3 lPt = wPt * w2l;

            float x = (getWidth() / 2 + lPt[0]) / getWidth();

            (*iter)->moveWidget(x);
        }
    }
}

void TFColorBar::addBarListener(BarListener *b)
{
    _listeners.push_back(b);
}

void TFColorBar::selectWidget(vvTFWidget *w)
{
    std::list<BarListener *>::iterator iter;

    for (iter = _listeners.begin(); iter != _listeners.end(); ++iter)
    {
        (*iter)->handleSelection(w);
    }
}

void TFColorBar::joystickEvent(InputDevice *)
{
}

void TFColorBar::wheelEvent(InputDevice *, int)
{
}
