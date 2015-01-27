/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DrawObj.H"
#include "DrawMgr.H"
#include <GL/glut.h>

#include "TFPanel.H"
#include "CUI.H"

using namespace osg;
using namespace cui;

class MyDebugObj2 : public DrawObj
{
protected:
    double _x, _y, _z;
    Vec3 _col;

public:
    MyDebugObj2(Vec3 color, double x, double y, double z)
        : DrawObj("MyDebugObj")
        , _x(x)
        , _y(y)
        , _z(z)
        , _col(color)
    {
    }
    void draw()
    {
        glColor3f(_col[0], _col[1], _col[2]);
        glPushMatrix();
        glTranslated(_x, _y, _z);
        glutSolidCube(0.1); //,7,7);
        glPopMatrix();
    }
};
void
addDebugPoint2(Vec3 col, double x, double y, double z)
{
    DRAWMGR::registerObj(new MyDebugObj2(col, x, y, z));
}

TFPanel::TFPanel(Interaction *interaction, osgDrawObj *odo, Appearance app, Movability mov)
    : Panel(interaction, odo, app, mov)
{
    _selectedWidget = NULL;

    _bGaussian = new Button(interaction);
    //  _bGaussian->addCardListener(this);
    _bGaussian->setText("New Gaussian");
    addCard(_bGaussian, 2, 3);

    _bPyramid = new Button(interaction);
    //_bPyramid->addCardListener(this);
    _bPyramid->setText("New Pyramid");
    addCard(_bPyramid, 3, 3);

    _bNewColor = new Button(interaction);
    //_bNewColor->addCardListener(this);
    _bNewColor->setText("New Color");
    addCard(_bNewColor, 4, 3);

    _bDelete = new Button(interaction);
    //_bDelete->addCardListener(this);
    _bDelete->setText("Delete");
    addCard(_bDelete, 5, 3);

    _cbSelectColor = new CheckBox(interaction);
    //_cbSelectColor->addCardListener(this);
    _cbSelectColor->setText("Select Color");
    _cbSelectColor->setVisible(false);
    addCard(_cbSelectColor, 3, 4);

    _dTopWidth = new Dial(interaction); //, false);
    //_dTopWidth->addCardListener(this);
    _dTopWidth->addDialChangeListener(this);
    _dTopWidth->setMin(0.0f);
    _dTopWidth->setMax(2.0f);
    _dTopWidth->setText("Top Width");
    _dTopWidth->setVisible(false);
    addCard(_dTopWidth, 4, 4);

    _dBottomWidth = new Dial(interaction); //, false);
    //_dBottomWidth->addCardListener(this);
    _dBottomWidth->addDialChangeListener(this);
    _dBottomWidth->setMin(0.0f);
    _dBottomWidth->setMax(2.0f);
    _dBottomWidth->setText("Bottom Width");
    _dBottomWidth->setVisible(false);
    addCard(_dBottomWidth, 5, 4);

    _bCancel = new Button(interaction);
    //_bCancel->addCardListener(this);
    _bCancel->setText("Cancel");
    addCard(_bCancel, 7, 4);

    _bOK = new Button(interaction);
    //_bOK->addCardListener(this);
    _bOK->setText("OK");
    addCard(_bOK, 6, 4);

    _selColor = 0;

    _bColor = new Button(interaction);
    //_bColor->addCardListener(this);
    _bColor->setText("Colors");
    _bColor->loadImage("resources/palette.tif");
    addCard(_bColor, 1, 4);

    _transFunc = new vvTransFunc();
    _transFunc->setDefaultColors(0);
    _transFunc->setDefaultAlpha(0);

    _barTexture = new TFColorBar(_interaction, 10, 0.5, _transFunc);
    _barTexture->showTexture(0);
    _barTexture->addBarListener(this);

    _node->addChild(_barTexture->getNode());

    Matrix barTrans;
    barTrans.makeTranslate(_topLeft[0] + _borderSizeX + _barTexture->getWidth() / 2.0f, _topLeft[1] - _borderSizeY - _barTexture->getHeight() / 2.0f, 2.0f * Widget::EPSILON_Z);
    _barTexture->setMatrix(barTrans);
    _barImage = new Image();

    _boxImage = new Image();
    _boxTexture = new TFColorBox(_interaction, 10.0f, 4.0f, _transFunc);
    _boxTexture->showTexture(0);
    _boxTexture->addBarListener(this);

    _node->addChild(_boxTexture->getNode());

    Matrix boxTrans;
    boxTrans.makeTranslate(_topLeft[0] + _borderSizeX + _barTexture->getWidth() / 2.0f, _topLeft[1] - _borderSizeY - _barTexture->getHeight() / 2.0f - 2.25, 2.0f * Widget::EPSILON_Z);
    _boxTexture->setMatrix(boxTrans);

    draw1DTF();

    setWidth(10.75);
}
TFPanel::~TFPanel()
{
}

void TFPanel::draw1DTF()
{
    drawColorTexture();
    drawPinBackground();
    drawPinLines();

    std::list<TFListener *>::iterator iter;
    for (iter = _TFListeners.begin(); iter != _TFListeners.end(); ++iter)
        (*iter)->setFunction(_transFunc);
}

void TFPanel::drawColorTexture()
{
    int width = 256;
    int height = 2;

    unsigned char *tempArray = new unsigned char[width * height * 4];

    _transFunc->makeColorBar(width, tempArray);

    //Invert the array...
    unsigned char *barArray = new unsigned char[width * height * 4];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                barArray[4 * ((height - 1 - i) * width + j) + k] = tempArray[4 * (i * width + j) + k];
            }
        }
    }

    delete tempArray;

    _barImage->setImage(width, height, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, barArray, Image::USE_NEW_DELETE);
    _barImage->dirty();

    _barTexture->setImage(0, _barImage);
}

void TFPanel::drawPinBackground()
{
    int boxWidth = 10;
    int boxHeight = 4;
    unsigned char *tempArray = new unsigned char[256 * 256 * 4];

    _transFunc->makeAlphaTexture(256, 256, tempArray);

    //Invert the array...
    unsigned char *boxArray = new unsigned char[256 * 256 * 4];
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            int offset = 4 * (256 * 256 - i * 256 - j);
            for (int k = 0; k < 4; k++)
            {
                boxArray[4 * ((256 - i) * 256 + j) + k] = tempArray[4 * (i * 256 + j) + k];
            }
        }
    }

    delete tempArray;

    _boxImage->setImage(256, 256, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, boxArray, Image::USE_NEW_DELETE);
    _boxImage->dirty();
    _boxTexture->setImage(0, _boxImage);
}

void TFPanel::drawPinLines()
{
    std::list<TFColorWidget *>::iterator iter;
    for (iter = _widgets.begin(); iter != _widgets.end(); ++iter)
    {
        if (*iter != NULL)
            (*iter)->setVisible(false);
        // delete *iter;
    }

    _transFunc->_widgets.first();
    for (int i = 0; i < _transFunc->_widgets.count(); ++i)
    {
        drawPinLine(_transFunc->_widgets.getData());
        _transFunc->_widgets.next();
    }
}

void TFPanel::drawPinLine(vvTFWidget *w)
{
    if (dynamic_cast<vvTFColor *>(w) != NULL)
    {
        float *tfPos = w->_pos;

        Vec3 pStart = Vec3(tfPos[0] * 10 - 5, tfPos[1], Widget::EPSILON_Z);
        Vec3 pEnd = Vec3(tfPos[0] * 10 - 5, tfPos[1] + 0.5, Widget::EPSILON_Z);

        TFColorWidget *tfcw;
        if (w == _selectedWidget)
        {
            tfcw = new TFColorWidget(_interaction, NULL, 8);
        }
        else
            tfcw = new TFColorWidget(_interaction, NULL, 2);
        tfcw->setVertices(pStart, pEnd);
        tfcw->setVisible(true);
        _barTexture->getNode()->addChild(tfcw->getNode());
        _widgets.push_back(tfcw);
    }
    else if (dynamic_cast<vvTFPyramid *>(w) != NULL || dynamic_cast<vvTFBell *>(w) != NULL)
    {
        float *tfPos = w->_pos;

        Vec3 pStart = Vec3(tfPos[0] * 10 - 5, -1.75, Widget::EPSILON_Z);
        Vec3 pEnd = Vec3(tfPos[0] * 10 - 5, 2.25, Widget::EPSILON_Z);

        TFColorWidget *tfcw;
        if (w == _selectedWidget)
            tfcw = new TFColorWidget(_interaction, NULL, 8);
        else
            tfcw = new TFColorWidget(_interaction, NULL, 2);
        tfcw->setVertices(pStart, pEnd);
        tfcw->setVisible(true);
        _boxTexture->getNode()->addChild(tfcw->getNode());

        _widgets.push_back(tfcw);
    }
    else
    {
        cerr << "Unknown pin line!" << endl;
    }
}

void TFPanel::addTFListener(TFListener *list)
{
    _TFListeners.push_back(list);
}
void TFPanel::cursorEnter(cui::InputDevice *dev)
{
    Panel::cursorEnter(dev);
}
void TFPanel::cursorUpdate(cui::InputDevice *dev)
{
    Panel::cursorUpdate(dev);
}
void TFPanel::cursorLeave(cui::InputDevice *dev)
{
    Panel::cursorLeave(dev);
}
void TFPanel::buttonEvent(cui::InputDevice *dev, int button)
{
    Panel::buttonEvent(dev, button);
}

void TFPanel::handleSelection(vvTFWidget *w)
{
    _selectedWidget = w;

    if (dynamic_cast<vvTFPyramid *>(w) != NULL) // || dynamic_cast<vvTFBell*>(w) != NULL)
    {
        vvTFPyramid *p = dynamic_cast<vvTFPyramid *>(w);
        _dBottomWidth->setValue(p->_bottom[0]);
        _dBottomWidth->setVisible(true);
        _dTopWidth->setValue(p->_top[0]);
        _dTopWidth->setVisible(true);
        _cbSelectColor->setVisible(false);
        _cbSelectColor->setChecked(false);
        std::list<TFListener *>::iterator iter;
        for (iter = _TFListeners.begin(); iter != _TFListeners.end(); ++iter)
            (*iter)->getNextColor(false);
    }
    else if (dynamic_cast<vvTFColor *>(w) != NULL)
    {
        _dBottomWidth->setVisible(false);
        _dTopWidth->setVisible(false);
        _cbSelectColor->setVisible(true);
    }
    else
    {
        _dBottomWidth->setVisible(false);
        _dTopWidth->setVisible(false);
        _cbSelectColor->setVisible(false);
        _cbSelectColor->setChecked(false);
        std::list<TFListener *>::iterator iter;
        for (iter = _TFListeners.begin(); iter != _TFListeners.end(); ++iter)
            (*iter)->getNextColor(false);
    }
    draw1DTF();
}

void TFPanel::moveWidget(float x)
{
    if (_selectedWidget != NULL)
    {
        if (x < 0)
            x = 0;
        else if (x > 1)
            x = 1;
        _selectedWidget->_pos[0] = x;
        draw1DTF();
    }
}

bool TFPanel::cardButtonEvent(Card *c, int button, int state)
{
    std::cerr << "TFPanel" << endl;

    if (button == 0 && state == 1)
    {
        if (c == _bGaussian)
        {
            vvTFWidget *w = new vvTFBell(vvColor(), false, 1.0f, 0.5f, 0.2f);
            _transFunc->_widgets.append(w, vvSLNode<vvTFWidget *>::NORMAL_DELETE);
            _selectedWidget = w;
        }
        else if (c == _bPyramid)
        {
            vvTFWidget *w = new vvTFPyramid(vvColor(), false, 1.0f, 0.5f, 0.4f, 0.2f);
            _transFunc->_widgets.append(w, vvSLNode<vvTFWidget *>::NORMAL_DELETE);
            _selectedWidget = w;
        }
        else if (c == _bNewColor)
        {
            vvTFWidget *w = new vvTFColor(vvColor(), 0.5f);
            _transFunc->_widgets.append(w, vvSLNode<vvTFWidget *>::NORMAL_DELETE);
            _selectedWidget = w;
        }
        else if (c == _bColor)
        {
            if (++_selColor > _transFunc->getNumDefaultColors())
                _selColor = 0;

            _transFunc->setDefaultColors(_selColor);
        }
        else if (c == _bDelete)
        {
            if (_selectedWidget != NULL)
            {
                _transFunc->_widgets.find(_selectedWidget);
                _transFunc->_widgets.remove();
                _selectedWidget = NULL;
            }
        }
        else if (c == _cbSelectColor)
        {
            std::list<TFListener *>::iterator iter;
            for (iter = _TFListeners.begin(); iter != _TFListeners.end(); ++iter)
                (*iter)->getNextColor(!_cbSelectColor->isChecked());
        }
        else if (c == _bCancel)
        {
            this->setVisible(false);
            std::list<TFListener *>::iterator iter;
            for (iter = _TFListeners.begin(); iter != _TFListeners.end(); ++iter)
                (*iter)->setTFVisible(false);
        }
        else if (c == _bOK)
        {
            std::list<TFListener *>::iterator iter;
            for (iter = _TFListeners.begin(); iter != _TFListeners.end(); ++iter)
            {
                (*iter)->setFunction(_transFunc);
                //(*iter)->setFunction(false);
            }
        }

        draw1DTF();
    }

    return false;
}

void TFPanel::dialValueChanged(Dial *dial, float newValue)
{
    cerr << newValue << endl;
    if (_selectedWidget != NULL)
    {
        vvTFPyramid *w;
        if ((w = dynamic_cast<vvTFPyramid *>(_selectedWidget)) != NULL)
        {
            if (dial == _dTopWidth)
            {
                w->_top[0] = newValue;
            }
            else if (dial == _dBottomWidth)
                w->_bottom[0] = newValue;
        }
    }
    draw1DTF();
}

bool TFPanel::cardCursorUpdate(Card *c, InputDevice *dev)
{

    Panel::cardCursorUpdate(c, dev);
    return false;
}

void TFPanel::setColor(Vec4 col)
{
    vvTFColor *w;
    if ((w = dynamic_cast<vvTFColor *>(_selectedWidget)) != NULL)
    {
        vvColor color;
        color.setRGB(col[0] / 255.0, col[1] / 255.0, col[2] / 255.0);
        w->_col = color;
        draw1DTF();
    }
}

void TFPanel::setTF(vvTransFunc *newTF)
{
    _transFunc = new vvTransFunc(newTF);
}

void TFPanel::displayWheel(bool b)
{
    _cbSelectColor->setChecked(b);
}
