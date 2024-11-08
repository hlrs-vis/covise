/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "TUITF2DEditor.h"
#include "TUITFEWidgets.h"

#include <QPolygon>
#include <math.h>

/*!
\class TUIColorPoint
\brief This class creates provides the marker widget
*/

/*****************************************************************************
 *
 * Class TUIColorPoint  (Marker)
 *
 *****************************************************************************/

TUIColorPoint::TUIColorPoint()
    : TUITFEWidget(TUITFEWidget::TF_COLOR)
{
}

void TUIColorPoint::setData(float x, QColor col)
{
    pos = x;
    color = col;
}

void TUIColorPoint::setColor(QColor col)
{
    color = col;
}

void TUIColorPoint::paint(QPainter &p, int w, bool active)
{
    p.setBackgroundMode(Qt::TransparentMode);

    QPolygon poly;

    int x = int(getX() * w);
    poly.setPoints(3,
                   x, TUITF1DEditor::colorWidgetPos,
                   x - TUITF1DEditor::markerSize / 2, TUITF1DEditor::colorWidgetPos + TUITF1DEditor::markerSize,
                   x + TUITF1DEditor::markerSize / 2, TUITF1DEditor::colorWidgetPos + TUITF1DEditor::markerSize);

    QPen pen;
    if (active)
    {
        pen.setWidth(3);
        p.setPen(pen);
        //p.drawPolyline(poly);
    }
    else
    {
        pen.setWidth(1);
        p.setPen(pen);
    }
    p.setBrush(QBrush(color, Qt::SolidPattern));
    p.drawPolygon(poly);
}

bool TUIColorPoint::contains(int x, int y, int w)
{
    int xx = int(pos * w);
    if ((abs(xx - x) < TUITF1DEditor::clickThreshold) && (abs(y - TUITF1DEditor::colorWidgetPos - TUITF1DEditor::halfMarkerSize) < TUITF1DEditor::clickThreshold))
        return true;
    return false;
}

/*!
\class TUIAlphaTriangle
\brief This class provides the widget for setting alpha trapezoid functions
*/

/*****************************************************************************
 *
 * Class TUIAlphaTriangle  (Marker)
 *
 *****************************************************************************/

TUIAlphaTriangle::TUIAlphaTriangle()
    : TUITFEWidget(TUITFEWidget::TF_PYRAMID)
{
    ww = 0;
}

void TUIAlphaTriangle::setData(float x, int alpha, float xb, float xt)
{
    this->pos = x;
    this->alpha = alpha;
    this->xb = xb;
    this->xt = xt;
    setModified();
}

void TUIAlphaTriangle::setAlpha(int alpha)
{
    if (alpha > 255)
        this->alpha = 255;
    else if (alpha < 0)
        this->alpha = 0;
    else
        this->alpha = alpha;
    setModified();
}

void TUIAlphaTriangle::setX(float x)
{
    pos = x;
    setModified();
}

void TUIAlphaTriangle::setXb(float xb)
{
    if (xb < this->xt)
        this->xb = xt;
    else
        this->xb = xb;
    setModified();
}

void TUIAlphaTriangle::setXt(float xt)
{
    if (xt < 0.0)
        this->xt = 0.0;
    else if (xt > this->xb)
        this->xt = this->xb;
    else
        this->xt = xt;
    setModified();
}

void TUIAlphaTriangle::paint(QPainter &p, int w, bool active)
{
    if (isModified() || ww != w)
    {
        ww = w;
        setModified(false);
        updateAlphaMap();
    }

    p.setBackgroundMode(Qt::TransparentMode);

    int x = int(getX() * w);
    int xBase = int((pos - xb / 2.0f) * w);
    int alphaOffset = TUITF1DEditor::markerSize + int((1.0 - (alpha / 255.0)) * TUITF1DEditor::panelSize);
    int xTop = int((pos + xt / 2.0f) * w);

    //first, draw the map
    p.drawImage(0, TUITF1DEditor::panelPos, alphaMap);

    QPen pen;

    pen.setWidth(2);
    p.setPen(pen);
    p.drawLine(xBase, alphaOffset, x, alphaOffset);
    p.drawLine(x, alphaOffset, xTop, alphaOffset);

    pen.setWidth(1);
    p.setPen(pen);

    if (active)
    {
        p.setBrush(QBrush(Qt::red));
    }
    else
    {
        p.setBrush(QBrush(Qt::gray));
    }

    QPolygon poly;

    poly.setPoints(3,
                   x - TUITF1DEditor::markerSize / 2, TUITF1DEditor::alphaWidgetPos,
                   x, TUITF1DEditor::alphaWidgetPos + TUITF1DEditor::markerSize,
                   x + TUITF1DEditor::markerSize / 2, TUITF1DEditor::alphaWidgetPos);

    //p.drawEllipse(x, TUITF1DEditor::alphaWidgetPos, TUITF1DEditor::markerSize, TUITF1DEditor::markerSize);
    p.drawPolygon(poly);

    if (active)
    {
        p.drawEllipse(xBase - TUITF1DEditor::markerSize / 2, alphaOffset, TUITF1DEditor::markerSize, TUITF1DEditor::markerSize);
        p.drawEllipse(x - TUITF1DEditor::markerSize / 2, alphaOffset, TUITF1DEditor::markerSize, TUITF1DEditor::markerSize);
        p.drawEllipse(xTop - TUITF1DEditor::markerSize / 2, alphaOffset, TUITF1DEditor::markerSize, TUITF1DEditor::markerSize);
    }
}

TUITFEWidget::HandleType TUIAlphaTriangle::testHit(int x, int y)
{
    int xPos = int(getX() * ww);
    int xBase = int((pos - xb / 2.0f) * ww);
    int alphaPos = TUITF1DEditor::panelPos + TUITF1DEditor::halfMarkerSize + int((1.0 - (alpha / 255.0)) * TUITF1DEditor::panelSize);
    int xTop = int((pos + xt / 2.0f) * ww);

    //base handle?
    if ((abs(x - xBase) < TUITF1DEditor::clickThreshold) && (abs(y - alphaPos) < TUITF1DEditor::clickThreshold))
        selectedHandle = HT_BOTTOM;

    else if ((abs(x - xTop) < TUITF1DEditor::clickThreshold) && (abs(y - alphaPos) < TUITF1DEditor::clickThreshold))
        selectedHandle = HT_TOP;

    else if ((abs(x - xPos) < TUITF1DEditor::clickThreshold) && (abs(y - alphaPos) < TUITF1DEditor::clickThreshold))
        selectedHandle = HT_MIDDLE;

    else
        selectedHandle = HT_NONE;

    return selectedHandle;
}

void TUIAlphaTriangle::updateAlphaMap()
{
    alphaMap = QImage(ww, TUITF1DEditor::panelSize, QImage::Format_ARGB32_Premultiplied);
    alphaMap.fill(0);

    QPainter p(&alphaMap);

    p.setBackgroundMode(Qt::TransparentMode);

    int x1Base = int((pos - xb / 2.0f) * ww);
    int x2Base = int((pos + xb / 2.0f) * ww);
    int alphaOffset = int((1.0 - (alpha / 255.0)) * TUITF1DEditor::panelSize);
    int x1Top = int((pos - xt / 2.0f) * ww);
    int x2Top = int((pos + xt / 2.0f) * ww);

    QPolygon poly;
    poly.setPoints(4,
                   x1Base, TUITF1DEditor::panelSize,
                   x1Top, alphaOffset,
                   x2Top, alphaOffset,
                   x2Base, TUITF1DEditor::panelSize);

    p.setBrush(QBrush(QColor(255, 255, 255, 100)));
    p.drawPolygon(poly);
}

bool TUIAlphaTriangle::contains(int x, int y, int w)
{
    int xx = int(pos * w);
    if ((abs(xx - x) < TUITF1DEditor::clickThreshold) && (abs(y - TUITF1DEditor::alphaWidgetPos - TUITF1DEditor::halfMarkerSize) < TUITF1DEditor::clickThreshold))
        return true;
    return false;
}

/*!
\class TUIAlphaFree
\brief This class provides the widget for designing alpha free-form functions
*/

/*****************************************************************************
 *
 * Class TUIAlphaFree  (Area)
 *
 *****************************************************************************/

TUIAlphaFree::TUIAlphaFree()
    : alphaMap(alphaMapSize, alphaMapSize, QImage::Format_ARGB32_Premultiplied)
{
    erase();
}

void TUIAlphaFree::beginDraw()
{
    drawing = true;
    lastX = -1;
    lastY = -1;
}

void TUIAlphaFree::endDraw()
{
    drawing = false;
}

void TUIAlphaFree::addLevel(float xPos, float yPos)
{
    hasData = true;

    int posX = int(xPos * (alphaMapSize - 1));
    int posY = int(yPos * (alphaMapSize - 1));

    if (drawing)
    {
        if (lastX >= 0)
        {
            if (posX > lastX)
            {
                for (int currX = lastX + 1; currX <= posX; ++currX)
                {
                    addAlphaInternal(currX, posY);
                }
            }
            else if (posX < lastX)
            {
                for (int currX = posX; currX < lastX; ++currX)
                {
                    addAlphaInternal(currX, posY);
                }
            }
        }
        lastX = posX;
        lastY = posY;
    }
    else
    {
        addAlphaInternal(posX, posY);
    }
}

void TUIAlphaFree::addAlphaInternal(int currX, int currY)
{
    QRgb c = alphaMap.pixel(currX, currY);
    if (c == 0) // clear
    {
        for (int i = (alphaMapSize - 1); i > currY; --i)
            alphaMap.setPixel(currX, i, 0x77FFFFFF);
    }
}

void TUIAlphaFree::removeAlphaInternal(int posX, int posY)
{
    QRgb c = alphaMap.pixel(posX, posY);
    if (c == 0x77FFFFFF) // occupied
    {
        for (int i = 0; i < posY; ++i)
            alphaMap.setPixel(posX, i, 0);
    }
}

void TUIAlphaFree::removeLevel(float xPos, float yPos)
{
    int posX = int(xPos * (alphaMapSize - 1));
    int posY = int(yPos * (alphaMapSize - 1));

    if (drawing)
    {
        if (lastX >= 0)
        {
            if (posX > lastX)
            {
                for (int currX = lastX + 1; currX <= posX; ++currX)
                {
                    removeAlphaInternal(currX, posY);
                }
            }
            else if (posX < lastX)
            {
                for (int currX = posX; currX < lastX; ++currX)
                {
                    removeAlphaInternal(currX, posY);
                }
            }
        }
        lastX = posX;
        lastY = posY;
    }
    else
    {
        removeAlphaInternal(posX, posY);
    }
}

int TUIAlphaFree::getAlpha(float xPos)
{
    int posX = int(xPos * (alphaMapSize - 1));

    for (int i = 0; i < (alphaMapSize - 1); ++i)
    {
        QRgb c = alphaMap.pixel(posX, i);
        if (c == 0)
            return i;
    }
    return 255;
}

void TUIAlphaFree::erase()
{
    drawing = false;
    hasData = false;
    alphaMap.fill(0);
}

void TUIAlphaFree::paint(QPainter &p, int w, bool)
{
    if (ww != w)
    {
        ww = w;
    }

    p.drawImage(QRect(0, TUITF1DEditor::panelPos, ww, TUITF1DEditor::panelSize), alphaMap,
                QRect(0, 0, alphaMap.width(), alphaMap.height()));
}

void TUIAlphaFree::toData(float *data)
{
    for (int j = 0; j < alphaMapSize; ++j)
    {
        data[j] = 0.0;
        for (int i = alphaMapSize - 1; i >= 0; --i)
        {
            QRgb c = alphaMap.pixel(j, i);
            if (c == 0)
            {
                data[j] = 1.0f - ((float)i / (alphaMapSize - 1));
                break;
            }
        }
    }
}

bool TUIAlphaFree::fromData(float *data, int count)
{
    if (count != alphaMapSize * 2)
        return false;

    for (int i = 0; i < count; i += 2)
    {
        addLevel(data[i], 1.0f - data[i + 1]);
    }
    return true;
}

//
// --------------------------------------------------------
TUIVirvoWidget::TUIVirvoWidget(TFKind myKind, QColor color, vvTransFunc *tf, float x, float y)
    : TUITFEWidget(myKind)
{
    this->tf = tf;
    vvWidget = NULL;
    vvColor col;

    col.setRGB(color.red() / 255.0f, color.green() / 255.0f, color.blue() / 255.0f);

    //tf->putUndoBuffer();
    switch (kind)
    {
    case TF_COLOR:
        vvWidget = new vvTFColor(col, x, y);
        break;

    case TF_PYRAMID:
        vvWidget = new vvTFPyramid(col, false, 1.0f, x, 0.4f, 0.2f, y);
        break;

    case TF_BELL:
        vvWidget = new vvTFBell(col, false, 1.0f, x, 0.2f, y);
        break;

    //case TF_CUSTOM:
    //   break;

    case TF_SKIP:
        vvWidget = new vvTFSkip(x, 0.2f, y);
        break;

    case TF_CUSTOM_2D_EXTRUDE:
        vvWidget = new vvTFCustom2D(true, 1.0f, x, y);
        break;

    case TF_CUSTOM_2D_TENT:
        vvWidget = new vvTFCustom2D(true, 1.0f, x, y);
        break;

    default:
        assert(false);
        break;
    }
    tf->_widgets.push_back(vvWidget);
}

TUIVirvoWidget::TUIVirvoWidget(TFKind myKind, vvTFWidget *w, vvTransFunc *tf)
    : TUITFEWidget(myKind)
{
    this->tf = tf;
    vvWidget = w;
    tf->_widgets.push_back(vvWidget);
}

TUIVirvoWidget::~TUIVirvoWidget()
{
    if (vvWidget != NULL)
        delete vvWidget;
}

void TUIVirvoWidget::setOwnColor(bool f)
{
    switch (kind)
    {
    case TF_PYRAMID:
    {
        vvTFPyramid *w = static_cast<vvTFPyramid *>(vvWidget);
        w->setOwnColor(f);
    }
    break;

    case TF_BELL:
    {
        vvTFBell *w = static_cast<vvTFBell *>(vvWidget);
        w->setOwnColor(f);
    }
    break;

    case TF_CUSTOM_2D:
    {
        vvTFCustom2D *w = static_cast<vvTFCustom2D *>(vvWidget);
        w->setOwnColor(f);
    }
    break;

    case TF_MAP:
    {
        vvTFCustomMap *w = static_cast<vvTFCustomMap *>(vvWidget);
        w->setOwnColor(f);
    }
    break;

    case TF_CUSTOM:
    case TF_CUSTOM_2D_EXTRUDE:
    case TF_CUSTOM_2D_TENT:
    case TF_SKIP:
    case TF_COLOR:
        break;
    }
}

bool TUIVirvoWidget::hasOwnColor()
{
    bool ownColor = false;
    switch (kind)
    {
    case TF_COLOR:
        ownColor = true;
        break;

    case TF_PYRAMID:
    {
        vvTFPyramid *w = static_cast<vvTFPyramid *>(vvWidget);
        ownColor = w->hasOwnColor();
    }
    break;

    case TF_BELL:
    {
        vvTFBell *w = static_cast<vvTFBell *>(vvWidget);
        ownColor = w->hasOwnColor();
    }
    break;

    case TF_CUSTOM_2D:
    {
        vvTFCustom2D *w = static_cast<vvTFCustom2D *>(vvWidget);
        ownColor = w->hasOwnColor();
    }
    break;

    case TF_MAP:
    {
        vvTFCustomMap *w = static_cast<vvTFCustomMap *>(vvWidget);
        ownColor = w->hasOwnColor();
    }
    break;

    case TF_CUSTOM:
    case TF_CUSTOM_2D_EXTRUDE:
    case TF_CUSTOM_2D_TENT:
    case TF_SKIP:
        break;
    }

    return ownColor;
}

void TUIVirvoWidget::setColor(QColor color)
{
    vvColor col;
    col.setRGB(color.red() / 255.0f,
               color.green() / 255.0f,
               color.blue() / 255.0f);

    switch (kind)
    {
    case TF_COLOR:
    {
        vvTFColor *w = static_cast<vvTFColor *>(vvWidget);
        w->_col[0] = col[0];
        w->_col[1] = col[1];
        w->_col[2] = col[2];
    }
    break;

    case TF_PYRAMID:
    {
        vvTFPyramid *w = static_cast<vvTFPyramid *>(vvWidget);
        if (w->hasOwnColor())
        {
            w->_col[0] = col[0];
            w->_col[1] = col[1];
            w->_col[2] = col[2];
        }
    }
    break;

    case TF_BELL:
    {
        vvTFBell *w = static_cast<vvTFBell *>(vvWidget);
        if (w->hasOwnColor())
        {
            w->_col[0] = col[0];
            w->_col[1] = col[1];
            w->_col[2] = col[2];
        }
    }
    break;

    case TF_CUSTOM_2D:
    {
        vvTFCustom2D *w = static_cast<vvTFCustom2D *>(vvWidget);
        if (w->hasOwnColor())
        {
            w->_col[0] = col[0];
            w->_col[1] = col[1];
            w->_col[2] = col[2];
        }
    }
    break;

    case TF_MAP:
    {
        vvTFCustomMap *w = static_cast<vvTFCustomMap *>(vvWidget);
        if (w->hasOwnColor())
        {
            w->_col[0] = col[0];
            w->_col[1] = col[1];
            w->_col[2] = col[2];
        }
    }
    break;

    case TF_CUSTOM:
    case TF_CUSTOM_2D_EXTRUDE:
    case TF_CUSTOM_2D_TENT:
    case TF_SKIP:
        break;
    }
}

void TUIVirvoWidget::setParam(int dim, int nParam, float value)
{
    switch (kind)
    {
    case TF_PYRAMID:
    {
        vvTFPyramid *w = static_cast<vvTFPyramid *>(vvWidget);
        if (nParam == 0) //bottom
        {
            //parameter is relative to widget center
            w->_bottom[dim] = value;
        }
        else if (nParam == 1) //top
        {
            w->_top[dim] = value;
        }
    }
    break;

    case TF_BELL:
    {
        vvTFBell *w = static_cast<vvTFBell *>(vvWidget);
        if (nParam == 0)
        {
            w->_size[dim] = value;
        }
    }
    break;

    default:
        break;
    }
}

float TUIVirvoWidget::getParam(int dim, int nParam)
{
    float retval = -1.0;

    switch (kind)
    {
    case TF_PYRAMID:
    {
        vvTFPyramid *w = static_cast<vvTFPyramid *>(vvWidget);
        if (nParam == 0) //bottom
        {
            retval = w->_bottom[dim];
        }
        else if (nParam == 1) //top
        {
            retval = w->_top[dim];
        }
    }
    break;

    case TF_BELL:
    {
        vvTFBell *w = static_cast<vvTFBell *>(vvWidget);
        if (nParam == 0)
        {
            retval = w->_size[dim];
        }
    }
    break;

    default:
        break;
    }

    return retval;
}

void TUIVirvoWidget::setXParam(int nParam, float value)
{
    setParam(0, nParam, value);
}

void TUIVirvoWidget::setYParam(int nParam, float value)
{
    setParam(1, nParam, value);
}

float TUIVirvoWidget::getXParam(int nParam)
{
    return getParam(0, nParam);
}

float TUIVirvoWidget::getYParam(int nParam)
{
    return getParam(1, nParam);
}

void TUIVirvoWidget::setColor(float r, float g, float b)
{
    switch (kind)
    {
    case TF_COLOR:
    {
        vvTFColor *w = static_cast<vvTFColor *>(vvWidget);
        w->_col[0] = r;
        w->_col[1] = g;
        w->_col[2] = b;
    }
    break;

    case TF_PYRAMID:
    {
        vvTFPyramid *w = static_cast<vvTFPyramid *>(vvWidget);
        if (w->hasOwnColor())
        {
            w->_col[0] = r;
            w->_col[1] = g;
            w->_col[2] = b;
        }
    }
    break;

    case TF_BELL:
    {
        vvTFBell *w = static_cast<vvTFBell *>(vvWidget);
        if (w->hasOwnColor())
        {
            w->_col[0] = r;
            w->_col[1] = g;
            w->_col[2] = b;
        }
    }
    break;

    case TF_CUSTOM_2D:
    {
        vvTFCustom2D *w = static_cast<vvTFCustom2D *>(vvWidget);
        if (w->hasOwnColor())
        {
            w->_col[0] = r;
            w->_col[1] = g;
            w->_col[2] = b;
        }
    }
    break;

    case TF_MAP:
    {
        vvTFCustomMap *w = static_cast<vvTFCustomMap *>(vvWidget);
        if (w->hasOwnColor())
        {
            w->_col[0] = r;
            w->_col[1] = g;
            w->_col[2] = b;
        }
    }
    break;

    case TF_CUSTOM:
    case TF_CUSTOM_2D_EXTRUDE:
    case TF_CUSTOM_2D_TENT:
    case TF_SKIP:
        break;
    }
}

void TUIVirvoWidget::getColorFloat(float col[3])
{
    col[0] = col[1] = col[2] = 0.8f;

    switch (kind)
    {
    case TF_COLOR:
    {
        vvTFColor *w = static_cast<vvTFColor *>(vvWidget);
        col[0] = w->_col[0];
        col[1] = w->_col[1];
        col[2] = w->_col[2];
    }
    break;

    case TF_PYRAMID:
    {
        vvTFPyramid *w = static_cast<vvTFPyramid *>(vvWidget);
        if (w->hasOwnColor())
        {
            col[0] = w->_col[0];
            col[1] = w->_col[1];
            col[2] = w->_col[2];
        }
    }
    break;

    case TF_BELL:
    {
        vvTFBell *w = static_cast<vvTFBell *>(vvWidget);
        if (w->hasOwnColor())
        {
            col[0] = w->_col[0];
            col[1] = w->_col[1];
            col[2] = w->_col[2];
        }
    }
    break;

    case TF_CUSTOM:
    case TF_CUSTOM_2D:
    case TF_CUSTOM_2D_EXTRUDE:
    case TF_CUSTOM_2D_TENT:
    case TF_MAP:
    case TF_SKIP:
        break;
    }
}

QColor TUIVirvoWidget::getColor()
{
    switch (kind)
    {
    case TF_COLOR:
    {
        vvTFColor *w = static_cast<vvTFColor *>(vvWidget);
        return QColor((int)(w->_col[0] * 255.0f), (int)(w->_col[1] * 255.0f), (int)(w->_col[2] * 255.0f));
    }

    case TF_PYRAMID:
    {
        vvTFPyramid *w = static_cast<vvTFPyramid *>(vvWidget);
        if (w->hasOwnColor())
            return QColor((int)(w->_col[0] * 255.0f), (int)(w->_col[1] * 255.0f), (int)(w->_col[2] * 255.0f));
        else
            return QColor(Qt::transparent);
    }

    case TF_BELL:
    {
        vvTFBell *w = static_cast<vvTFBell *>(vvWidget);
        if (w->hasOwnColor())
            return QColor((int)(w->_col[0] * 255.0f), (int)(w->_col[1] * 255.0f), (int)(w->_col[2] * 255.0f));
        else
            return QColor(Qt::transparent);
    }

    case TF_CUSTOM_2D:
    {
        vvTFCustom2D *w = static_cast<vvTFCustom2D *>(vvWidget);
        if (w->hasOwnColor())
            return QColor((int)(w->_col[0] * 255.0f), (int)(w->_col[1] * 255.0f), (int)(w->_col[2] * 255.0f));
        else
            return QColor(Qt::transparent);
    }

    case TF_MAP:
    {
        vvTFCustomMap *w = static_cast<vvTFCustomMap *>(vvWidget);
        if (w->hasOwnColor())
            return QColor((int)(w->_col[0] * 255.0f), (int)(w->_col[1] * 255.0f), (int)(w->_col[2] * 255.0f));
        else
            return QColor(Qt::transparent);
    }

    case TF_CUSTOM:
    case TF_CUSTOM_2D_EXTRUDE:
    case TF_CUSTOM_2D_TENT:
    case TF_SKIP:
        break;
    }
    return QColor(Qt::transparent);
}

bool TUIVirvoWidget::contains(int x, int y, int /*w*/)
{
    //ignore w, fixed width
    int xx = int(getX() * TUITF2DEditor::panelSize);
    int yy = int(getY() * TUITF2DEditor::panelSize);

    if ((abs(xx - x) < TUITFEditor::clickThreshold) && (abs(yy - y) < TUITFEditor::clickThreshold))
        return true;
    return false;
}

TUITFEWidget::HandleType TUIVirvoWidget::testHit(int x, int y)
{
    float xPos = (float)x / TUITF2DEditor::panelSize;
    float yPos = (float)y / TUITF2DEditor::panelSize;
    float ht = (float)TUITFEditor::clickThreshold / TUITF2DEditor::panelSize;

    this->selectedHandle = TUITFEWidget::HT_NONE;

    if ((fabsf(xPos - getX()) < ht) && (fabsf(yPos - getY()) < ht))
    {
        this->selectedHandle = TUITFEWidget::HT_MIDDLE;
    }
    else
    {
        if (kind == TF_PYRAMID || kind == TF_BELL)
        {
            // check for "bottom" handle selection
            if ((fabsf(xPos - (getXParam(0) / 2.0f + getX())) < ht) && (fabsf(yPos - (getYParam(0) / 2.0f + getY())) < ht))
            {
                this->selectedHandle = TUITFEWidget::HT_BOTTOM;
            }
            else
            {
                // "top" is only for triangle/pyramid
                // top widget decoration is skewed to the right, so we add ht to x
                if (kind == TF_PYRAMID)
                {
                    if ((fabsf(xPos - (getXParam(1) / 2.0f + getX() + ht)) < ht) && (fabsf(yPos - (getYParam(1) / 2.0f + getY())) < ht))
                    {
                        this->selectedHandle = TUITFEWidget::HT_TOP;
                    }
                }
            }
        }
    }

    return selectedHandle;
}
