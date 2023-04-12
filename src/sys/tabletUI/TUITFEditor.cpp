/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "TUITFEditor.h"
#include <QPainter>
#include <QFrame>

#include "TUIApplication.h"
#include "TUIFunctionEditorTab.h"
#include <net/tokenbuffer.h>
#if QT_VERSION > QT_VERSION_CHECK(6, 0, 0)
#define POSITIONX position().x()
#define POSITIONY position().y()
#else
#define POSITIONX x()
#define POSITIONY y()
#endif

static float s_stdColorMap[] = {
    0.0, 0.000000, 0.000000, 1.000000,
    0.5, 1.000000, 0.000000, 0.000000,
    1.0, 1.000000, 1.000000, 0.000000,
};

void TUITFEditor::removeCurrentMarker()
{
    removeMarker(selectedPoint);
    selectedPoint = NULL;
    //repaint();
    emit functionChanged();
}

void TUITF1DEditor::removeMarker(TUITFEWidget *w)
{
    if (w == NULL)
        return;

    if (w->getKind() == TUITFEWidget::TF_COLOR)
    {
        TUIColorPoint *cp = static_cast<TUIColorPoint *>(w);
        // check if the last or first marker was selected
        int index = colorPoints.indexOf(cp);
        if (index <= 0)
            return;

        if (index + 1 == colorPoints.size())
            return;

        colorPoints.removeAt(index);
        delete cp;
    }
    else
    {
        TUIAlphaTriangle *ap = static_cast<TUIAlphaTriangle *>(w);

        int index = alphaPoints.indexOf(ap);
        alphaPoints.removeAt(index);
        delete ap;
    }
    if (selectedPoint == w)
        selectedPoint = NULL;
    emit deletePoint(w);
}

//
// make a new color point in the manipulation area
//
TUIColorPoint *TUITF1DEditor::makeColorPoint(float pos, QColor col)
{
    TUIColorPoint *cp = new TUIColorPoint();
    //connect(cp, SIGNAL(removePoint(TUITFEWidget*)), this, SLOT(pointRemoved(TUITFEWidget*)));
    cp->setData(pos, col);

    int i;
    for (i = 0; i < colorPoints.size(); ++i)
    {
        if (colorPoints.at(i)->getX() > pos)
            break;
    }

    colorPoints.insert(i, cp);

    addMarker(cp);
    return cp;
}

//
// make a new point in the manipulation area
//
TUIAlphaTriangle *TUITF1DEditor::makeAlphaPoint(float x, int a, float xb, float xt)
{
    TUIAlphaTriangle *ap = new TUIAlphaTriangle();
    //connect(ap, SIGNAL(removePoint(TUITFEWidget*)), this, SLOT(pointRemoved(TUITFEWidget*)));
    ap->setData(x, a, xb, xt);

    int i;
    for (i = 0; i < alphaPoints.size(); ++i)
    {
        if (alphaPoints.at(i)->getX() > x)
            break;
    }
    alphaPoints.insert(i, ap);

    addMarker(ap);
    return ap;
}

void TUITF1DEditor::parseMessage(covise::TokenBuffer &tb)
{
    uint32_t listLength;

    // first receive colors
    tb >> listLength;
    colorPoints.clear();

    if (listLength > 0)
    {
        for (uint32_t i = 0; i < listLength * 4; i += 4)
        {
            // for each entry: r, g, b channels (float), pos (float)
            // but the updateColorMap function expects rgbax, so lets
            // add an opaque alpha component. We deal with alpha below
            float r, g, b, x;
            tb >> r;
            tb >> g;
            tb >> b;
            tb >> x;
            makeColorPoint(x, QColor(int(255.0f * r), int(255.0f * g), int(255.0f * b), 255));
        }
    }

    // then, the alpha widgets
    tb >> listLength;

    if (listLength > 0)
    {
        for (uint32_t i = 0; i < listLength; ++i)
        {
            uint32_t widgetType;
            //TF_PYRAMID == 1, TF_CUSTOM == 4
            //
            tb >> widgetType;
            switch (widgetType)
            {
            case TUITFEWidget::TF_PYRAMID:
            {
                float alpha;
                float x, xb, xt;
                tb >> alpha;
                tb >> x;
                tb >> xb;
                tb >> xt;
                makeAlphaPoint(x, int(alpha * 255), xb, xt);
            }
            break;

            case TUITFEWidget::TF_CUSTOM:
            {
                uint32_t entryNum;
                tb >> entryNum;
                float *values = new float[entryNum * 2];
                for (uint32_t i = 0; i < entryNum * 2; i += 2)
                {
                    tb >> values[i]; //pos
                    tb >> values[i + 1]; //alpha value;
                }
                // entries are guaranteed to be in order!
                this->alphaFreeForm.fromData(values, entryNum * 2);
                delete[] values;
            }
            break;

            default:
                assert(false);
                //we should never receive other widget types;
                break;
            }
        }
    }
}

void TUITF1DEditor::valueChanged(covise::TokenBuffer &tb)
{
    // first, send color points
    tb << (int)this->colorPoints.count();
    for (int i = 0; i < this->colorPoints.count(); ++i)
    {
        tb << (float)this->colorPoints.at(i)->getColor().red() / 255.0f;
        tb << (float)this->colorPoints.at(i)->getColor().green() / 255.0f;
        tb << (float)this->colorPoints.at(i)->getColor().blue() / 255.0f;
        tb << this->colorPoints.at(i)->getX();
    }

    // then, alpha widgets
    int numAlphaPoints = this->alphaPoints.count();
    if (this->alphaFreeForm.isNotEmpty())
        ++numAlphaPoints;

    tb << numAlphaPoints;
    for (int i = 0; i < this->alphaPoints.count(); ++i)
    {
        tb << TUITFEWidget::TF_PYRAMID; //1
        tb << (float)(this->alphaPoints.at(i)->getAlpha()) / 255.0f;
        tb << this->alphaPoints.at(i)->getX();
        tb << this->alphaPoints.at(i)->getXb();
        tb << this->alphaPoints.at(i)->getXt();
    }

    if (this->alphaFreeForm.isNotEmpty())
    {
        // send also the alpha map designed by free hand
        tb << TUITFEWidget::TF_CUSTOM; //4

        float data[TUIAlphaFree::alphaMapSize];
        this->alphaFreeForm.toData(data);

        tb << TUIAlphaFree::alphaMapSize; //data lenght;

        for (int i = 0; i < TUIAlphaFree::alphaMapSize; ++i)
        {
            float pos = (float)i / (TUIAlphaFree::alphaMapSize - 1);
            tb << pos;
            tb << data[i];
        }
    }
}

void TUITF1DEditor::addMarker(TUITFEWidget *)
{
    this->repaint();
}

/*!
   \class *
   \brief This class provides a widget for displaying the actual colortable
*/

/*****************************************************************************
 *
 * Class TUIColorRGBTable (Color table)
 *
 *****************************************************************************/

TUITF1DEditor::TUITF1DEditor(TUIFunctionEditorTab *c, QWidget *p)
    : TUITFEditor(c, p)
{
    // two marker containers plus the colormap panel
    setFixedHeight(2 * (markerSize + 4) + panelSize);
    drawFree = false;
    rightButtonDown = false;
}

TUITF1DEditor::~TUITF1DEditor()
{
    if (colorPoints.size() != 0)
    {
        while (!colorPoints.isEmpty())
            delete colorPoints.takeFirst();
        colorPoints.clear();
    }

    if (alphaPoints.size() != 0)
    {
        while (!alphaPoints.isEmpty())
            delete alphaPoints.takeFirst();
        alphaPoints.clear();
    }
}

void TUITF1DEditor::eraseAlphaFree()
{
    alphaFreeForm.erase();
    repaint();
    emit functionChanged();
}

void TUITF1DEditor::paintEvent(QPaintEvent *e)
{
    QWidget::paintEvent(e);

    QPainter p(this);

    //!
    //! 1) draw the color table
    //!
    if (!parentControl || !parentControl->initFlag)
    {
        fprintf(stderr, "!init\n");
        p.fillRect(0, 0, width(), height(), QBrush(Qt::blue, Qt::SolidPattern));
        return;
    }

    // loop over all interpolation markers
    // first, the color ones
    for (int i = 0; i < colorPoints.size() - 1; i++)
    {
        // set a gradientpoint
        TUIColorPoint *left = colorPoints.at(i);
        TUIColorPoint *right = colorPoints.at(i + 1);

        int xleft = int(left->getX() * width());
        int xright = int(right->getX() * width());

        QLinearGradient lgrad(xleft, 0., xright, 0.);

        QColor c1(left->getColor());
        lgrad.setColorAt(0., c1);

        QColor c2(right->getColor());
        lgrad.setColorAt(1., c2);

        p.fillRect(xleft, panelPos, (xright - xleft + 1), panelSize, QBrush(lgrad));
    }

    //!
    //! 2) then, superimpose alpha free-form
    //!
    alphaFreeForm.paint(p, width(), false);

    //!
    //! 3) draw the color and alpha widgets
    //!

    for (int i = 0; i < colorPoints.size(); i++)
    {
        TUITFEWidget *cp = colorPoints.at(i);
        //int x = int(cp->getX()* width()) - markerSize/2;
        cp->paint(p, width(), (selectedPoint == cp));
    }

    for (int i = 0; i < alphaPoints.size(); i++)
    {
        TUITFEWidget *cp = alphaPoints.at(i);
        //int x = int(cp->getX()* width()) - markerSize/2;
        //cp->move(x, alphaWidgetPos);
        cp->paint(p, width(), (selectedPoint == cp));
    }
}

void TUITF1DEditor::clearAlphaDrawFlag()
{
    drawFree = false;
    alphaFreeForm.endDraw();
    emit functionChanged();
}

void TUITF1DEditor::setAlphaDrawFlag()
{
    drawFree = true;
    alphaFreeForm.beginDraw();
}

TUITFEWidget *TUITF1DEditor::newAlphaPoint(float newX)
{
    return makeAlphaPoint(newX, 255, 0.5f, 0.1f);
}

TUITFEWidget *TUITF1DEditor::newColorPoint(float newX)
{
    // load standard map if there are no points defined
    //if(colorPoints.count() < 2)
    //{
    //   loadDefault();
    //}

    TUIColorPoint *cpnew = NULL;
    TUIColorPoint *left = NULL;
    TUIColorPoint *right = NULL;

    float xleft = 0.f;
    float xright = 0.f;

    if (colorPoints.first())
    {
        left = colorPoints.first();
        xleft = left->getX();
    }

    // find the neighbours for this new index
    int count = 0;
    QColor col(Qt::blue);

    for (int i = 1; i < colorPoints.size(); i++)
    {
        count++;
        right = colorPoints.at(i);
        xright = right->getX();
        if (newX > xleft && newX < xright)
        {
            // calculate color & alpha
            float diff = (xright - xleft);

            QColor rc = right->getColor();
            QColor lc = left->getColor();
            float rdiff = float(rc.red() - lc.red()) / diff;
            float gdiff = float(rc.green() - lc.green()) / diff;
            float bdiff = float(rc.blue() - lc.blue()) / diff;
            //float adiff = float(right->getAlpha()     - left->getAlpha())    / diff;

            int r = lc.red() + int(rdiff * (newX - xleft));
            int g = lc.green() + int(gdiff * (newX - xleft));
            int b = lc.blue() + int(bdiff * (newX - xleft));
            //int a = left->getAlpha()    + int( adiff * (newX-xleft) );

            col = QColor(r, g, b);
            break;
        }
        xleft = xright;
        left = right;
    }

    // make new marker
    cpnew = makeColorPoint(newX, col);
    return cpnew;
}

//!
//! user has selected a new current value
//!
void TUITF1DEditor::newWidgetValue(float newval)
{
    if (selectedPoint == NULL)
        return;

    switch (selectedPoint->getKind())
    {
    case TUITFEWidget::TF_COLOR:
    {
        TUIColorPoint *cp = static_cast<TUIColorPoint *>(selectedPoint);

        // check if value is possible
        // look for neighbours

        TUIColorPoint *left;
        TUIColorPoint *right;
        int index = colorPoints.indexOf(cp);
        if (index == -1)
            return;

        if (index == 0)
            left = cp;
        else
            left = colorPoints.at(index - 1);

        if (index + 1 == colorPoints.size())
            right = cp;
        else
            right = colorPoints.at(index + 1);

        // set new current value
        float currValue = qMin(newval, right->getX());
        currValue = qMax(currValue, left->getX());

        // move point
        cp->setX(currValue);
        repaint();
        emit functionChanged();
    }
    break;

    case TUITFEWidget::TF_PYRAMID:
    {
        // move point
        selectedPoint->setX(newval);
        repaint();
        emit functionChanged();
    }
    break;

    default:
        // other cases: ignore them for now
        break;
    }
}

void TUITF1DEditor::loadDefault()
{
    // load standard map if there are no points defined
    if (colorPoints.count() < 2)
    {
        for (int i = 0; i < sizeof(s_stdColorMap) / (sizeof(float)); i += 4)
            makeColorPoint(s_stdColorMap[i], QColor(int(255.0f * s_stdColorMap[i + 1]), int(255.0f * s_stdColorMap[i + 2]), int(255.0f * s_stdColorMap[i + 3])));
    }
    // makeAlphaPoint(1.0f, 255, 2.0f, 0.0f);
}

void TUITF1DEditor::mousePressEvent(QMouseEvent *e)
{
    if (e->button() == Qt::RightButton)
    {
        rightButtonDown = true;
        // for now, right button always used for drawing alpha
        setAlphaDrawFlag();
    }

    if (e->button() == Qt::LeftButton)
    {
        // first, see where the mouse clicked (which region was selected)
        bool alphaRegion = false;
        bool colorRegion = false;

        QVector<TUITFEWidget *> list;
        float xpos = (float)(e->POSITIONX) / (float)width();

        // alpha point?
        int testPoint = ::abs(e->POSITIONY - alphaWidgetPos - halfMarkerSize);
        if (testPoint < clickThreshold)
        {
            // look if a alpha marker was pressed
            for (int i = 0; i < alphaPoints.size(); i++)
            {
                TUITFEWidget *wp = alphaPoints.at(i);
                if (wp->contains(e->POSITIONX, e->POSITIONY, width()))
                    list.append(wp);
            }
            alphaRegion = true;
        }
        // color point?
        else
        {
            testPoint = ::abs(e->POSITIONY - colorWidgetPos - halfMarkerSize);
            if (testPoint < clickThreshold)
            {

                // look if a color marker was pressed
                for (int i = 0; i < colorPoints.size(); i++)
                {
                    TUITFEWidget *wp = colorPoints.at(i);
                    if (wp->contains(e->POSITIONX, e->POSITIONY, width()))
                        list.append(wp);
                }
                colorRegion = true;
            }
        }

        // no marker found
        // can be a new point OR can be an handle
        if (list.isEmpty())
        {
            if (alphaRegion)
            {
                if (selectedPoint != NULL)
                {
                    selectedPoint = NULL; // first click just deselects
                    emit pickPoint(selectedPoint);
                }
                else
                {
                    TUITFEWidget *w = this->newAlphaPoint(xpos);
                    selectedPoint = w;
                    emit newPoint(w);
                    emit functionChanged();
                }
            }
            else if (colorRegion)
            {
                if (selectedPoint != NULL)
                {
                    selectedPoint = NULL;
                    emit pickPoint(selectedPoint);
                }
                else
                {
                    TUITFEWidget *w = this->newColorPoint(xpos);
                    selectedPoint = w;
                    emit newPoint(w);
                    emit functionChanged();
                }
            }
            else //color map area
            {
                if (selectedPoint != NULL)
                {
                    TUITFEWidget::HandleType ht = selectedPoint->testHit(e->POSITIONX, e->POSITIONY);
                    if (ht == TUITFEWidget::HT_NONE)
                    {
                        // no widget, unselect
                        selectedPoint = NULL;
                        emit pickPoint(selectedPoint);
                        setAlphaDrawFlag();
                    }
                }
                else
                {
                    //nothing selected, we can draw
                    setAlphaDrawFlag();
                }
            }
        }
        // found exactly one point --> set this point active
        else if (list.count() == 1)
        {
            selectedPoint = list.at(0);
            selectedPoint->setActivated();
            emit pickPoint(selectedPoint);
        }
        // found more points --> ignore first and last color point (not movable)
        else
        {
            if (list.contains(colorPoints.first()))
                selectedPoint = list.at(1);
            else if (list.contains(colorPoints.last()))
                selectedPoint = list.at(0);
            else
                selectedPoint = list.last();

            selectedPoint->setActivated();
            emit pickPoint(selectedPoint);
        }
    }
}

void TUITF1DEditor::mouseReleaseEvent(QMouseEvent *)
{
    rightButtonDown = false;
    clearAlphaDrawFlag();
    //if(selectedPoint)
    //   emit updatePoint(selectedPoint);
}

void TUITF1DEditor::mouseMoveEvent(QMouseEvent *e)
{
    if (drawFree)
    {
        float xPos = (float)e->POSITIONX / width();
        float yPos = ((e->POSITIONY - TUITF1DEditor::panelPos)) / (float)TUITF1DEditor::panelSize;
        if (yPos > 1.0f)
            yPos = 1.0f;
        if (yPos < 0.0f)
            yPos = 0.0f;

        if (rightButtonDown)
            alphaFreeForm.removeLevel(xPos, yPos);
        else
            alphaFreeForm.addLevel(xPos, yPos);

        this->repaint();
    }
    else
    {
        if (!selectedPoint)
            return;

        switch (selectedPoint->getKind())
        {
        case TUITFEWidget::TF_COLOR:
        {
            int index = colorPoints.indexOf(static_cast<TUIColorPoint *>(selectedPoint));

            // first point cannot be moved
            if (index <= 0)
                return;

            // last point cannot be moved
            if (index + 1 == colorPoints.size())
                return;

            // get neighbours
            float xx = float(e->POSITIONX) / float(width());
            float xmin = (colorPoints.at(index - 1))->getX();
            float xmax = (colorPoints.at(index + 1))->getX();

            // you cannot move points over the neighbour points
            if (xx <= xmin || xx >= xmax)
                return;

            //selectedPoint->move(e->x() - offset, colorWidgetPos);
            selectedPoint->setX(xx);
            //selectedPoint->update();
            //this->repaint();
            emit movePoint(selectedPoint);
            emit functionChanged();
        }
        break;

        case TUITFEWidget::TF_PYRAMID:
        {
            switch (selectedPoint->getSelectedHandle())
            {
            case TUITFEWidget::HT_NONE:
            {
                // get neighbours
                float xx = float(e->POSITIONX) / float(width());
                float xmin = 0.0f;
                float xmax = 1.0f;

                // you cannot move points over the edge
                if (xx <= xmin || xx >= xmax)
                    return;

                selectedPoint->setX(xx);

                emit movePoint(selectedPoint);
                emit functionChanged();
            }
            break;

            case TUITFEWidget::HT_BOTTOM:
            {
                TUIAlphaTriangle *ap = static_cast<TUIAlphaTriangle *>(selectedPoint);
                float xb = (ap->getX() - float(e->POSITIONX) / float(width())) * 2.0f;
                ap->setXb(xb);
                this->repaint();
            }
            break;

            case TUITFEWidget::HT_TOP:
            {
                TUIAlphaTriangle *ap = static_cast<TUIAlphaTriangle *>(selectedPoint);
                float xt = ((float(e->POSITIONX) / float(width())) - ap->getX()) * 2.0f;
                ap->setXt(xt);
                this->repaint();
            }
            break;

            case TUITFEWidget::HT_MIDDLE:
            {
                TUIAlphaTriangle *ap = static_cast<TUIAlphaTriangle *>(selectedPoint);
                int alpha = 255 - (((e->POSITIONY - TUITF1DEditor::panelPos) * 255) / TUITF1DEditor::panelSize);
                ap->setAlpha(alpha);
                this->repaint();
            }
            break;
            }
        }
        break;

        default:
            //we don't care about other cases
            break;
        }
    }
}
