/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <net/tokenbuffer.h>
#include "TUITF2DEditor.h"
#include "TUIFunctionEditorTab.h"

#include <qmessagebox.h>
#include <math.h>
#if QT_VERSION > QT_VERSION_CHECK(6, 0, 0)
#define POSITIONX position().x()
#define POSITIONY position().y()
#else
#define POSITIONX x()
#define POSITIONY y()
#endif

using std::list;

Canvas::Canvas(TUITF2DEditor *parent)
    : QWidget(parent)
{
    setFixedSize(TUITF2DEditor::panelSize, TUITF2DEditor::panelSize);
    this->editor = parent;
    this->tfTexture = new uchar[TUITF2DEditor::textureSize * TUITF2DEditor::textureSize * 4];

    textureDirty = true;
    textureImage = NULL;

    currentBackType = BackBlack;

    // to allow overpainting of the editor alpha map image
    // and of the contour point list
    setAutoFillBackground(false);
}

Canvas::~Canvas()
{
    delete[] tfTexture;
}

void Canvas::drawBackGroundHistogram(QPainter &painter)
{
    QBrush clearBrush(Qt::gray);

    int histIndex;
    int index[2];

    painter.fillRect(0, 0, TUITF2DEditor::panelSize, TUITF2DEditor::panelSize, clearBrush);

    for (int y = 0; y < TUITF2DEditor::panelSize; ++y)
    {
        for (int x = 0; x < TUITF2DEditor::panelSize; ++x)
        {
            index[0] = int(float(x) / float(TUITF2DEditor::panelSize) * float(editor->histoBuckets[0]));
            index[1] = int(float(y) / float(TUITF2DEditor::panelSize) * float(editor->histoBuckets[1]));

            histIndex = index[0] + index[1] * editor->histoBuckets[0];
            int c = 255 - editor->histoData[histIndex];
            painter.setPen(QColor(c, c, c));
            painter.drawPoint(x, TUITF2DEditor::panelSize - y - 1);
        }
    }
}

void Canvas::drawBackGroundCheckers(QPainter &painter)
{
    QBrush blackBrush(Qt::black);
    QBrush whiteBrush(Qt::white);

    bool even;
    int steps = TUITF2DEditor::panelSize / TUITF2DEditor::tileSize;
    for (int j = 0; j < steps; ++j)
    {
        even = (j % 2 == 0);
        for (int i = 0; i < steps; ++i)
        {
            if (even)
                painter.fillRect(i * TUITF2DEditor::tileSize, j * TUITF2DEditor::tileSize, TUITF2DEditor::tileSize, TUITF2DEditor::tileSize, blackBrush);
            else
                painter.fillRect(i * TUITF2DEditor::tileSize, j * TUITF2DEditor::tileSize, TUITF2DEditor::tileSize, TUITF2DEditor::tileSize, whiteBrush);
            even = !even;
        }
    }
}

void Canvas::paintEvent(QPaintEvent * /*e*/)
{
    QPainter painter;
    painter.begin(this);

    BackType backType;

    // select the backgound type to render
    if (currentBackType == BackHistogram && this->editor->histoData == NULL)
        backType = BackBlack;
    else
        backType = currentBackType;

    switch (backType)
    {
    case BackChecker:
        drawBackGroundCheckers(painter);
        break;

    case BackHistogram:
        drawBackGroundHistogram(painter);
        break;

    case BackBlack:
        painter.fillRect(0, 0, TUITF2DEditor::panelSize, TUITF2DEditor::panelSize, QBrush(Qt::black));
        break;
    }

    draw2DTFTexture(painter);
    draw2DTFWidgets(painter);

    // now superimpose the 2D drawing
    if (editor->imgValid)
    {
        painter.drawImage(0, 0, editor->image);
    }

    if (editor->points.size() > 1)
    {
        painter.setPen(Qt::blue);

        list<vvTFPoint *>::iterator prev_it = editor->points.begin();
        list<vvTFPoint *>::iterator next_it = ++prev_it;
        for (int i = 1; i < editor->points.size(); ++i)
        {
            int x1 = int((**prev_it)._pos[0] * TUITF2DEditor::panelSize);
            // remember that OpenGL coordinates are cartesian,
            // while Qt coordinates are typical of window systems (flipped on y)
            int y1 = TUITF2DEditor::panelSize - int((**prev_it)._pos[1] * TUITF2DEditor::panelSize);

            int x2 = int((**next_it)._pos[0] * TUITF2DEditor::panelSize);
            int y2 = TUITF2DEditor::panelSize - int((**next_it)._pos[1] * TUITF2DEditor::panelSize);

            painter.drawLine(x1, y1, x2, y2);

            prev_it = next_it;
            ++next_it;
        }

        //connect also the firts and the last one
        next_it = prev_it;
        prev_it = editor->points.begin();

        int x1 = int((**prev_it)._pos[0] * TUITF2DEditor::panelSize);
        // remember that OpenGL coordinates are cartesian,
        // while Qt coordinates are typical of window systems (flipped on y)
        int y1 = TUITF2DEditor::panelSize - int((**prev_it)._pos[1] * TUITF2DEditor::panelSize);

        int x2 = int((**next_it)._pos[0] * TUITF2DEditor::panelSize);
        int y2 = TUITF2DEditor::panelSize - int((**next_it)._pos[1] * TUITF2DEditor::panelSize);

        painter.drawLine(x1, y1, x2, y2);
    }

    painter.end();
}

void Canvas::draw2DTFTexture(QPainter &painter)
{
    if (textureDirty)
    {
        editor->tf.make2DTFTexture2(TUITF2DEditor::textureSize, TUITF2DEditor::textureSize, tfTexture, 0.0f, 1.0f, 0.0f, 1.0f);

        //convert it to Qt Image
        if (textureImage != NULL)
            delete textureImage;
        textureImage = new QImage(tfTexture, TUITF2DEditor::textureSize, TUITF2DEditor::textureSize, QImage::Format_ARGB32);

        textureDirty = false;
    }
    painter.drawImage(this->rect(), *textureImage);
}

void Canvas::draw2DTFWidgets(QPainter &painter)
{
    std::list<TUIVirvoWidget *>::iterator it;
    for (it = editor->widgets.begin(); it != editor->widgets.end(); ++it)
    {
        draw2DWidget(painter, *it);
    }
}

void Canvas::draw2DWidget(QPainter &painter, TUIVirvoWidget *w)
{
    float xHalf = 0.0f, yHalf = 0.0f; // half width and height
    float xTop = -1.0f, yTop = -1.0f;
    bool selected = false;
    bool hasTop = false;
    bool hasBottom = false;

    float hm = (float)TUITF2DEditor::halfMarkerSize / (float)TUITF2DEditor::panelSize;

    if (w->getKind() == TUITFEWidget::TF_PYRAMID)
    {
        vvTFPyramid *pw = static_cast<vvTFPyramid *>(w->vvWidget);

        hasBottom = true;
        xHalf = pw->_bottom[0] / 2.0f;
        yHalf = pw->_bottom[1] / 2.0f;
        hasTop = true;
        xTop = pw->_top[0] / 2.0f;
        yTop = pw->_top[1] / 2.0f;
    }
    else if (w->kind == TUITFEWidget::TF_BELL)
    {
        vvTFBell *bw = static_cast<vvTFBell *>(w->vvWidget);

        hasBottom = true;
        xHalf = bw->_size[0] / 2.0f;
        yHalf = bw->_size[1] / 2.0f;
    }

    TUIVirvoWidget *currentMarker = static_cast<TUIVirvoWidget *>(editor->getSelectedMarker());
    if (currentMarker != NULL)
        selected = (w == currentMarker);

    QPen myPen(QColor(0, 0, 255));
    painter.setPen(myPen);
    if (selected)
    {
        //points[0] = QPoint(int((w->vvWidget->_pos[0] - xHalf) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yHalf) * TUITF2DEditor::panelSize)); // bottom left
        //points[1] = QPoint(int((w->vvWidget->_pos[0] + xHalf) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yHalf) * TUITF2DEditor::panelSize));   // bottom right
        //points[2] = QPoint(int((w->vvWidget->_pos[0] + xHalf) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] + yHalf) * TUITF2DEditor::panelSize));   // top right
        //points[3] = QPoint(int((w->vvWidget->_pos[0] - xHalf) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] + yHalf) * TUITF2DEditor::panelSize));   // top left

        int x = int((w->vvWidget->_pos[0] - xHalf) * TUITF2DEditor::panelSize);
        int y = int((1.0f - w->vvWidget->_pos[1] - yHalf) * TUITF2DEditor::panelSize);
        painter.setBrush(QBrush(Qt::transparent));
        painter.drawRect(x, y, int(xHalf * 2 * TUITF2DEditor::panelSize), int(yHalf * 2 * TUITF2DEditor::panelSize));

        myPen.setWidth(3);
        painter.setPen(myPen);
        painter.setBrush(QBrush(w->getColor()));

        QPoint points[4];
        points[0] = QPoint(int((w->vvWidget->_pos[0] + hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1]) * TUITF2DEditor::panelSize));
        points[1] = QPoint(int((w->vvWidget->_pos[0]) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] + hm) * TUITF2DEditor::panelSize));
        points[2] = QPoint(int((w->vvWidget->_pos[0] - hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1]) * TUITF2DEditor::panelSize));
        points[3] = QPoint(int((w->vvWidget->_pos[0]) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - hm) * TUITF2DEditor::panelSize));
        painter.drawPolygon(points, 4);

        myPen.setWidth(1);
        painter.setPen(myPen);
        if (hasBottom)
        {
            //bottom
            points[0] = QPoint(int((w->vvWidget->_pos[0] + xHalf + hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yHalf) * TUITF2DEditor::panelSize));
            points[1] = QPoint(int((w->vvWidget->_pos[0] + xHalf) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yHalf + hm) * TUITF2DEditor::panelSize));
            points[2] = QPoint(int((w->vvWidget->_pos[0] + xHalf - hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yHalf) * TUITF2DEditor::panelSize));
            points[3] = QPoint(int((w->vvWidget->_pos[0] + xHalf) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yHalf - hm) * TUITF2DEditor::panelSize));
            painter.drawPolygon(points, 4);
        }

        if (hasTop)
        {
            //top
            points[0] = QPoint(int((w->vvWidget->_pos[0] + xTop + 2 * hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yTop) * TUITF2DEditor::panelSize));
            points[1] = QPoint(int((w->vvWidget->_pos[0] + xTop + 2 * hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yTop + hm) * TUITF2DEditor::panelSize));
            points[2] = QPoint(int((w->vvWidget->_pos[0] + xTop) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yTop) * TUITF2DEditor::panelSize));
            points[3] = QPoint(int((w->vvWidget->_pos[0] + xTop + 2 * hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - yTop - hm) * TUITF2DEditor::panelSize));
            painter.drawPolygon(points, 4);
        }
    }
    else
    {
        myPen.setWidth(1);
        painter.setPen(myPen);
        painter.setBrush(QBrush(w->getColor()));

        QPoint points[3];
        points[0] = QPoint(int((w->vvWidget->_pos[0]) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] + hm) * TUITF2DEditor::panelSize));
        points[1] = QPoint(int((w->vvWidget->_pos[0] - hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - hm) * TUITF2DEditor::panelSize));
        points[2] = QPoint(int((w->vvWidget->_pos[0] + hm) * TUITF2DEditor::panelSize), int((1.0f - w->vvWidget->_pos[1] - hm) * TUITF2DEditor::panelSize));
        painter.drawPolygon(points, 3);
    }
}

//
// 2D Editor implementation
//

TUITF2DEditor::TUITF2DEditor(TUIFunctionEditorTab *c, QWidget *parent)
    : TUITFEditor(c, parent)
    , image(textureSize, textureSize, QImage::Format_ARGB32)
{
    this->setFixedHeight(panelSize);
    this->setMinimumWidth(panelSize);
    this->canvas2D = new Canvas(this);

    //keep it consistent with btnAddColor->setChecked in TUIFunctionEditorTab::make2DEditor
    currentWidgetType = TUITFEWidget::TF_COLOR;
    imgValid = false;

    currentAlpha = 128;
    currentBrushWidth = 4;

    deviceDown = false;
    leftButtonDown = false;
    rightButtonDown = false;

    insertionMode = WidgetFixed;
    alphaChannelType = AlphaPressure;
    lineWidthType = NoLineWidth;

    histoData = NULL;
    histoBuckets[0] = 0;
    histoBuckets[1] = 0;
}

TUITF2DEditor::~TUITF2DEditor()
{
    delete canvas2D;
}

void TUITF2DEditor::setCurrentWidgetType(TUITFEWidget::TFKind wt)
{
    this->currentWidgetType = wt;
    switch (wt)
    {
    case TUITFEWidget::TF_CUSTOM_2D:
        insertionMode = WidgetDrawPoints;
        break;

    case TUITFEWidget::TF_MAP:
        insertionMode = WidgetDrawMap;
        break;

    default:
        insertionMode = WidgetFixed;
        break;
    }
}

TUIVirvoWidget *TUITF2DEditor::make2DWidget(float xPos, float yPos, TUITFEWidget::TFKind kind)
{
    TUIVirvoWidget *widget = new TUIVirvoWidget(kind, parentControl->getCurrentColor(), &tf, xPos, yPos);
    if (kind == TUITFEWidget::TF_CUSTOM_2D_EXTRUDE || kind == TUITFEWidget::TF_CUSTOM_2D_TENT)
    {
        // copy point data to it
        vvTFCustom2D *w2 = static_cast<vvTFCustom2D *>(widget->vvWidget);
        list<vvTFPoint *>::iterator it;
        for (it = points.begin(); it != points.end(); ++it)
            w2->addPoint(*it);
    }
    widgets.push_back(widget);
    return widget;
}

void TUITF2DEditor::parseMessage(covise::TokenBuffer &tb)
{
    uint32_t listLength;

    // first receive colors
    tb >> listLength;
    widgets.clear();

    if (listLength > 0)
    {
        for (uint32_t i = 0; i < listLength * 4; i += 4)
        {
            // for each entry: r, g, b channels (float), pos (float)
            // but the updateColorMap function expects rgbax, so lets
            // add an opaque alpha component. We deal with alpha below
            float r, g, b, x, y;
            tb >> r;
            tb >> g;
            tb >> b;
            tb >> x;
            tb >> y;
            TUIVirvoWidget *w = make2DWidget(x, y, TUITFEWidget::TF_COLOR);
            w->setColor(r, g, b);
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
                float y, yb, yt;
                tb >> alpha;
                tb >> x;
                tb >> xb;
                tb >> xt;
                tb >> y;
                tb >> yb;
                tb >> yt;

                TUIVirvoWidget *w = make2DWidget(x, y, TUITFEWidget::TF_PYRAMID);

                int ownColor;
                tb >> ownColor;
                if (ownColor)
                {
                    w->setOwnColor(true);
                    float r, g, b;
                    tb >> r;
                    tb >> g;
                    tb >> b;
                    w->setColor(r, g, b);
                }

                w->setAlpha(alpha);
                w->setXParam(0, xb);
                w->setXParam(1, xt);
                w->setYParam(0, yb);
                w->setYParam(1, yt);
            }
            break;

            case TUITFEWidget::TF_BELL:
            {
                float alpha;
                float x, xb;
                float y, yb;
                tb >> alpha;
                tb >> x;
                tb >> xb;
                tb >> y;
                tb >> yb;

                TUIVirvoWidget *w = make2DWidget(x, y, TUITFEWidget::TF_BELL);

                int ownColor;
                tb >> ownColor;
                if (ownColor)
                {
                    w->setOwnColor(true);
                    float r, g, b;
                    tb >> r;
                    tb >> g;
                    tb >> b;
                    w->setColor(r, g, b);
                }

                w->setAlpha(alpha);
                w->setXParam(0, xb);
                w->setYParam(0, yb);
            }
            break;

            case TUITFEWidget::TF_CUSTOM_2D:
            {
                float alpha1, alpha2;
                float x, y;
                int extrude;

                tb >> alpha1;
                tb >> alpha2;
                tb >> x;
                tb >> y;
                tb >> extrude;

                vvTFCustom2D *w = new vvTFCustom2D((bool)extrude, alpha1, x, y);
                TUIVirvoWidget *widget = new TUIVirvoWidget(TUITFEWidget::TF_CUSTOM_2D, w, &tf);
                widgets.push_back(widget);

                int ownColor;
                tb >> ownColor;
                if (ownColor)
                {
                    widget->setOwnColor(true);
                    float r, g, b;
                    tb >> r;
                    tb >> g;
                    tb >> b;
                    widget->setColor(r, g, b);
                }

                uint32_t pointNum;
                tb >> pointNum;
                for (unsigned int i = 0; i < pointNum; ++i)
                {
                    tb >> alpha2;
                    tb >> x;
                    tb >> y;
                    w->addPoint(alpha2, x, y);
                }
            }
            break;

            case TUITFEWidget::TF_MAP:
            {
                float alpha;
                float x, xb;
                float y, yb;
                tb >> alpha;
                tb >> x;
                tb >> xb;
                tb >> y;
                tb >> yb;

                vvTFCustomMap *w = new vvTFCustomMap(x, xb, y, yb);

                TUIVirvoWidget *widget = new TUIVirvoWidget(TUITFEWidget::TF_MAP, w, &tf);
                widgets.push_back(widget);

                int ownColor;
                tb >> ownColor;
                if (ownColor)
                {
                    widget->setOwnColor(true);
                    float r, g, b;
                    tb >> r;
                    tb >> g;
                    tb >> b;
                    widget->setColor(r, g, b);
                }

                // store map info
                int mapDim;
                tb >> mapDim;
                for (int i = 0; i < mapDim; ++i)
                    tb >> w->_map[i];
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

void TUITF2DEditor::valueChanged(covise::TokenBuffer &tb)
{
    // first, send color points
    uint32_t numColors = tf.getNumWidgets(vvTFWidget::TF_COLOR);
    tb << numColors;

    list<TUIVirvoWidget *>::iterator it;

    for (it = widgets.begin(); it != widgets.end(); ++it)
    {
        if ((**it).getKind() == TUITFEWidget::TF_COLOR)
        {
            vvTFColor *w = static_cast<vvTFColor *>((**it).vvWidget);
            tb << w->_col[0];
            tb << w->_col[1];
            tb << w->_col[2];
            tb << w->_pos[0];
            tb << w->_pos[1];
        }
    }

    // then, alpha widgets
    tb << (uint32_t)(widgets.size() - numColors);

    for (it = widgets.begin(); it != widgets.end(); ++it)
    {
        switch ((**it).getKind())
        {
        case TUITFEWidget::TF_PYRAMID:
        {
            vvTFPyramid *w = static_cast<vvTFPyramid *>((**it).vvWidget);
            tb << TUITFEWidget::TF_PYRAMID; //1
            tb << w->_opacity;
            tb << w->_pos[0];
            tb << w->_bottom[0];
            tb << w->_top[0];
            tb << w->_pos[1];
            tb << w->_bottom[1];
            tb << w->_top[1];

            tb << (int)w->hasOwnColor();
            if (w->hasOwnColor())
            {
                tb << w->_col[0];
                tb << w->_col[1];
                tb << w->_col[2];
            }
        }
        break;

        case TUITFEWidget::TF_BELL:
        {
            vvTFBell *w = static_cast<vvTFBell *>((**it).vvWidget);
            tb << TUITFEWidget::TF_BELL; //2
            tb << w->_opacity;
            tb << w->_pos[0];
            tb << w->_size[0];
            tb << w->_pos[1];
            tb << w->_size[1];

            tb << (int)w->hasOwnColor();
            if (w->hasOwnColor())
            {
                tb << w->_col[0];
                tb << w->_col[1];
                tb << w->_col[2];
            }
        }
        break;

        case TUITFEWidget::TF_CUSTOM_2D:
        {
            vvTFCustom2D *w = static_cast<vvTFCustom2D *>((**it).vvWidget);
            tb << TUITFEWidget::TF_CUSTOM_2D;
            tb << w->_opacity;
            tb << w->_centralPoint->_opacity;
            tb << w->_centralPoint->_pos[0];
            tb << w->_centralPoint->_pos[1];
            tb << (int)w->_extrude;

            tb << (int)w->hasOwnColor();
            if (w->hasOwnColor())
            {
                tb << w->_col[0];
                tb << w->_col[1];
                tb << w->_col[2];
            }

            uint32_t pointNum = (uint32_t)w->_points.size();
            tb << pointNum;

            list<vvTFPoint *>::iterator point_it;
            for (point_it = w->_points.begin(); point_it != w->_points.end(); ++point_it)
            {
                tb << (**point_it)._opacity;
                tb << (**point_it)._pos[0];
                tb << (**point_it)._pos[1];
            }
        }
        break;

        case TUITFEWidget::TF_MAP:
        {
            vvTFCustomMap *w = static_cast<vvTFCustomMap *>((**it).vvWidget);
            tb << TUITFEWidget::TF_MAP;
            tb << w->_opacity;
            tb << w->_pos[0];
            tb << w->_size[0];
            tb << w->_pos[1];
            tb << w->_size[1];

            tb << (int)w->hasOwnColor();
            if (w->hasOwnColor())
            {
                tb << w->_col[0];
                tb << w->_col[1];
                tb << w->_col[2];
            }

            int dim = w->_dim[0] * w->_dim[1] * w->_dim[2];
            tb << dim;
            for (int i = 0; i < dim; ++i)
                tb << w->_map[i];
        }
        break;

        default:
            break;
        }
    }
}

// normalize (range 0..255) and store histogram data
void TUITF2DEditor::setHistogramData(int xDim, int yDim, int *values)
{
    if (histoData)
        delete[] histoData;
    histoData = values;

    histoBuckets[0] = xDim;
    histoBuckets[1] = yDim;

    // normalize

    // find max
    int max = 0;
    for (int i = 0; i < xDim * yDim; ++i)
    {
        assert(histoData[i] >= 0);
        if (histoData[i] > 0)
        {
#ifdef _WIN32_WCE
            histoData[i] = (int)log((double)histoData[i]);
#else
            histoData[i] = (int)logf((float)histoData[i]);
#endif
        }
        max = std::max(histoData[i], max);
    }

    if (max > 0)
    {
        float fx = 255.0f / (float)max;

        for (int i = 0; i < xDim * yDim; ++i)
        {
            int newValue = int((float)histoData[i] * fx);
            assert(newValue >= 0 && newValue < 256);
            histoData[i] = newValue;
        }
    }
}

void TUITF2DEditor::addMarker(TUITFEWidget *)
{
    //TODO; remove this method
}

void TUITF2DEditor::removeMarker(TUITFEWidget *wp)
{
    TUIVirvoWidget *w = dynamic_cast<TUIVirvoWidget *>(wp);
    if (w == NULL)
        return;

    list<TUIVirvoWidget *>::iterator it;

    for (it = widgets.begin(); it != widgets.end(); ++it)
    {
        if (*it == w)
        {
            widgets.erase(it);
            break;
        }
    }

    for (std::vector<vvTFWidget *>::iterator it = tf._widgets.begin();
         it != tf._widgets.end();
         ++it)
    {
        if (w->vvWidget == *it)
        {
            tf._widgets.erase(it);
            break;
        }
    }

    this->canvas2D->setDirtyFlag();
    this->repaint();
}

void TUITF2DEditor::initDrawMap(const QPoint &p)
{
    imgValid = true;
    previousPoint = p;

    for (int j = 0; j < image.height(); ++j)
        for (int i = 0; i < image.width(); ++i)
        {
            image.setPixel(i, j, qRgba(0, 255, 0, 0));
        }
}

void TUITF2DEditor::mousePressEvent(QMouseEvent *e)
{
    if (e->button() == Qt::RightButton)
    {
        rightButtonDown = true;
    }

    if (e->button() == Qt::LeftButton)
    {
        leftButtonDown = true;
    }

    switch (insertionMode)
    {
    case WidgetDrawPoints:
        // start to collect data to (eventually) create
        // a vvTFCustom2D widget
        points.clear();
        this->repaint();
        break;

    case WidgetDrawMap:
    {
        // start to "draw" the temporary image with data from pen.
        initDrawMap(e->pos());
    }
    break;

    case WidgetFixed: //default:
    {
        float posx = (float)e->POSITIONX / panelSize;
        //openGL y coordinates are flipped
        float posy = 1.0f - ((float)e->POSITIONY / panelSize);

        if (posx >= 0.0f && posx <= 1.0f && posy >= 0.0f && posx <= 1.0f)
        {
            if (selectedPoint == NULL)
            {
                // no mark selected; select one or create one
                std::list<TUIVirvoWidget *>::iterator it;
                for (it = widgets.begin(); it != widgets.end(); ++it)
                {
                    TUITFEWidget::HandleType ht = (**it).testHit(e->POSITIONX, panelSize - e->POSITIONY);
                    if (ht != TUITFEWidget::HT_NONE)
                    {
                        selectedPoint = *it;
                        emit pickPoint(selectedPoint);
                        break;
                    }
                }

                //have we found something?
                if (selectedPoint == NULL)
                {
                    if (currentWidgetType == TUITFEWidget::TF_CUSTOM_2D_EXTRUDE || currentWidgetType == TUITFEWidget::TF_CUSTOM_2D_TENT)
                    {
                        QMessageBox::information(this, "Erroneous selection", "For using the selected tool, you have to select a 'Draw contour' widget first");
                    }
                    else
                    {
                        // no -> click on empty point -> create new
                        TUIVirvoWidget *newWidget = make2DWidget(posx, posy, currentWidgetType);
                        this->selectedPoint = newWidget;
                        emit newPoint(newWidget);

                        this->canvas2D->setDirtyFlag();
                        this->repaint();
                    }
                }
                // else we selected something else, deal with it in mouseMove
            }
            else
            {
                // something was already selected
                if (currentWidgetType == TUITFEWidget::TF_CUSTOM_2D_EXTRUDE || currentWidgetType == TUITFEWidget::TF_CUSTOM_2D_TENT)
                {
                    if (selectedPoint->getWidgetType() != TUITFEWidget::TF_CUSTOM_2D)
                    {
                        QMessageBox::information(this, "Erroneous selection", "For using the selected tool, you have to select a 'Draw contour' widget first");
                    }
                    else
                    {
                        // we have a free form widget, and we want to extrude it or build a tent

                        // make2DWidget copies the correct points from the selectedPoint
                        // (we know for sure it is a TF_CUSTOM_2D)
                        TUIVirvoWidget *newWidget = make2DWidget(posx, posy, currentWidgetType);

                        // then, we proceed as usual
                        this->selectedPoint = newWidget;
                        emit newPoint(newWidget);

                        this->canvas2D->setDirtyFlag();
                        this->repaint();
                    }
                }
                else
                {
                    bool pointSelected = false;
                    std::list<TUIVirvoWidget *>::iterator it;
                    for (it = widgets.begin(); it != widgets.end(); ++it)
                    {
                        TUITFEWidget::HandleType ht = (**it).testHit(e->POSITIONX, panelSize - e->POSITIONY);
                        if (ht != TUITFEWidget::HT_NONE)
                        {
                            pointSelected = true;
                            selectedPoint = *it;
                            emit pickPoint(selectedPoint);
                            break;
                        }
                    }

                    if (pointSelected == false)
                    {
                        selectedPoint = NULL;
                        emit pickPoint(selectedPoint);
                    }

                    this->canvas2D->setDirtyFlag();
                    this->repaint();
                }
            }
        }
    } //end WidgetFixed case
    } // end switch

    QWidget::mousePressEvent(e);
}

void TUITF2DEditor::mouseMoveEvent(QMouseEvent *e)
{
    switch (insertionMode)
    {
    case WidgetDrawPoints:
        if (leftButtonDown)
        {
            addFreePoint(e->POSITIONX, panelSize - e->POSITIONY, currentAlpha);
            this->repaint();
        }
        break;

    case WidgetDrawMap:
        if (leftButtonDown)
        {
            //update brush
            QColor myColor(Qt::darkCyan);
            myColor.setAlpha(currentAlpha);
            myPen.setWidth(currentBrushWidth);
            myBrush.setColor(myColor);
            myPen.setColor(myColor);

            handleAlpha(e->pos());
            this->repaint();
        }
        break;

    case WidgetFixed:
    {
        if (selectedPoint == NULL)
            return;

        TUIVirvoWidget *currentMarker = static_cast<TUIVirvoWidget *>(selectedPoint);

        switch (currentMarker->getSelectedHandle())
        {
        case TUITFEWidget::HT_NONE:
            return;

        case TUITFEWidget::HT_MIDDLE:
        {
            float xPos = (float)e->POSITIONX / panelSize;
            //openGL y coordinates are flipped
            float yPos = 1.0f - ((float)e->POSITIONY / panelSize);
            if (yPos > 1.0f)
                yPos = 1.0f;
            if (yPos < 0.0f)
                yPos = 0.0f;

            currentMarker->setX(xPos);
            currentMarker->setY(yPos);

            this->canvas2D->setDirtyFlag();
            this->repaint();
        }
        break;

        case TUITFEWidget::HT_BOTTOM:
        {
            //change widget dimension
            float xPos = (float)e->POSITIONX / panelSize;
            //openGL y coordinates are flipped
            float yPos = 1.0f - ((float)e->POSITIONY / panelSize);
            if (yPos > 1.0f)
                yPos = 1.0f;
            if (yPos < 0.0f)
                yPos = 0.0f;

            float xval = ::fabsf(xPos - currentMarker->getX()) * 2.0f;
            float yval = ::fabsf(yPos - currentMarker->getY()) * 2.0f;
            currentMarker->setXParam(0, xval);
            currentMarker->setYParam(0, yval);

            this->canvas2D->setDirtyFlag();
            this->repaint();
        }
        break;

        case TUITFEWidget::HT_TOP:
        {
            //only pyramid has top too

            //change widget dimension
            float xPos = (float)e->POSITIONX / panelSize;
            //openGL y coordinates are flipped
            float yPos = 1.0f - ((float)e->POSITIONY / panelSize);
            if (yPos > 1.0f)
                yPos = 1.0f;
            if (yPos < 0.0f)
                yPos = 0.0f;

            float xval = ::fabsf(xPos - currentMarker->getX()) * 2.0f;
            float yval = ::fabsf(yPos - currentMarker->getY()) * 2.0f;

            currentMarker->setXParam(1, xval);
            currentMarker->setYParam(1, yval);

            this->canvas2D->setDirtyFlag();
            this->repaint();
        }
        break;
        }
    }
    }

    QWidget::mouseMoveEvent(e);
}

void TUITF2DEditor::handleMapRelease(float /*x*/, float /*y*/)
{
    if (insertionMode == WidgetDrawMap)
    {
        // convert to a widget (vvTFCustomMap)
        // find min, max for x and y in order to compute _size and _pos
        int xMax = -1, xMin = textureSize, yMax = -1, yMin = textureSize;
        for (int j = 0; j < textureSize; ++j)
            for (int i = 0; i < textureSize; ++i)
            {
                //int idx = (j * textureSize + i) * 4;
                int alpha = qAlpha(image.pixel(i, j));
                if (alpha > 0)
                {
                    xMin = std::min(xMin, i);
                    yMin = std::min(yMin, j);
                    xMax = std::max(xMax, i);
                    yMax = std::max(yMax, j);
                }
            }

        if (xMax < 0 || yMax < 0)
            return;

        //center widget position
        float xPos = ((float)(xMin + xMax) / 2.0f) / (float)textureSize;
        float yPos = ((float)textureSize - (float)(yMin + yMax) / 2.0f) / (float)textureSize;

        float xSize = (float)(xMax - xMin + 1) / (float)textureSize;
        float ySize = (float)(yMax - yMin + 1) / (float)textureSize;

        vvTFCustomMap *w = new vvTFCustomMap(xPos, xSize, yPos, ySize);
        //w->_opacity = 1.0f; //not used

        for (int j = yMin; j <= yMax; ++j)
            for (int i = xMin; i <= xMax; ++i)
            {
                //int idx = (j * textureSize + i) * 4;
                int alpha = qAlpha(image.pixel(i, j));

                float xPos = (float)i / textureSize;
                float yPos = ((float)textureSize - j) / (float)textureSize;
                //1.0f - (float)(j + 1) / textureSize

                if (alpha > 0)
                {
                    float opacity = (float)(alpha) / 255.0f;
                    w->setOpacity(opacity, xPos, yPos);
                }
                else
                    w->setOpacity(0.0f, xPos, yPos);
            }

        TUIVirvoWidget *widget = new TUIVirvoWidget(TUITFEWidget::TF_MAP, w, &tf);
        widgets.push_back(widget);

        // now that the data is transferred, we don't need our support image anymore
        imgValid = false;

        this->canvas2D->setDirtyFlag();
        this->repaint();
    }
}

void TUITF2DEditor::mouseReleaseEvent(QMouseEvent *e)
{
    // handle free drawing! (update on release)

    // here we handle only the free map case (WidgetDrawMap)

    // WidgetDrawPoint (values in the point list) are transferred into a vvTFCustom2D
    // when a extrusion or tent is created
    if (e->button() == Qt::LeftButton)
    {
        float posx = (float)e->POSITIONX / panelSize;
        //openGL y coordinates are flipped
        float posy = 1.0f - ((float)e->POSITIONY / panelSize);

        handleMapRelease(posx, posy);
        leftButtonDown = false;

        emit functionChanged();
    }

    if (e->button() == Qt::RightButton)
    {
        rightButtonDown = false;
    }

    QWidget::mouseReleaseEvent(e);
}

void TUITF2DEditor::repaint()
{
    this->canvas2D->repaint();
    QWidget::repaint();
}

void TUITF2DEditor::addFreePoint(int x, int y, int alpha)
{
    if (alpha > 0)
    {
        //add the point
        points.push_back(new vvTFPoint((float)alpha / 255.0f, (float)x / 255.0f, (float)y / 255.0f));
    }
}

//
// Tablet handling (pressure)
//

// in case of table handling with pressure, our strategy is the following
// 1) till the tablet draws (draw free button is pressed for the first time and then
// re-pressed or changed) store "drawing" in a QImage.
// 2) when painting, super-impose Qimage to the generated texture
// 3) when stop painting,
//      a) convert image into vvTFCustomMap widget
//      b) re-generate TF texture
//      c) clear QImage
void TUITF2DEditor::tabletEvent(QTabletEvent *e)
{
    // we need special handling only in the free map drawing case
    // otherwise we will use only the standard mouse handlers
    if (insertionMode == WidgetDrawMap)
    {
        switch (e->type())
        {
        case QEvent::TabletPress:
            if (!deviceDown)
            {
                deviceDown = true;
                initDrawMap(e->pos());
            }
            break;

        case QEvent::TabletRelease:
            if (deviceDown)
            {
                deviceDown = false;
                float posx = (float)e->POSITIONX / panelSize;
                //openGL y coordinates are flipped
                float posy = 1.0f - ((float)e->POSITIONY / panelSize);
                handleMapRelease(posx, posy);

                emit functionChanged();
            }
            break;

        case QEvent::TabletMove:
            if (deviceDown)
            {
                updateBrush(e);

                //QColor myColor(Qt::darkCyan);
                //myColor.setAlpha(currentAlpha);
                //myPen.setWidth(currentBrushWidth);
                //myBrush.setColor(myColor);
                //myPen.setColor(myColor);

                handleAlpha(e->pos());
                canvas2D->update();
                //e->accept();
            }
            break;

        default:
            break;
        }
    }
}

void TUITF2DEditor::handleAlpha(const QPoint &point)
{
    QPainter p;
    p.setBackgroundMode(Qt::TransparentMode);
    p.begin(&image);
    paintImage(p, point);
    p.end();

    previousPoint = point;
}

void TUITF2DEditor::paintImage(QPainter &painter, const QPoint &pos)
{
    //QPoint brushAdjust(10, 10);

    painter.setBrush(myBrush);
    painter.setPen(myPen);
    painter.drawLine(previousPoint, pos);
}

void TUITF2DEditor::updateBrush(QTabletEvent *e)
{
    QColor myColor(Qt::darkCyan);

    switch (alphaChannelType)
    {
    case AlphaPressure:
    {
        // more pressure -> more opaque
        int alpha = int(e->pressure() * this->currentAlpha);
        myColor.setAlpha(alpha);
    }
    break;

    case AlphaTilt:
    {
        int vValue = int(((e->yTilt() + 60.0) / 120.0) * 255);
        int hValue = int(((e->xTilt() + 60.0) / 120.0) * 255);

        myColor.setAlpha(std::max(abs(vValue - 127), abs(hValue - 127)));
    }
    break;

    default:
        myColor.setAlpha(255);
    }

    switch (lineWidthType)
    {
    case LineWidthPressure:
        myPen.setWidthF(e->pressure() * 10 + 1);
        break;
    case LineWidthTilt:
    {
        int vValue = int(((e->yTilt() + 60.0) / 120.0) * 255);
        int hValue = int(((e->xTilt() + 60.0) / 120.0) * 255);

        myPen.setWidthF(std::max(abs(vValue - 127), abs(hValue - 127)) / 12);
    }
    break;
    default:
        myPen.setWidth(currentBrushWidth);
    }

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    if (e->pointerType() == QTabletEvent::Eraser)
#else
    if (e->pointerType() == QPointingDevice::PointerType::Eraser)
#endif
    {
        myColor.setAlpha(0);
        myPen.setWidthF(e->pressure() * 10 + 1);
    }

    myBrush.setColor(myColor);
    myPen.setColor(myColor);
}

void TUITF2DEditor::changedBrushWidth(int w)
{
    currentBrushWidth = w;
}

void TUITF2DEditor::changedOwnColor(int state)
{
    if (selectedPoint != NULL)
    {
        TUIVirvoWidget *w = static_cast<TUIVirvoWidget *>(selectedPoint);
        if (state != 0 && !w->hasOwnColor())
        {
            w->setOwnColor(true);
            w->setColor(this->parentControl->getCurrentColor());
            canvas2D->setDirtyFlag();
            canvas2D->repaint();
        }
        else if (state == 0 && w->hasOwnColor())
        {
            w->setOwnColor(false);
            canvas2D->setDirtyFlag();
            canvas2D->repaint();
        }
    }
}

void TUITF2DEditor::setBackType(int newBack)
{
    canvas2D->setBackType(newBack);
}
