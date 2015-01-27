/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TUI_TFE2D_H_INCLUDED
#define TUI_TFE2D_H_INCLUDED

#include "TUITFEditor.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <virvo/vvtransfunc.h>
#endif
#include <list>
#include <QTabletEvent>

class Canvas;

class TUITF2DEditor : public TUITFEditor
{
    Q_OBJECT

public:
    //add support for tablet pressure
    enum AlphaChannelType
    {
        AlphaPressure,
        AlphaTilt,
        NoAlpha
    };
    enum LineWidthType
    {
        LineWidthPressure,
        LineWidthTilt,
        NoLineWidth
    };

    enum WidgetInsertionType
    {
        WidgetFixed,
        WidgetDrawPoints,
        WidgetDrawMap
    };

    AlphaChannelType alphaChannelType;
    LineWidthType lineWidthType;

    struct Point
    {
        int x, y, a;
        Point()
        {
            x = 0;
            y = 0;
            a = 0;
        }
        Point(int px, int py, int pa)
        {
            x = px;
            y = py;
            a = pa;
        }
    };

    static const int panelSize = 256; //384;
    static const int tileSize = panelSize / 16;
    static const bool useChecker = false;
    static const int textureSize = 256;

    vvTransFunc tf;
    std::list<TUIVirvoWidget *> widgets;
    void setCurrentWidgetType(TUITFEWidget::TFKind);

    std::list<vvTFPoint *> points;
    WidgetInsertionType insertionMode;
    int currentAlpha;
    int currentBrushWidth;

    TUITF2DEditor(TUIFunctionEditorTab *c, QWidget *parent = 0);
    ~TUITF2DEditor();

    void parseMessage(covise::TokenBuffer &tb);
    void valueChanged(covise::TokenBuffer &tb);
    void addMarker(TUITFEWidget *);
    void removeMarker(TUITFEWidget *);
    void loadDefault()
    { /* do nothing */
    }

private:
    TUITFEWidget::TFKind currentWidgetType;

    Canvas *canvas2D;
    TUIVirvoWidget *make2DWidget(float xPos, float yPos, TUITFEWidget::TFKind);

    //free draw support
    void midPointLine(int x0, int y0, int x1, int y1, int alpha0, int alpha1);
    void closePolygon();
    void addFreePoint(int x, int y, int alpha);

    void internalFloodFill(uchar *map, int x, int y, int xDim, int yDim, uchar oldV, uchar newV);
    void uniformFillFreeArea(uchar *map, int x, int y, int xDim, int yDim, uchar newV);
    void drawFreeContour(uchar *map, int xDim, int yDim);

    void initDrawMap(const QPoint &p);

protected:
    void mousePressEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);

public:
    //
    // Free alpha draw data structures and functions
    //
    QPoint previousPoint;
    QImage image;
    bool imgValid;
    QBrush myBrush;
    QPen myPen;

    void handleAlpha(const QPoint &point);
    void paintImage(QPainter &painter, const QPoint &pos);
    void updateBrush(QTabletEvent *e);

    //
    // Histogram data structures and functions
    //
    int *histoData;
    int histoBuckets[2];
    // normalize (range 0..255) and store histogram data
    void setHistogramData(int xDim, int yDim, int *values);

protected:
    // Tablet pressure handling
    QTabletEvent::TabletDevice myTabletDevice;
    bool deviceDown;
    bool leftButtonDown;
    bool rightButtonDown;

    void tabletEvent(QTabletEvent *e);
    void handleMapRelease(float x, float y);

    void repaint();

public slots:
    void changedBrushWidth(int w);
    void changedOwnColor(int state);
    void setBackType(int newBack);
};

//
//
//
class Canvas : public QWidget
{
    Q_OBJECT

public:
    enum BackType
    {
        BackChecker,
        BackHistogram,
        BackBlack
    };
    Canvas(TUITF2DEditor *parent);
    ~Canvas();

    void setDirtyFlag()
    {
        textureDirty = true;
    }
    void setBackType(int newBack)
    {
        currentBackType = (BackType)newBack;
        repaint();
    }

protected:
    //void initializeGL();
    //void resizeGL(int width, int height);
    //void setupViewport(int width, int height);

    void paintEvent(QPaintEvent *e);
    //void mousePressEvent(QMouseEvent *event);
    //void mouseMoveEvent(QMouseEvent *event);

private:
    TUITF2DEditor *editor;
    uchar *tfTexture;
    bool textureDirty;
    QImage *textureImage;
    BackType currentBackType;

    void drawBackGroundHistogram(QPainter &painter);
    void drawBackGroundCheckers(QPainter &painter);

    void draw2DTFTexture(QPainter &);
    void draw2DTFWidgets(QPainter &);
    void draw2DWidget(QPainter &, TUIVirvoWidget *w);
};
#endif //TUI_TFE2D_H_INCLUDED
