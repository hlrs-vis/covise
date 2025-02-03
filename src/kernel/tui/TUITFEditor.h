/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TUI_TFE_H_INCLUDED
#define TUI_TFE_H_INCLUDED

#include <QWidget>
#include <QLineEdit>
#include <net/covise_connect.h>

#include "TUITFEWidgets.h"

class TUIFunctionEditorTab;

class TUITFEditor : public QWidget
{
    Q_OBJECT

public:
    static const int markerSize = 14; // size of color markers
    static const int halfMarkerSize = 14 / 2;
    static const int clickThreshold = halfMarkerSize + 1;

    TUITFEditor(TUIFunctionEditorTab *c, QWidget *parent = 0)
        : QWidget(parent)
        , parentControl(c)
        , selectedPoint(NULL)
    {
    }
    virtual ~TUITFEditor()
    {
    }

    virtual void addMarker(TUITFEWidget *) = 0;
    virtual void removeMarker(TUITFEWidget *) = 0;

    TUITFEWidget *getSelectedMarker()
    {
        return selectedPoint;
    }

    virtual void parseMessage(covise::TokenBuffer &tb) = 0;
    virtual void valueChanged(covise::TokenBuffer &tb) = 0;
    virtual void newWidgetValue(float value)
    {
        (void)value;
    }
    virtual void loadDefault() = 0;

protected:
    TUIFunctionEditorTab *parentControl;
    TUITFEWidget *selectedPoint;

    bool rightButtonDown;

    //bool drawFree;

    virtual void setAlphaDrawFlag()
    {
    }
    virtual void clearAlphaDrawFlag()
    {
    }

public slots:
    virtual void setDrawAlphaFree(bool b)
    {
        if (b)
            setAlphaDrawFlag();
        else
            clearAlphaDrawFlag();
    }
    virtual void eraseAlphaFree()
    {
    }
    virtual void removeCurrentMarker();

signals:
    void functionChanged();

    void newPoint(TUITFEWidget *w);
    void deletePoint(TUITFEWidget *w);
    void pickPoint(TUITFEWidget *wp);
    void movePoint(TUITFEWidget *wp);
};

// This class allows to edit 1D transfer functions.
// The widget is made of three parts: alpha ramp markers,
// a color map, markers to edit the map color points
class TUITF1DEditor : public TUITFEditor
{
    Q_OBJECT

public: //constants
    static const int panelSize = 96; // height of color table
    static const int alphaWidgetPos = 2; // position of the alpha widgets line
    static const int panelPos = alphaWidgetPos + markerSize;
    //position of the color widgets line
    static const int colorWidgetPos = panelPos + panelSize;

public:
    TUITF1DEditor(TUIFunctionEditorTab *c, QWidget *parent = 0);
    ~TUITF1DEditor();

    void addMarker(TUITFEWidget *);
    void removeMarker(TUITFEWidget *w);
    void parseMessage(covise::TokenBuffer &tb);
    void valueChanged(covise::TokenBuffer &tb);
    void newWidgetValue(float value);
    void loadDefault();

private:
    QList<TUIColorPoint *> colorPoints;
    QList<TUIAlphaTriangle *> alphaPoints;
    TUIAlphaFree alphaFreeForm;

    TUIAlphaTriangle *makeAlphaPoint(float x, int a, float xb, float xt);
    TUIColorPoint *makeColorPoint(float pos, QColor col);

    TUITFEWidget *newColorPoint(float xPos);
    TUITFEWidget *newAlphaPoint(float xPos);

    bool drawFree;

protected:
    void mousePressEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
    void paintEvent(QPaintEvent *e);

public slots:
    void setDrawAlphaFree(bool b)
    {
        if (b)
            setAlphaDrawFlag();
        else
            clearAlphaDrawFlag();
    }

    void eraseAlphaFree();
    void setAlphaDrawFlag();
    void clearAlphaDrawFlag();
};
#endif //TUI_TFE_H_INCLUDED
