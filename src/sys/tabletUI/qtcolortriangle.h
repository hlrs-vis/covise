/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
 **
 ** Copyright (C) 2003-2004 Trolltech AS.  All rights reserved.
 **
 ** Licensees holding valid Qt Enterprise Edition licenses may use this
 ** file in accordance with the Qt Solutions License Agreement provided
 ** with the Solution.
 **
 ** This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
 ** WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 ** PURPOSE.
 **
 ** Please email sales@trolltech.com for information
 ** about Qt Solutions License Agreements.
 **
 ** Contact info@trolltech.com if any conditions of this licensing are
 ** not clear to you.
 **
 */
#ifndef QTCOLORTRIANGLE_H
#define QTCOLORTRIANGLE_H
#include <QWidget>
#include <QFrame>
#include <QImage>

class ColorTriangle;
class ColorDisplay;
class QPainter;
class QResizeEvent;
class QMouseEvent;
class QKeyEvent;

#if defined(Q_WS_WIN)
#if !defined(QT_QTCOLORTRIANGLE_EXPORT) && !defined(QT_QTCOLORTRIANGLE_IMPORT)
#define QT_QTCOLORTRIANGLE_EXPORT
#elif defined(QT_QTCOLORTRIANGLE_IMPORT)
#if defined(QT_QTCOLORTRIANGLE_EXPORT)
#undef QT_QTCOLORTRIANGLE_EXPORT
#endif
#define QT_QTCOLORTRIANGLE_EXPORT __declspec(dllimport)
#elif defined(QT_QTCOLORTRIANGLE_EXPORT)
#undef QT_QTCOLORTRIANGLE_EXPORT
#define QT_QTCOLORTRIANGLE_EXPORT __declspec(dllexport)
#endif
#else
#define QT_QTCOLORTRIANGLE_EXPORT
#endif

/*
    Used to store color values in the range 0..255 as doubles.
*/
struct DoubleColor
{
    double r, g, b;

    DoubleColor()
        : r(0.0)
        , g(0.0)
        , b(0.0)
    {
    }
    DoubleColor(double red, double green, double blue)
        : r(red)
        , g(green)
        , b(blue)
    {
    }
    DoubleColor(const DoubleColor &c)
        : r(c.r)
        , g(c.g)
        , b(c.b)
    {
    }
};

/*
    Used to store x-y coordinate values as doubles.
*/
struct DoublePoint
{
    double x;
    double y;

    DoublePoint()
        : x(0)
        , y(0)
    {
    }
    DoublePoint(double xx, double yy)
        : x(xx)
        , y(yy)
    {
    }
    DoublePoint(const DoublePoint &c)
        : x(c.x)
        , y(c.y)
    {
    }
};
/*
    Used to store pairs of DoubleColor and DoublePoint in one structure.
*/
struct Vertex
{
    DoubleColor color;
    DoublePoint point;

    Vertex(const DoubleColor &c, const DoublePoint &p)
        : color(c)
        , point(p)
    {
    }
    Vertex(const QColor &c, const DoublePoint &p)
        : color(DoubleColor((double)c.red(), (double)c.green(),
                            (double)c.blue()))
        , point(p)
    {
    }
};

class QT_QTCOLORTRIANGLE_EXPORT QtColorTriangle : public QWidget
{
    Q_OBJECT
public:
    QtColorTriangle(QWidget *parent = 0);
    ~QtColorTriangle();

    int heightForWidth(int w) const;

    QColor color() const;

signals:
    void colorChanged(const QColor &col);
    void released(const QColor &col);

public slots:
    void setColor(const QColor &col);

private:
    ColorTriangle *triangle;
    ColorDisplay *display;
};

class ColorTriangle : public QFrame
{
    Q_OBJECT

public:
    ColorTriangle(QWidget *parent = 0);
    ~ColorTriangle();

    QSize sizeHint() const;

    void polish();

    QColor color() const;

signals:
    void colorChanged(const QColor &col);
    void released(const QColor &col);

public slots:
    void setColor(const QColor &col);

protected:
    virtual void paintEvent(QPaintEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    void keyPressEvent(QKeyEvent *e);
    void resizeEvent(QResizeEvent *);
    void drawTrigon(QImage *p, const DoublePoint &a, const DoublePoint &b,
                    const DoublePoint &c, const QColor &color);

private:
    double radiusAt(const DoublePoint &pos, const QRect &rect) const;
    double angleAt(const DoublePoint &pos, const QRect &rect) const;
    DoublePoint movePointToTriangle(double x, double y, const Vertex &a,
                                    const Vertex &b, const Vertex &c) const;

    DoublePoint pointFromColor(const QColor &col) const;
    QColor colorFromPoint(const DoublePoint &p) const;

    void genBackground();

    QImage bg;
    double a, b, c;
    DoublePoint pa, pb, pc, pd;

    QColor curColor;
    int curHue;

    bool mustGenerateBackground;
    int penWidth;
    int ellipseSize;

    int outerRadius;
    DoublePoint selectorPos;

    enum SelectionMode
    {
        Idle,
        SelectingHue,
        SelectingSatValue
    } selMode;
};

/*
    ColorDisplay provides the rectangular frame beneath the color
    triangle. It displays the currently selected color.
*/
class ColorDisplay : public QFrame
{
    Q_OBJECT
public:
    ColorDisplay(QWidget *parent = 0);
    ~ColorDisplay();

    QColor color() const;

    QSize sizeHint() const;

public slots:
    void setColor(const QColor &c);

protected:
    virtual void paintEvent(QPaintEvent *e);

private:
    QColor c;
};
#endif
