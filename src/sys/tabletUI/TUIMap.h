/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_MAP_H
#define CO_TUI_MAP_H

#include <QObject>
#include <QMatrix>
#include <QPolygon>
#include <QPixmap>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPolygonItem>

#include "TUIElement.h"

class QTimer;
class QContextMenuEvent;
class QDropEvent;
class QMouseEvent;
class QResizeEvent;
class QDragEnterEvent;
class QGridLayout;
class QFrame;

class CamItem;
class NodeItem;

// We use a global variable to save memory - all the brushes and pens in
// the mesh are shared.

class QTabWidget;
class QPixmap;
class QPushButton;

class MapCanvas : public QGraphicsScene
{
    Q_OBJECT
public:
    MapCanvas();
    virtual ~MapCanvas();
    void setTiles(const QPixmap &pixmap, int h, int v,
                  int tileHeight, int tileWidth);
    void setTile(int x, int y, int tilenum);

private:
    QRect tileRect(int x, int y) const;
    QRect tileRect(int tileNum) const;

    QVector<QVector<int> > tiles;
    QPixmap tilePixmap;
    int tileW, tileH;
    int hTiles, vTiles;
    void drawBackground(QPainter *painter, const QRectF &rect);

protected:
};

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUIMap : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUIMap(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIMap();
    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);

    /// get the Element's classname
    virtual char *getClassName();
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(char *);

    bool doZoom;
    bool doPan;
    bool doCam;
    int rdx, rdy, rhx, rhy;
    float scale;
    void doScale();
    CamItem *camera;
    NodeItem *from;
    NodeItem *to;

public slots:

    void zoomIn();
    void zoomOut();
    void viewAll();
    void zoomFrame(bool);
    void pan(bool);
    void cam(bool);

protected:
    QMatrix wm;
    MapCanvas *canvas;
    QGraphicsView *view;
    QTabWidget *tabWidget;
    QFrame *frame;
    QGridLayout *layout;
    QGridLayout *buttonsLayout;
    QPushButton *zoomInButton;
    QPushButton *zoomOutButton;
    QPushButton *viewAllButton;
    QPushButton *zoomFrameButton;
    QPushButton *panButton;
    QPushButton *camButton;
    QPixmap *pm;
};

class MapCanvasView : public QGraphicsView
{
    Q_OBJECT
public:
    MapCanvasView(TUIMap *tm, QGraphicsScene *canvas,
                  QWidget *parent = 0);
    virtual ~MapCanvasView();

protected:
    void viewportResizeEvent(QResizeEvent *e);
    void contentsMousePressEvent(QMouseEvent *e);
    void contentsMouseMoveEvent(QMouseEvent *e);
    void contentsContextMenuEvent(QContextMenuEvent *e);

private:
    bool connected, grouped, rubberMove, firstMove, nodeMove;
    int xx, yy, xg, yg, rx, ry, startCPosX, startCPosY;
    int xoff, yoff;
    int scroll_time;
    int initialTime;
    int myMargin;
    TUIMap *tuiMap;
    QFrame *rubber;
    QTimer *scroll_timer;
    QRect view;
    QPoint m_pos;
    void startAutoScroll();
    void stopAutoScroll();
    QGraphicsItem *moving;
    QPoint moving_start;
    void moveRubber(QMouseEvent *e);

protected:
    void contentsDragEnterEvent(QDragEnterEvent *e);
    void contentsDropEvent(QDropEvent *e);
    void contentsMouseReleaseEvent(QMouseEvent *e);

protected slots:
    void doAutoScroll();
};

class CamItem : public QGraphicsPolygonItem
{
public:
    CamItem(NodeItem *, NodeItem *, QGraphicsScene *canvas);
    void setFromPoint(int x, int y);
    void setToPoint(int x, int y);
    void moveBy(double dx, double dy);
    void recalc(int fx, int fy, int tx, int ty);

private:
    QPolygon pa, pao;
    NodeItem *from;
    NodeItem *to;
    bool dontUpdateFromAndTo;
};

class NodeItem : public QGraphicsEllipseItem
{
public:
    NodeItem(QGraphicsScene *canvas);
    ~NodeItem()
    {
    }

    void addCamFrom(CamItem *c)
    {
        camItemFrom = c;
    }
    void addCamTo(CamItem *c)
    {
        camItemTo = c;
    }

    void moveBy(double dx, double dy);

    //    QPoint center() { return boundingRect().center(); }
private:
    CamItem *camItemFrom;
    CamItem *camItemTo;
};
#endif
