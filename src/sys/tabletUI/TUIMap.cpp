/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#include <QTimer>
#include <QPushButton>
#include <QCursor>
#include <QTabWidget>
#include <QFrame>
#include <QContextMenuEvent>
#include <QDropEvent>
#include <QResizeEvent>
#include <QGridLayout>
#include <QPixmap>
#include <QMouseEvent>
#include <QDragEnterEvent>
#include <QMimeData>

#include "TUIMap.h"
#include "TUIApplication.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159
#endif

static QBrush *tb = 0;
static QPen *tp = 0;
MapCanvas::MapCanvas()
    : QGraphicsScene()
{
}

MapCanvas::~MapCanvas()
{
}

void MapCanvas::setTiles(const QPixmap &pixmap, int h, int v,
                         int tileHeight, int tileWidth)
{
    tilePixmap = pixmap;
    tileW = tileWidth;
    tileH = tileHeight;
    hTiles = h;
    vTiles = v;

    tiles.resize(v);
    for (int y = 0; y < v; ++y)
        tiles[y].resize(h);
}

void MapCanvas::setTile(int x, int y, int tilenum)
{
    tiles[y][x] = tilenum;
    update(tileRect(x, y));
}

QRect MapCanvas::tileRect(int x, int y) const
{
    return QRect(x * tileW, y * tileH, tileW, tileH);
}

QRect MapCanvas::tileRect(int tileNum) const
{
    int numHTiles = tilePixmap.width() / tileW;
    int numVTiles = tilePixmap.height() / tileH;
    return tileRect(tileNum % numVTiles, tileNum / numHTiles);
}

void MapCanvas::drawBackground(QPainter *painter, const QRectF &exposed)
{
    for (int y = 0; y < vTiles; ++y)
    {
        for (int x = 0; x < hTiles; ++x)
        {
            QRect destRect = tileRect(x, y);
            if (exposed.intersects(destRect))
            {
                painter->drawPixmap(destRect, tilePixmap,
                                    tileRect(tiles[y][x]));
            }
        }
    }
}

/// Constructor
TUIMap::TUIMap(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    if (!tb)
        tb = new QBrush(Qt::red);
    if (!tp)
        tp = new QPen(Qt::black);
    pm = new QPixmap(name);
    if (pm->isNull())
    {
        canvas = (MapCanvas *)new QGraphicsScene(0, 0, 100, 100);
    }
    else
    {
        canvas = (MapCanvas *)new QGraphicsScene();
        canvas->setTiles(*pm, 1, 1, pm->width(), pm->height());
    }
#if 0
   // TODO porting to QT4
   canvas->setCacheMode(QGraphicsView::CacheBackground);
#endif
    QFrame *frame1 = new QFrame(w);
    frame1->setFrameStyle(QFrame::NoFrame);

    tabWidget = new QTabWidget(frame1);
    tabWidget->setObjectName(name);

    QFrame *frame = new QFrame(frame1);
    frame->setFrameStyle(QFrame::StyledPanel);

    layout = new QGridLayout(frame1);
    widget = frame1;
    layout->addWidget(tabWidget, 0, 0);
    layout->addWidget(frame, 0, 1);

    buttonsLayout = new QGridLayout(frame);
    buttonsLayout->setRowStretch(8, 100);
    zoomInButton = new QPushButton(frame);
    zoomOutButton = new QPushButton(frame);
    viewAllButton = new QPushButton(frame);
    zoomFrameButton = new QPushButton(frame);
    panButton = new QPushButton(frame);
    camButton = new QPushButton(frame);
    from = new NodeItem(canvas);
    to = new NodeItem(canvas);
    camera = new CamItem(from, to, canvas);

    QPixmap pIn("zoomIn.png");
    if (pIn.isNull())
    {
        zoomInButton->setText("zoomIn");
    }
    else
    {
        zoomInButton->setIcon(pIn);
    }
    QPixmap pOut("zoomOut.png");
    if (pOut.isNull())
    {
        zoomOutButton->setText("zoomOut");
    }
    else
    {
        zoomOutButton->setIcon(pOut);
    }

    QPixmap pAll("viewAll.png");
    if (pAll.isNull())
    {
        viewAllButton->setText("viewAll");
    }
    else
    {
        viewAllButton->setIcon(pAll);
    }
    QPixmap pFrame("zoomFrame.png");
    if (pFrame.isNull())
    {
        zoomFrameButton->setText("zoomFrame");
    }
    else
    {
        zoomFrameButton->setIcon(pFrame);
    }
    QPixmap pPan("pan.png");
    if (pPan.isNull())
    {
        panButton->setText("pan");
    }
    else
    {
        panButton->setIcon(pPan);
    }
    QPixmap pCam("Cam.png");
    if (pCam.isNull())
    {
        camButton->setText("Cam");
    }
    else
    {
        camButton->setIcon(pCam);
    }

    zoomInButton->setFixedSize(zoomInButton->sizeHint());
    zoomOutButton->setFixedSize(zoomOutButton->sizeHint());
    viewAllButton->setFixedSize(viewAllButton->sizeHint());
    panButton->setFixedSize(panButton->sizeHint());
    zoomFrameButton->setFixedSize(zoomFrameButton->sizeHint());
    camButton->setFixedSize(camButton->sizeHint());
    zoomInButton->setCheckable(false);
    zoomOutButton->setCheckable(false);
    viewAllButton->setCheckable(false);
    zoomFrameButton->setCheckable(true);
    panButton->setCheckable(true);
    camButton->setCheckable(true);

    connect(zoomInButton, SIGNAL(pressed()), this, SLOT(zoomIn()));
    connect(zoomOutButton, SIGNAL(pressed()), this, SLOT(zoomOut()));
    connect(viewAllButton, SIGNAL(pressed()), this, SLOT(viewAll()));
    connect(zoomFrameButton, SIGNAL(toogled(bool)), this, SLOT(zoomFrame(bool)));
    connect(panButton, SIGNAL(toggled(bool)), this, SLOT(pan(bool)));
    connect(camButton, SIGNAL(toggled(bool)), this, SLOT(cam(bool)));

    buttonsLayout->addWidget(zoomInButton, 0, 0);
    buttonsLayout->addWidget(zoomOutButton, 1, 0);
    buttonsLayout->addWidget(viewAllButton, 2, 0);
    buttonsLayout->addWidget(zoomFrameButton, 3, 0);
    buttonsLayout->addWidget(panButton, 4, 0);
    buttonsLayout->addWidget(camButton, 5, 0);

    view = new MapCanvasView(this, canvas, tabWidget);
    view->setObjectName(name);
    tabWidget->insertTab(0, view, name);
    scale = 1;
    wm.scale(1, 1); // Zooms in by 2 times
    view->setMatrix(wm);
    doZoom = false;
    doPan = false;
    doCam = false;
    //canvas->setFixedSize(canvas->sizeHint());
    //connect(b, SIGNAL(pressed()), this, SLOT(pressed())) ;
    //connect(b, SIGNAL(released()), this, SLOT(released())) ;
}

/// Destructor
TUIMap::~TUIMap()
{
    delete widget;
}

void TUIMap::zoomIn()
{
    scale *= 1.1f;
    if (scale >= 1.0f)
    {
        scale = 1.0f;
    }
    else
    {
        wm.scale(1.1, 1.1); // Zooms in by 0.2
        view->setMatrix(wm);
        view->update();
    }
    std::cerr << "zoomIn" << scale << std::endl;
}

void TUIMap::zoomOut()
{
    scale *= 0.9f;
    wm.scale(0.9, 0.9); // Zooms out by 0.2
    view->setMatrix(wm);
    view->update();
    std::cerr << "zoomOut" << scale << std::endl;
}

void TUIMap::viewAll()
{
    wm.reset();
    view->setMatrix(wm);
    view->fitInView(view->sceneRect());
}

void TUIMap::zoomFrame(bool dz)
{
    if (dz)
    {
        doZoom = true;
        if (panButton->isChecked())
            panButton->toggle();
        if (camButton->isChecked())
            camButton->toggle();
    }
    else
    {
        doZoom = false;
    }
}

void TUIMap::doScale()
{
    wm.reset();
    rhx = (int)((float)rhx / scale);
    rhy = (int)((float)rhy / scale);
    view->scale(rhx, rhy);
#if 0
   view->setMatrix( wm );
   float sx=(float)view->visibleWidth()/(float)rdx;
   float sy=(float)view->visibleHeight()/(float)rdy;
   if(sy<sx)
      scale = sy;
   else
      scale = sx;
   if(scale>1.0)
      scale = 1;
   wm.reset();
   view->setContentsPos(rhx,rhy);
   wm.scale( scale, scale );
   view->setWorldMatrix( wm );
#endif
    view->update();
}

void TUIMap::pan(bool dp)
{
    if (dp)
    {
        doPan = true;
        if (zoomFrameButton->isChecked())
            zoomFrameButton->toggle();
        if (camButton->isChecked())
            camButton->toggle();
    }
    else
    {
        doPan = false;
    }
}

void TUIMap::cam(bool dc)
{
    if (dc)
    {
        doCam = true;
        if (zoomFrameButton->isChecked())
            zoomFrameButton->toggle();
        if (panButton->isChecked())
            panButton->toggle();
        from->show();
        to->show();
        camera->show();
    }
    else
    {
        doCam = false;
        from->hide();
        to->hide();
        camera->hide();
    }
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIMap::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIMap::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUIMap::getClassName() const
{
    return "TUIMap";
}

bool TUIMap::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return TUIElement::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

MapCanvasView::~MapCanvasView()
{
}

MapCanvasView::MapCanvasView(TUIMap *tm,
                             QGraphicsScene *canvas, QWidget *parent)
    : QGraphicsView(canvas, parent)
{
    tuiMap = tm;
    myMargin = 50;
    initialTime = 100;
    scroll_time = 100;
    scroll_timer = new QTimer();
    connect(scroll_timer, SIGNAL(timeout()),
            this, SLOT(doAutoScroll()));

    // init a rubber reactangle
    rubber = new QFrame(viewport());
#if 0
   addChild(rubber);
#endif
    rubber->setFrameStyle(QFrame::Panel | QFrame::Sunken);
    rubber->hide();
    moving = NULL;
}

//------------------------------------------------------------------------
// follow the rubber banding rectangle
//------------------------------------------------------------------------
void MapCanvasView::moveRubber(QMouseEvent *e)
{
    if (tuiMap->doZoom)
    {
        if (firstMove)
        {
            firstMove = false;
            rubber->show();
            setCursor(QCursor(Qt::SizeAllCursor));
        }

        tuiMap->rhx = rx;
        tuiMap->rhy = ry;
        if (rx < e->x())
            tuiMap->rdx = e->x() - rx;
        else
        {
            tuiMap->rdx = rx - e->x();
            tuiMap->rhx = e->x();
        }
        if (ry < e->y())
            tuiMap->rdy = e->y() - ry;
        else
        {
            tuiMap->rdy = ry - e->y();
            tuiMap->rhy = e->y();
        }

#if 0
      moveChild(rubber, tuiMap->rhx, tuiMap->rhy);
      rubber->resize(tuiMap->rdx, tuiMap->rdy);
#endif
    }
    else
    {
        rubber->hide();
    }
}

//------------------------------------------------------------------------
// look if drag object can be decoded
//------------------------------------------------------------------------
void MapCanvasView::contentsDragEnterEvent(QDragEnterEvent *event)
{
    if (event->mimeData()->hasText())
        event->acceptProposedAction();
}

//------------------------------------------------------------------------
// reread the dragged module information
// request the node from the controller
//------------------------------------------------------------------------
void MapCanvasView::contentsDropEvent(QDropEvent *event)
{

    if (event->mimeData()->hasText())
    {
        //QString text = event->mimeData()->text();
        event->accept();
    }

    else
        event->ignore();
}

//------------------------------------------------------------------------
// start the auto scroll option max. 30 msec
//------------------------------------------------------------------------
void MapCanvasView::startAutoScroll()
{
    if (!scroll_timer->isActive())
    {
        scroll_time = initialTime;
        scroll_timer->start(scroll_time);
    }
}

//------------------------------------------------------------------------
// stop the auto scroll
//------------------------------------------------------------------------
void MapCanvasView::stopAutoScroll()
{
    scroll_timer->stop();
}

//------------------------------------------------------------------------
// scroll the QScrollView
//------------------------------------------------------------------------
void MapCanvasView::doAutoScroll()
{
    int dx = 0;
    int dy = 0;

    // get the global cursor position
    QPoint p = viewport()->mapFromGlobal(QCursor::pos());
    if (scroll_time)
    {
        scroll_time--;
        scroll_timer->start(scroll_time);
    }

    int l = qMax(1, initialTime - scroll_time);

    // border of viewport reached ???
    if (p.y() > viewport()->height() - myMargin)
    {
        dy = l;
    }
    else if (p.y() < myMargin)
    {
        dy = -l;
    }
    else if (p.x() > viewport()->width() - myMargin)
    {
        dx = l;
    }
    else if (p.x() < myMargin)
    {
        dx = -l;
    }

    // set a new upper, left corner
    if (dx || dy)
    {
#if 0
      // TODO porting to qt4
      setContentsPos( QMAX( contentsX()+dx, 0 ), QMAX( contentsY()+dy, 0 ) );
#endif
    }

    // stop autoscroll
    else
    {
        stopAutoScroll();
    }

    //QPoint c = viewportToContents(p);
}

void MapCanvasView::viewportResizeEvent(QResizeEvent * /*e*/)
{
#if 0
   // TODO porting to QT4 but as the base class implementation ignores e
   // this can be removed
   QGraphicsView::viewportResizeEvent(e);
#endif
    //std::cerr << e << std::endl;
}

void MapCanvasView::contentsContextMenuEvent(QContextMenuEvent * /*e*/)
{
//std::cerr << e << std::endl;
#if 0
   // TODO porting to QT4
   Q3CanvasView::contentsContextMenuEvent(e);
#endif
}

//------------------------------------------------------------------------
// process mouse press events
//------------------------------------------------------------------------
void MapCanvasView::contentsMousePressEvent(QMouseEvent *e)
{
    /*
   QPoint p = inverseWorldMatrix().map(e->pos());
   if(tuiMap->doCam)
   {
         QCanvasItemList l=canvas()->collisions(p);
         for (QCanvasItemList::Iterator it=l.begin(); it!=l.end(); ++it) {
             if ( (*it)->rtti() == imageRTTI ) {
                 ImageItem *item= (ImageItem*)(*it);
                 if ( !item->hit( p ) )
                      continue;
             }            moving = *it;
             moving_start = p;
             return;
         }
   }
   */
    moving = 0;

    //QPoint offset = e->pos() - m_pos;
    m_pos = e->pos();
    //std::cerr << e->pos().x() << std::endl;

    firstMove = true;
    rx = e->x();
    ry = e->y();
// TODO proper porting to qt4
#if 0
   startCPosX = contentsX();
   startCPosY = contentsY();
#endif
    startCPosX = 0;
    startCPosY = 0;
}

//------------------------------------------------------------------------
// process mouse release events
//------------------------------------------------------------------------
void MapCanvasView::contentsMouseReleaseEvent(QMouseEvent *)
{
    // a rubber rectangle was finished
    if (rubberMove)
    {
        rubber->hide();
        if (tuiMap->doZoom)
        {
            tuiMap->doScale();
        }
        firstMove = true;
        rubberMove = false;
    }
    // stop the auto scroll and reset cursor
    stopAutoScroll();
    setCursor(QCursor(Qt::ArrowCursor));
}

//------------------------------------------------------------------------
// process mouse motion events on canvas
//------------------------------------------------------------------------
void MapCanvasView::contentsMouseMoveEvent(QMouseEvent *e)
{
    //QPoint offset = e3->pos() - m_pos;
    m_pos = e->pos();
    std::cerr << e->pos().x() << std::endl;

    if (tuiMap->doPan)
    {
// TODO proper fix
#if 0
      setContentsPos(startCPosX+(rx-e->x()),startCPosY+(ry-e->y()));
#endif
        ensureVisible(QRectF(startCPosX + (rx - e->x()), startCPosY + (ry - e->y()), 5, 5));
    }

    if (moving)
    {
        QPoint p = matrix().inverted().map(e->pos());
        moving->moveBy(p.x() - moving_start.x(),
                       p.y() - moving_start.y());
        moving_start = p;
        scene()->update();
    }

#if 0
   // proper porting to qt4
   QGraphicsView::contentsMouseMoveEvent(e);
#endif
    // follow the rubber banding rectangle
    rubberMove = true;
    moveRubber(e);
}

void CamItem::moveBy(double dx, double dy)
{
    if (!dontUpdateFromAndTo)
    {
        from->moveBy(dx, dy);
        to->moveBy(dx, dy);
    }
    QGraphicsPolygonItem::moveBy(dx, dy);
}

CamItem::CamItem(NodeItem *f, NodeItem *t, QGraphicsScene *canvas)
    : QGraphicsPolygonItem(0), pa(8), pao(8)
{
    from = f;
    to = t;
#define CAMSIZE 10
    pa.setPoint(0, -CAMSIZE, 0);
    pa.setPoint(1, CAMSIZE, 0);
    pa.setPoint(2, CAMSIZE, (int)(2.5 * CAMSIZE));
    pa.setPoint(3, 0, (int)(2.5 * CAMSIZE));
    pa.setPoint(4, CAMSIZE / 2, 4 * CAMSIZE);
    pa.setPoint(5, -CAMSIZE / 2, 4 * CAMSIZE);
    pa.setPoint(6, 0, (int)(2.5 * CAMSIZE));
    pa.setPoint(7, -CAMSIZE, (int)(2.5 * CAMSIZE));

    pao.setPoint(0, -CAMSIZE, 0);
    pao.setPoint(1, CAMSIZE, 0);
    pao.setPoint(2, CAMSIZE, (int)(2.5 * CAMSIZE));
    pao.setPoint(3, 0, (int)(2.5 * CAMSIZE));
    pao.setPoint(4, CAMSIZE / 2, 4 * CAMSIZE);
    pao.setPoint(5, -CAMSIZE / 2, 4 * CAMSIZE);
    pao.setPoint(6, 0, (int)(2.5 * CAMSIZE));
    pao.setPoint(7, -CAMSIZE, (int)(2.5 * CAMSIZE));

#undef CAMSIZE
    setBrush(QColor(0, 0, 0));
    from->addCamFrom(this);
    to->addCamTo(this);
    recalc(int(from->x()), int(from->y()), int(to->x()), int(to->y()));
    setPolygon(pa);
    setZValue(127);
    dontUpdateFromAndTo = false;
}

void CamItem::recalc(int fx, int fy, int tx, int ty)
{
    if (fx == tx && fy == ty)
        return;
    dontUpdateFromAndTo = true;
    double dx = tx - fx;
    double dy = ty - fy;
    double len = sqrt(dx * dx + dy * dy);
    dy /= len;
    dx /= len;
    //std::cerr << "dx" << dx << std::endl;
    //std::cerr << "dy" << dy << std::endl;
    //std::cerr << "len" << len << std::endl;
    double angle = acos(dy);
    if (dx > 0)
        angle *= -1;
    QMatrix rot;
    std::cerr << "angle" << angle << std::endl;
    rot.rotate(angle * 180.0 / M_PI);
    for (int i = 0; i < 8; i++)
    {
        pa.setPoint(i, (pao.point(i)) * rot);
    }
    setPolygon(pa);
    setPos(fx, fy);
    dontUpdateFromAndTo = false;
}

void CamItem::setFromPoint(int x, int y)
{
    recalc(x, y, (int)to->x(), (int)to->y());
}

void CamItem::setToPoint(int x, int y)
{
    recalc((int)from->x(), (int)from->y(), x, y);
}

void NodeItem::moveBy(double dx, double dy)
{
    QGraphicsEllipseItem::moveBy(dx, dy);

    if (camItemTo)
        camItemTo->setToPoint(int(x()), int(y()));
    if (camItemFrom)
        camItemFrom->setFromPoint(int(x()), int(y()));
}

NodeItem::NodeItem(QGraphicsScene *canvas)
    : QGraphicsEllipseItem(0, 0, 10, 10, 0)
{

    camItemFrom = NULL;
    camItemTo = NULL;
    setPen(*tp);
    setBrush(*tb);
    setZValue(128);
}
