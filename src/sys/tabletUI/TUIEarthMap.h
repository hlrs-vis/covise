/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_EARTHMAP_H
#define CO_TUI_EARTHMAP_H

#include <QtGlobal>
#if QT_VERSION >= 0x050900
#define HAVE_TUIEARTHMAP
#endif

#include <QObject>
#include <QMatrix>
#include <QPolygon>
#include <QPixmap>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPolygonItem>
#ifdef HAVE_TUIEARTHMAP
#include <QGeoPath>
#endif


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
class QQuickView;


/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
class TUIEarthMap : public QObject, public TUIElement
{
    Q_OBJECT

public:
    TUIEarthMap(int id, int type, QWidget *w, int parent, QString name);
    virtual ~TUIEarthMap();
#ifdef HAVE_TUIEARTHMAP
    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

	void setValue(TabletValue type, covise::TokenBuffer &tb);

    float latitude, longitude, altitude;


protected:
    QWidget *container;
    QQuickView *quickView;
    QGeoPath geopath;
    bool centerHeli;
#endif
};
#endif
