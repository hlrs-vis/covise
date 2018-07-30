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
#include <QQuickView>
#include <QQmlEngine>
#include <QQuickItem>
#include <QQmlContext>
#include <QGeoPath>

#include "TUIEarthMap.h"
#include "TUIApplication.h"
#include <net/tokenbuffer.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159
#endif

static QBrush *tb = 0;
static QPen *tp = 0;

/// Constructor
TUIEarthMap::TUIEarthMap(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    if (!tb)
        tb = new QBrush(Qt::red);
    if (!tp)
        tp = new QPen(Qt::black);
    quickView = new QQuickView();
    container = QWidget::createWindowContainer(quickView, w);
    container->setMinimumSize(500, 700);
    container->setMaximumSize(500, 700);
    container->setFocusPolicy(Qt::TabFocus);
    widget = container;
    
    int size = 3;

    QGeoPath geopath;
    geopath.addCoordinate(QGeoCoordinate(50.9, 6.5));
    geopath.addCoordinate(QGeoCoordinate(50.8, 6.5));
    geopath.addCoordinate(QGeoCoordinate(50.8, 6.6));

    quickView->engine()->rootContext()->setContextProperty("geopath", QVariant::fromValue(geopath));
    size = geopath.path().size();
    quickView->engine()->rootContext()->setContextProperty("size", size);

    quickView->engine()->addImportPath(QString(":/imports"));
    quickView->setSource(QUrl(QString("qrc:///mapviewer.qml")));

    quickView->setWidth(640);
    quickView->setHeight(640);
    quickView->show();
    QObject *item = quickView->rootObject();
    Q_ASSERT(item);

   // QMetaObject::invokeMethod(item, "initializeProviders",
    //    Q_ARG(QVariant, QVariant::fromValue(parameters)));

    

}

/// Destructor
TUIEarthMap::~TUIEarthMap()
{
    delete widget;
}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIEarthMap::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIEarthMap::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUIEarthMap::getClassName() const
{
    return "TUIEarthMap";
}

bool TUIEarthMap::isOfClassName(const char *classname) const
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



void TUIEarthMap::setValue(TabletValue type, covise::TokenBuffer &tb)
{
	if (type == TABLET_FLOAT)
	{

        tb >> latitude;
        tb >> longitude;
        tb >> altitude;
	
        QVariant returnedValue;
	QMetaObject::invokeMethod(quickView->rootObject(), "setMarker",
        Q_RETURN_ARG(QVariant, returnedValue),
        Q_ARG(QVariant, QVariant::fromValue(QString("Helicopter"))),
        Q_ARG(QVariant, QVariant::fromValue(latitude)),
        Q_ARG(QVariant, QVariant::fromValue(longitude)),
        Q_ARG(QVariant, QVariant::fromValue(altitude)));

	}
	TUIElement::setValue(type, tb);
}
