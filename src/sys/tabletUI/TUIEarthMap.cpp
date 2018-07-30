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
#include <QStringLiteral>
#include <QQmlEngine>
#include <QQuickItem>

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
    QVariantMap parameters;
    QStringList args;

    // Fetch tokens from the environment, if present
    const QByteArray mapboxMapID = qgetenv("MAPBOX_MAP_ID");
    const QByteArray mapboxAccessToken = qgetenv("MAPBOX_ACCESS_TOKEN");
    const QByteArray hereAppID = qgetenv("HERE_APP_ID");
    const QByteArray hereToken = qgetenv("HERE_TOKEN");
    const QByteArray esriToken = qgetenv("ESRI_TOKEN");

    if (!mapboxMapID.isEmpty())
        parameters["mapbox.map_id"] = QString::fromLocal8Bit(mapboxMapID);
    if (!mapboxAccessToken.isEmpty()) {
        parameters["mapbox.access_token"] = QString::fromLocal8Bit(mapboxAccessToken);
        parameters["mapboxgl.access_token"] = QString::fromLocal8Bit(mapboxAccessToken);
    }
    if (!hereAppID.isEmpty())
        parameters["here.app_id"] = QString::fromLocal8Bit(hereAppID);
    if (!hereToken.isEmpty())
        parameters["here.token"] = QString::fromLocal8Bit(hereToken);
    if (!esriToken.isEmpty())
        parameters["esri.token"] = QString::fromLocal8Bit(esriToken);

    if (!args.contains(QStringLiteral("osm.useragent")))
        parameters[QStringLiteral("osm.useragent")] = QStringLiteral("QtLocation Mapviewer example");

    quickView->engine()->addImportPath(QStringLiteral(":/imports"));
    //engine.load(QUrl(QStringLiteral("qrc:///mapviewer.qml")));
    quickView->setSource(QUrl(QStringLiteral("qrc:///mapviewer.qml")));

    QObject *item = quickView->rootObject();
    Q_ASSERT(item);

    QMetaObject::invokeMethod(item, "initializeProviders",
        Q_ARG(QVariant, QVariant::fromValue(parameters)));

    

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
	}
	TUIElement::setValue(type, tb);
}
