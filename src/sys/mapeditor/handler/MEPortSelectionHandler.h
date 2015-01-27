/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_SELECTEDPORTLIST_H
#define ME_SELECTEDPORTLIST_H

#include <QObject>
#include <QHash>
#include <QString>

#include "ports/MEPort.h"

class QGraphicsSceneContextMenuEvent;
class QAction;

class MEHost;
class MEPort;
class MEDataPort;

namespace covise
{
class coSendBuffer;
}

class MEPortSelectionHandler : public QObject
{
    Q_OBJECT

public:
    enum Type
    {
        Clicked,
        Connectable,
        HoverConnectable,
        HoverConnected,
        NumSelections
    };

    MEPortSelectionHandler();
    ~MEPortSelectionHandler();

    static MEPortSelectionHandler *instance();

    bool isEmpty(Type type);
    int count(Type type);
    void clear(Type type);
    bool contains(Type type, MEPort *);
    void addPort(Type type, MEPort *);
    void removePort(Type type, MEPort *);
    void showPossiblePorts(MEDataPort *, QGraphicsSceneContextMenuEvent *);

private:
    QVector<MEPort *> m_selectedPortList[NumSelections];
    QVector<QAction *> m_portConnectionList;
    QHash<QString, MEPort *> m_translate;

    QStringList m_popupItems;
    QMenu *m_portPopup;
    MEDataPort *m_clickedPort;

private slots:

    void connectPortCB();
};
#endif
