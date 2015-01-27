/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_NODELISTHANDLER_H
#define ME_NODELISTHANDLER_H

#include <QObject>

#include <nodes/MENode.h>
#include "MEPortSelectionHandler.h"

class MEHost;
class MENode;
class MEPort;
class MEDataPort;
class MEGraphicsView;

namespace covise
{
class coSendBuffer;
}

class MENodeListHandler : public QObject
{
    Q_OBJECT

public:
    MENodeListHandler();
    ~MENodeListHandler();

    static MENodeListHandler *instance();

    bool isListEmpty();
    bool nameAlreadyExist(const QString &currName);
    bool nodesForHost(const QString &hostname);
    int count();
    void clearList();
    void printSibling();
    void removeNode(MENode *node);
    void searchMatchingPorts(MEPortSelectionHandler::Type type, MEPort *port);
    void searchMatchingDataPorts(MEDataPort *port);
    void showMatchingNodes(const QString &text);
    MENode *addNode(MEGraphicsView *parent);
    MENode *getNode(const QString &mname, const QString &instance, const QString &hname);
    MENode *getNode(int);
    QList<MENode *> getNodesForHost(MEHost *host);
    QVector<MENode *> getNodes() const;

public slots:

    void findUsedNodes(const QString &category);
    void findUsedNodes2(const QString &category, const QString &module);
    void selectAllNodes();

private:
    QVector<MENode *> nodeList;
};
#endif
