/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "MENodeListHandler.h"
#include "MEMainHandler.h"
#include "MEPortSelectionHandler.h"
#include "widgets/MEGraphicsView.h"
#include "ports/MEDataPort.h"
#include "ports/MEParameterPort.h"

#include <QDebug>

//======================================================================

MENodeListHandler::MENodeListHandler()
    : QObject()
{
}

MENodeListHandler *MENodeListHandler::instance()
{
    static MENodeListHandler *singleton = 0;
    if (singleton == 0)
        singleton = new MENodeListHandler();

    return singleton;
}

//======================================================================
MENodeListHandler::~MENodeListHandler()
//======================================================================
{
    clearList();
}

int MENodeListHandler::count()
{
    return nodeList.count();
}

//------------------------------------------------------------------------
// clear list
//------------------------------------------------------------------------
void MENodeListHandler::clearList()
{
    qDeleteAll(nodeList);
    nodeList.clear();
}

//------------------------------------------------------------------------
// add a node
//------------------------------------------------------------------------
MENode *MENodeListHandler::addNode(MEGraphicsView *parent)
{
    MENode *node = new MENode(parent);
    parent->scene()->addItem(node);
    nodeList.append(node);
    if (parent->autoSelectNewNodes())
        node->setSelected(true);
    return node;
}

//------------------------------------------------------------------------
// remove a node
//------------------------------------------------------------------------
void MENodeListHandler::removeNode(MENode *node)
{
    nodeList.remove(nodeList.indexOf(node));

    // disable execution mode
    if (nodeList.isEmpty())
        MEMainHandler::instance()->enableExecution(false);

    delete node;
}

QVector<MENode *> MENodeListHandler::getNodes() const
{
    return nodeList;
}

//------------------------------------------------------------------------
// get all nodes for a certain host
//------------------------------------------------------------------------
QList<MENode *> MENodeListHandler::getNodesForHost(MEHost *host)
{
    QList<MENode *> list;

    foreach (MENode *nptr, nodeList)
    {
        if (host == nptr->getHost())
            list << nptr;
    }
    return list;
}

//------------------------------------------------------------------------
// look if used nodes are running on this host
//------------------------------------------------------------------------
bool MENodeListHandler::nodesForHost(const QString &text)
{
    foreach (MENode *nptr, nodeList)
    {
        if (text == nptr->getHost()->getShortname())
            return true;
    }
    return false;
}

//------------------------------------------------------------------------
// show all matching canvas nodes and module in the tree
// for a given user search string
//------------------------------------------------------------------------
void MENodeListHandler::showMatchingNodes(const QString &text)
{

    // clear old selected nodes
    MEGraphicsView::instance()->reset();

    // show nodes on canvas
    foreach (MENode *node, nodeList)
    {
        if (node->getCategory().contains(text, Qt::CaseInsensitive) || node->getName().contains(text, Qt::CaseInsensitive))
        {
            node->setSelected(true);
        }
    }
}

//------------------------------------------------------------------------
// print the sibling info of the nodelist
//------------------------------------------------------------------------
void MENodeListHandler::printSibling()
{
    foreach (MENode *nptr, nodeList)
        nptr->printSibling();
}

//------------------------------------------------------------------------
// select all nodes on canvasArea
//------------------------------------------------------------------------
void MENodeListHandler::selectAllNodes()
{
    MEGraphicsView::instance()->reset();

    foreach (MENode *node, nodeList)
        node->setSelected(true);
}

//------------------------------------------------------------------------
// show all canvas nodes for a given module and category name
//------------------------------------------------------------------------
void MENodeListHandler::findUsedNodes2(const QString &category, const QString &module)
{

    MEGraphicsView::instance()->reset();

    foreach (MENode *node, nodeList)
    {
        if (node->getCategory() == category && node->getName() == module)
        {
            node->setSelected(true);
        }
    }
}

//------------------------------------------------------------------------
// show all canvas nodes in a given category
//------------------------------------------------------------------------
void MENodeListHandler::findUsedNodes(const QString &category)
{

    MEGraphicsView::instance()->reset();

    foreach (MENode *node, nodeList)
    {
        if (node->getCategory() == category)
        {
            node->setSelected(true);
        }
    }
}

//------------------------------------------------------------------------
// look if a node has the given name
//------------------------------------------------------------------------
bool MENodeListHandler::nameAlreadyExist(const QString &name)
{
    foreach (MENode *nptr, nodeList)
    {
        if (name == nptr->getTitle())
        {
            return true;
        }
    }
    return false;
}

//------------------------------------------------------------------------
// search matching output ports
//------------------------------------------------------------------------
void MENodeListHandler::searchMatchingDataPorts(MEDataPort *dp)
{
    const QStringList dt = dp->getDataTypes();
    int ni = dt.count();

    foreach (MENode *node, nodeList)
    {
        if (node->getNodeID() == dp->getNode()->getNodeID())
            continue;

        switch (dp->getPortType())
        {
        case MEPort::DIN:
        case MEPort::MULTY_IN:
        {
            foreach (MEDataPort *port, node->doutlist)
            {
                if (!port->isVisible())
                    continue;

                for (int i = 0; i < ni; i++)
                {
                    if (port->getDataTypes().contains(dt[i]) != 0)
                    {
                        MEPortSelectionHandler::instance()->addPort(MEPortSelectionHandler::Connectable, port);
                        break;
                    }
                }
            }
        }
        break;

        case MEPort::DOUT:
        {
            foreach (MEDataPort *port, node->dinlist)
            {
                if (!port->isVisible())
                    continue;

                for (int i = 0; i < ni; i++)
                {
                    if (port->getDataTypes().contains(dt[i]) != 0 && !((port->getPortType() != MEPort::MULTY_IN) && (port->getNoOfLinks() != 0)))
                    {
                        MEPortSelectionHandler::instance()->addPort(MEPortSelectionHandler::Connectable, port);
                        break;
                    }
                }
            }
        }
        break;
        }
    }
}

//------------------------------------------------------------------------
// search matching output ports
//------------------------------------------------------------------------
void MENodeListHandler::searchMatchingPorts(MEPortSelectionHandler::Type type, MEPort *clickedPort)
{

    MEDataPort *dp = qobject_cast<MEDataPort *>(clickedPort);
    if (!dp)
        return;

    // loop over all nodes
    foreach (MENode *node, nodeList)
    {
        // don't examine same node
        if (node == clickedPort->getNode())
            continue;

        switch (dp->getPortType())
        {
        case MEPort::DIN:
        {
            foreach (MEDataPort *port, node->doutlist)
            {
                if (port->isConnectable() && MEDataPort::arePortsCompatible(dp, port))
                {
                    MEPortSelectionHandler::instance()->addPort(type, port);
                }
            }
        }
        break;

        case MEPort::DOUT:
        {
            foreach (MEDataPort *port, node->dinlist)
            {
                if (port->isConnectable() && MEDataPort::arePortsCompatible(port, dp))
                {
                    MEPortSelectionHandler::instance()->addPort(type, port);
                }
            }
        }
        break;
        }
    }
}

//------------------------------------------------------------------------
// look if list is mepty
//------------------------------------------------------------------------
bool MENodeListHandler::isListEmpty()
{
    return nodeList.isEmpty();
}

//------------------------------------------------------------------------
// get a node for a given name, instance and hostname
//------------------------------------------------------------------------
MENode *MENodeListHandler::getNode(const QString &mname, const QString &number, const QString &hname)
{

    foreach (MENode *nptr, nodeList)
    {
        if ((nptr->getName() == mname) && (nptr->getHostname() == hname) && (nptr->getNumber() == number))
        {
            return (nptr);
        }
    }

    return (NULL);
}

//------------------------------------------------------------------------
// get a node for a given name, instance and hostname
//------------------------------------------------------------------------
MENode *MENodeListHandler::getNode(int id)
{
    foreach (MENode *nptr, nodeList)
    {
        if (nptr->getNodeID() == id)
            return (nptr);
    }

    return (NULL);
}
