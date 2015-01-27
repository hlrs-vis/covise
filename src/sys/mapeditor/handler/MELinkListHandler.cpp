/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "MELinkListHandler.h"
#include "ports/MEDataPort.h"

/*!
   \class MELinkListHandler
   \brief This class handles the list of connection lines
*/

MELinkListHandler::MELinkListHandler()
    : QObject()
{
}

MELinkListHandler *MELinkListHandler::instance()
{
    static MELinkListHandler *singleton = 0;
    if (singleton == 0)
        singleton = new MELinkListHandler();

    return singleton;
}

MELinkListHandler::~MELinkListHandler()
{
    clearList();
}

//!
// !clear list
//!
void MELinkListHandler::clearList()
{
    qDeleteAll(linkList);
    linkList.clear();
}

//!
//! add a new connection
//!
void MELinkListHandler::addLink(MENode *, MEPort *p1, MENode *, MEPort *p2)
{
    MENodeLink *link = new MENodeLink(p1, p2);
    linkList.append(link);
}

//!
//! delete a new connection
//!
void MELinkListHandler::deleteLink(MENode *n1, MEPort *p1, MENode *n2, MEPort *p2)
{
    MENodeLink *link = getLink(n1, p1, n2, p2);
    linkList.remove(linkList.indexOf(link));
    delete link;
}

//!
//! node was moved, remove old lines & create new one
//!
void MELinkListHandler::resetLinks(MENode *node)
{
    foreach (MENodeLink *nptr, linkList)
    {
        if (nptr->port1->getNode() == node || nptr->port2->getNode() == node)
            nptr->moveLines();
    }
}

//!
//! redraw all links for a given node (node was hidden)
//!
void MELinkListHandler::highlightPortAndLinks(MEPort *port, bool state)
{

    foreach (MENodeLink *link, linkList)
    {
        if (link->port1 == port || link->port2 == port)
            link->highlightLines(state);
    }
}

//!
//! get all links for a given output port
//!
QVector<MENodeLink *> MELinkListHandler::getLinksOut(MEPort *port)
{
    QVector<MENodeLink *> links;

    foreach (MENodeLink *link, linkList)
    {
        if (link->port1 == port)
            links.push_back(link);
    }

    return links;
}

//!
//! get all links for a given input port
//!
QVector<MENodeLink *> MELinkListHandler::getLinksIn(MEPort *port)
{
    QVector<MENodeLink *> links;

    foreach (MENodeLink *link, linkList)
    {
        if (link->port2 == port)
            links.push_back(link);
    }

    return links;
}

//!
//! remove all links for a given node
//!
void MELinkListHandler::removeLinks(MENode *node)
{
    QVector<MENodeLink *> tmplist;

    // look for all links containing this node & store it
    foreach (MENodeLink *nptr, linkList)
    {
        if (nptr->port1->getNode() == node || nptr->port2->getNode() == node)
        {
            tmplist << nptr;
        }
    }

    // delete links
    foreach (MENodeLink *nptr, tmplist)
    {
        linkList.remove(linkList.indexOf(nptr));
        delete nptr;
    }

    tmplist.clear();
}

//!
//! get a link for two given ports
//!
MENodeLink *MELinkListHandler::getLink(MENode *n1, MEPort *p1, MENode *n2, MEPort *p2)
{

    foreach (MENodeLink *nptr, linkList)
    {
        if (nptr->port1 == p1 && nptr->port2 == p2 && nptr->port1->getNode() == n1 && nptr->port2->getNode() == n2)
        {
            return (nptr);
        }
        if (nptr->port1 == p2 && nptr->port2 == p1 && nptr->port1->getNode() == n2 && nptr->port2->getNode() == n1)
        {
            return (nptr);
        }
    }

    return (NULL);
}

//!
//! update tooltip for input ports after execution
//!
void MELinkListHandler::updateInputDataPorts(MEPort *port)
{

    MEDataPort *dp = qobject_cast<MEDataPort *>(port);
    if (dp)
    {

        // search port in connection list
        foreach (MENodeLink *nptr, linkList)
        {
            if (nptr->port1 == port)
            {
                MEDataPort *pto = static_cast<MEDataPort *>(nptr->port2);
                dp->getDataObjectInfo();

                // update help text
                QString text = pto->getDataNameList();
                QString tip = pto->getDescription() + "::" + text;
                pto->setToolTip(tip);
                QString help = pto->getDataTypes().join(";");
                pto->updateHelpText(help);
            }
        }
    }
}
