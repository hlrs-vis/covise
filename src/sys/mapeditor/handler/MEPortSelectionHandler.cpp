/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QDebug>
#include <QMenu>
#include <QGraphicsSceneContextMenuEvent>

#include "MEPortSelectionHandler.h"
#include "MEMainHandler.h"
#include "MENodeListHandler.h"
#include "widgets/MEGraphicsView.h"
#include "nodes/MENode.h"
#include "ports/MEDataPort.h"

/*!
   \class MEPortSelectionHandler
   \brief Class handles all requests concerning selected ports

*/

MEPortSelectionHandler::MEPortSelectionHandler()
    : QObject()
    , m_portPopup(NULL)
    , m_clickedPort(NULL)
{
}

MEPortSelectionHandler *MEPortSelectionHandler::instance()
{
    static MEPortSelectionHandler *singleton = 0;
    if (singleton == 0)
        singleton = new MEPortSelectionHandler();

    return singleton;
}

//!
//! reset all lists
//!
MEPortSelectionHandler::~MEPortSelectionHandler()
{
    clear(HoverConnectable);
    clear(HoverConnected);
    clear(Clicked);
    clear(Connectable);
}

//!
//! look if list is empty
//!
bool MEPortSelectionHandler::isEmpty(MEPortSelectionHandler::Type type)
{
    return m_selectedPortList[type].isEmpty();
}

//!
//! add a selected port
//!
void MEPortSelectionHandler::addPort(MEPortSelectionHandler::Type type, MEPort *port)
{
    if (!m_selectedPortList[type].contains(port))
        m_selectedPortList[type].append(port);

    if (type == Clicked)
        port->setSelected(MEPort::Clicked);
    else if (type == Connectable)
        port->setSelected(MEPort::Target);
    else if (type == HoverConnectable || type == HoverConnected)
        port->setSelected(MEPort::Highlight);
}

//!
//! remove a selected port
//!
void MEPortSelectionHandler::removePort(MEPortSelectionHandler::Type type, MEPort *port)
{
    static QMap<int, MEPort::SelectionType> selectionmap;
    if (selectionmap.isEmpty())
    {
        selectionmap[HoverConnectable] = MEPort::Highlight;
        selectionmap[HoverConnected] = MEPort::Highlight;
        selectionmap[Clicked] = MEPort::Clicked;
        selectionmap[Connectable] = MEPort::Target;
    }

    int idx = m_selectedPortList[type].indexOf(port);
    if (idx >= 0)
        m_selectedPortList[type].remove(idx);

    MEPort::SelectionType selectiontype = selectionmap[type];
    bool select = false;
    for (int i = 0; i < NumSelections; ++i)
    {
        if (i == type)
            continue;
        if (selectionmap[i] == selectiontype
            && m_selectedPortList[i].contains(port))
        {

            select = true;
            break;
        }
    }
    port->setSelected(selectiontype, select);
}

//!
// look if the given port is inside the list
//!
bool MEPortSelectionHandler::contains(MEPortSelectionHandler::Type type, MEPort *port)
{
    return m_selectedPortList[type].contains(port);
}

//!
//! return number of ports in list
//!
int MEPortSelectionHandler::count(MEPortSelectionHandler::Type type)
{
    return m_selectedPortList[type].count();
}

//!
//! clear all list and reset some state
//!
void MEPortSelectionHandler::clear(MEPortSelectionHandler::Type type)
{
    while (!m_selectedPortList[type].isEmpty())
    {
        removePort(type, m_selectedPortList[type][0]);
    }

    m_selectedPortList[type].clear();
    if (type == Connectable)
    {
        m_translate.clear();
        m_popupItems.clear();
    }
}

//!
//! right mouse button click , show a list of matching ports, event from MEPort class
//!
void MEPortSelectionHandler::showPossiblePorts(MEDataPort *port, QGraphicsSceneContextMenuEvent *e)
{

    // init a popup menu for port connection

    if (m_portPopup)
        delete m_portPopup;
    m_portPopup = new QMenu(0);

    m_portConnectionList.clear();
    m_clickedPort = port;

#ifndef YAC

    // show the current dataobject type if available

    QString text = port->getDataNameList();
    QAction *ac = m_portPopup->addAction(text);
    ac->setFont(MEMainHandler::s_boldFont);
    m_portPopup->addSeparator();
#endif

    // show possible ports for a connection

    int portType = port->getPortType();
    if ((portType == MEPort::DIN && port->getNoOfLinks() == 0) ||

#ifdef YAC
        portType == MEPort::MULTY_IN ||
#endif
        portType == MEPort::DOUT || port->getNode()->getCategory() == "Renderer")
    {

        // reset lists and state

        MEGraphicsView::instance()->clearPortSelections(HoverConnectable);

        // search for matching ports

        MENodeListHandler::instance()->searchMatchingPorts(HoverConnectable, m_clickedPort);

        // no matching port found at all

        if (m_selectedPortList[HoverConnectable].isEmpty())
            MEGraphicsView::instance()->clearPortSelections(HoverConnectable);

        // build popup menu

        else
        {
            // build the translation map (port<->text)
            // used for sorting items

            if (port->isConnectable())
            {
                for ( int i = 0; i < m_selectedPortList[HoverConnectable].count(); i++)
                {
                    // create string out of infos
                    MEPort *p = m_selectedPortList[HoverConnectable].at(i);
                    QString tmp = p->getNode()->getNodeTitle() + "::" + p->getName();

                    // add item and corresponding port to dictionary
                    m_translate.insert(tmp, p);

                    // add items to show list
                    m_popupItems << tmp;
                }

                // sort popup items
                m_popupItems.sort();

                // fill popup menu
                for ( int i = 0; i < m_popupItems.count(); i++)
                {
                    QAction *ac = m_portPopup->addAction(m_popupItems[i]);
                    m_portConnectionList.append(ac);
                    connect(ac, SIGNAL(triggered()), this, SLOT(connectPortCB()));
                }
            }
        }
    }

// add callback for showing data content

#ifndef YAC

    if (MEMainHandler::instance()->cfg_DeveloperMode)
    {
        m_portPopup->addSeparator();
        if (portType == MEPort::DIN)
        {
            foreach (MEDataPort *dp, port->connectedPorts)
            {
                if (dp->getDataObject())
                {
                    QAction *ac = m_portPopup->addAction(QString("View: " + dp->getPortObjectName()));
                    connect(ac, SIGNAL(triggered()), dp, SLOT(showDataContent()));
                }
            }
        }

        else if (port->getDataObject())
        {
            QAction *ac = m_portPopup->addAction(QString("View: " + port->getPortObjectName()));
            connect(ac, SIGNAL(triggered()), m_clickedPort, SLOT(showDataContent()));
        }
    }
#endif

    // show popup
    m_portPopup->exec(e->screenPos());
}

//!
//! a port from the popuplist was selected for connection
//!
void MEPortSelectionHandler::connectPortCB()
{
    // send line connection to controller
    // reset ports

    // object that sent the signal
    QAction *ac = (QAction *)sender();

    // find position in list
    int id = m_portConnectionList.indexOf(ac);

    // get port to corresponding text
    MEPort *port = m_translate.value(m_popupItems[id]);
    MEGraphicsView::instance()->sendLine(m_clickedPort, port);
    MEGraphicsView::instance()->clearPortSelections(MEPortSelectionHandler::Connectable);
}
