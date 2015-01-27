/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QVBoxLayout>
#include <QPushButton>

#include "MEModuleParameterLine.h"
#include "MEModuleParameter.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "nodes/MENode.h"
#include "ports/MEParameterPort.h"
#include "ports/MEParameterAppearance.h"

#define addIcon(pb, pixmap, receiver, callback, tooltip)    \
    pb = new QPushButton(m_textFrame);                      \
    pb->setIcon(pixmap);                                    \
    pb->setFlat(true);                                      \
    pb->setAutoDefault(false);                              \
    pb->setToolTip(tooltip);                                \
    hb1->addWidget(pb);                                     \
    connect(pb, SIGNAL(clicked()), this, SLOT(callback())); \
    connect(pb, SIGNAL(clicked()), pb, SLOT(setFocus()));

/*!
    \class MEModuleParameterLine
    \brief Widget shows a single parameter port

    Used in MEModuleParameter
*/

MEModuleParameterLine::MEModuleParameterLine(MEParameterPort *port, QFrame *frame, QWidget *widget)
    : QObject()
    , m_port(port)
    , m_appearanceTypes(NULL)
    , m_textFrame(frame)
    , m_container(widget)
    , m_secondLine(NULL)
    , m_mappedPB(NULL)
    , m_lightPB(NULL)
{

    // m_textFrame contains for each port pixmaps & parameter name
    // m_container contains the widgets for a port parameter
    QVBoxLayout *vb1 = new QVBoxLayout(m_textFrame);
    vb1->setMargin(2);
    vb1->setSpacing(2);

    // create layout for pixmaps and name
    QBoxLayout *hb1 = new QHBoxLayout();
    vb1->addLayout(hb1);
    hb1->setMargin(2);
    hb1->setSpacing(2);

    // pinCol -- set pin pixmap
    addIcon(m_mappedPB, MEMainHandler::instance()->pm_pinup, this, mappedCB, "Click on this icon to map/unmap\nthis parameter to/from the Control Panel");
    m_mappedPB->setFocusPolicy(Qt::NoFocus);
    if (m_port->isMapped())
        m_mappedPB->setIcon(MEMainHandler::instance()->pm_pindown);

#ifdef YAC
    // visibleCol -- set light pixmap
    addIcon(m_lightPB, MEMainHandler::instance()->pm_lightoff, this, lightCB, "Click on this icon to show/hide\nthe module icon port in the canvas area");
    m_lightPB->setFocusPolicy(Qt::NoFocus);
    if (m_port->isShown())
        m_lightPB->setIcon(MEMainHandler::instance()->pm_lighton);
#endif

    // nameCol -- set port name
    m_appearanceTypes = new MEParameterAppearance(m_textFrame, m_port);
    hb1->addWidget(m_appearanceTypes);

    // add only a stretching factor
    hb1->addStretch(1);
    vb1->addStretch(1);

    // set appearance & values in m_conatiner
    m_port->makeLayout(MEParameterPort::MODULE, m_container);

#ifndef YAC
    m_textFrame->setEnabled(m_port->isEnabled());
    m_container->setEnabled(m_port->isEnabled());
#endif
}

MEModuleParameterLine::~MEModuleParameterLine()
{
    // nothing necessary
    // all things are killed by Qt
}

//!
//! change mapped pixmap, called from MEMessageHandler
//!
void MEModuleParameterLine::changeMappedPixmap(bool on)
{
    if (m_mappedPB)
    {
        if (on)
            m_mappedPB->setIcon(MEMainHandler::instance()->pm_pindown);
        else
            m_mappedPB->setIcon(MEMainHandler::instance()->pm_pinup);
    }
}

//!
//! change light pixmap, called from MEMessageHandler
//!
void MEModuleParameterLine::changeLightPixmap(bool on)
{
    if (m_lightPB)
    {
        if (on)
            m_lightPB->setIcon(MEMainHandler::instance()->pm_lighton);
        else
            m_lightPB->setIcon(MEMainHandler::instance()->pm_lightoff);
    }
}

//!
//! light pixmap was clicked, only used for YAC
//!
void MEModuleParameterLine::lightCB()
{
#ifdef YAC
    covise::coSendBuffer sb;
    sb << m_port->getNode()->getNodeID() << m_port->getName();
    if (m_port->isShown())
        sb << 0;
    else
        sb << 1;
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_PORT_VISIBLE, sb);
#endif
}

//!
//! map/unmap pixmap was clicked
//!
void MEModuleParameterLine::mappedCB()
{
#ifdef YAC

    covise::coSendBuffer sb;
    sb << m_port->getNode()->getNodeID() << m_port->getName();
    if (m_port->isMapped())
        sb << 0;
    else
        sb << 1;
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_PORT_MAPPED, sb);

#else

    if (m_port->isMapped())
        m_port->sendPanelMessage("RM_PANEL");

    else
        m_port->sendPanelMessage("ADD_PANEL");
#endif

    MEMainHandler::instance()->mapWasChanged("RM/ADD_PANEL");
}

//!
//! enable/disable a parameter in the module parameter window
//!
void MEModuleParameterLine::setEnabled(bool sensitive)
{
    m_textFrame->setEnabled(sensitive);
    m_container->setEnabled(sensitive);
}

//!
//! change the color for activated lines
//!
void MEModuleParameterLine::colorTextFrame(bool on)
{
    QPalette palette;
    if (on)
        palette.setColor(QPalette::Window, m_port->getNode()->getColor());

    else
        palette.setColor(QPalette::Window, MEMainHandler::defaultPalette.color(QPalette::Window));

    m_textFrame->setPalette(palette);
    m_textFrame->setAutoFillBackground(true);
}
