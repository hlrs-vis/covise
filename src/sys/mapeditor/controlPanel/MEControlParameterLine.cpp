/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QHBoxLayout>
#include <QLabel>

#include "MEControlParameterLine.h"
#include "MEControlParameter.h"
#include "handler/MEMainHandler.h"
#include "nodes/MENode.h"
#include "ports/MEParameterPort.h"

/*!
    \class MEConMEControlParameterLinetrolParameter
    \brief Widget shows content of one module parameter
*/

MEControlParameterLine::MEControlParameterLine(QWidget *parent, MEParameterPort *port)
    : QWidget(parent)
    , m_port(port)
{

    // create a horizontal layout
    m_boxLayout = new QHBoxLayout(this);
    m_boxLayout->setMargin(2);
    m_boxLayout->setSpacing(2);

    // create the parameter text
    m_parameterName = new QLabel(m_port->getName(), this);
    m_parameterName->setFont(MEMainHandler::s_boldFont);
    QString tipText("<b>Type: </b>" + m_port->getParamTypeString() + "<br><i>" + m_port->getDescription() + "</i>");
    m_parameterName->setToolTip(tipText);

    // check length of label
    // perhaps a recalculate is needed
    int m_labelWidth = m_port->getNode()->getControlInfo()->getLabelWidth();
    if (m_parameterName->sizeHint().width() > m_labelWidth)
    {
        m_labelWidth = m_parameterName->sizeHint().width();
        emit recalculateSize(m_labelWidth);
    }
    m_parameterName->setFixedWidth(m_labelWidth);

    // add label to layout
    m_boxLayout->addWidget(m_parameterName);

    // make layout for single parameter
    makeLayout();

    // SIGNAL/SLOT for all control lines
    connect(this, SIGNAL(recalculateSize(int)), m_port->getNode()->getControlInfo(), SLOT(recalculate(int)));

    // show widget
    show();
}

MEControlParameterLine::~MEControlParameterLine()
//!
{
    /*   if(timer)
      {
         MEMainHandler::instance()->timerList.remove(MEMainHandler::instance()->timerList.indexOf(timer));
         delete timer;
      }*/
}

//!
//! change color of current parameter name
//!
void MEControlParameterLine::colorTextFrame(bool on)
{
    QPalette palette;
    if (on)
        palette.setColor(QPalette::Window, m_port->getNode()->getColor());

    else
        palette.setColor(QPalette::Window, MEMainHandler::defaultPalette.color(QPalette::Window));

    m_parameterName->setPalette(palette);
    m_parameterName->setAutoFillBackground(true);
}

//!
//! make a layout depending on parameter type
//!
void MEControlParameterLine::makeLayout()
{
    QWidget *w = new QWidget(this);
    m_boxLayout->addWidget(w);
    m_port->makeLayout(MEParameterPort::CONTROL, w);
}
