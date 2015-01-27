/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QLabel>
#include <QGridLayout>
#include <QVBoxLayout>

#include "MEModuleParameter.h"
#include "MEModuleParameterLine.h"
#include "handler/MEMainHandler.h"
#include "nodes/MENode.h"
#include "ports/MEParameterPort.h"

/*!
    \class MEModuleParameter
    \brief Main widget for all parameters of a module

    MEModuleParameterLine shows each module parameter. <br>
    Used in MEModulePanel
*/

MEModuleParameter::MEModuleParameter(QWidget *parent, MENode *node)
    : QScrollArea(parent)
    , m_node(node)
{
    // create the content widget
    m_main = new QWidget();
    setWidgetResizable(true);
    setWidget(m_main);

    // create the main layout
    QVBoxLayout *main = new QVBoxLayout(m_main);
    main->setMargin(2);
    main->setSpacing(2);

    // create an info line at the top containing module description
    QLabel *label = new QLabel(m_node->getDescription(), m_main);
    label->setFont(MEMainHandler::s_boldFont);
    label->setAutoFillBackground(true);
    label->setFixedHeight(label->minimumSizeHint().height() + 10);
    QPalette palette;
    palette.setBrush(backgroundRole(), m_node->getColor());
    label->setPalette(palette);
    label->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
    main->addWidget(label);

    // create a grid layout for module parameter information
    // disable stretching of first column
    QGridLayout *grid = new QGridLayout();
    grid->setColumnStretch(0, 0);
    grid->setColumnStretch(1, 1);
    main->addLayout(grid);

    int row = 0;
    foreach (MEParameterPort *port, m_node->pinlist)
    {
        // create a container widget with layout for 2 possible rows
        // the 2. column is only used for stretching so that lables keep their position
        QFrame *textFrame = new QFrame(m_main);
        textFrame->setFrameStyle(QFrame::NoFrame);
        QWidget *contentFrame = new QWidget(m_main);

        grid->addWidget(textFrame, row, 0);
        grid->addWidget(contentFrame, row, 1);

        // appCol & valCol -- set the appearance and init some parameter
        port->createParameterLine(textFrame, contentFrame);

        row++;
    }

    // add a stretching value to the end
    main->addStretch(1);

    // set the right master-slave state
    setEnabled(MEMainHandler::instance()->isMaster());
}

MEModuleParameter::~MEModuleParameter()
{
    // nothing necessary
    // all things are killed by Qt
}

//!
//! Inform main module parameter window that a parameter has been changed
//!
void MEModuleParameter::paramChanged(bool state)
{
    emit disableDiscard(state);
}
