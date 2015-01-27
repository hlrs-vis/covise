/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include "MEControlPanel.h"
#include "MEControlParameter.h"
#include "MEControlParameterLine.h"
#include "widgets/MEUserInterface.h"
#include "handler/MEMainHandler.h"
#include "modulePanel/MEModulePanel.h"
#include "nodes/MENode.h"
#include "ports/MEParameterPort.h"

/*!
    \class MEControlParameter
    \brief Main container window for all parameter ports of one module
*/

MEControlParameter::MEControlParameter(MENode *node)
    : QFrame(0)
    , m_labelWidth(100)
    , m_node(node)
    , m_mainContent(NULL)
    , m_infoPB(NULL)
    , m_showPB(NULL)
    , m_execPB(NULL)
    , m_helpPB(NULL)
{

    setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);

    // create a vertical layout
    // contains a header & a container widget
    QVBoxLayout *vbox = new QVBoxLayout(this);
    vbox->setMargin(2);
    vbox->setSpacing(2);

    // create the info label widget
    vbox->addWidget(createHeader());

    // create a container widget & layout that contains certain ports
    m_mainContent = new QWidget(this);
    m_vlist = new QVBoxLayout(m_mainContent);
    vbox->addWidget(m_mainContent);
    vbox->addStretch(1);

    // catch  the book signals from module node
    connect(m_node, SIGNAL(bookIconChanged()), this, SLOT(bookCB()));

    setMasterState(MEMainHandler::instance()->isMaster());
}

//!
MEControlParameter::~MEControlParameter()
//!
{
    m_parameterList.clear();
    MEControlPanel::instance()->removeControlInfo(this);
}

#define addButton(widget, pixmap, box, receiver, callback, tooltip) \
    widget = new QPushButton(header);                               \
    widget->setIcon(pixmap);                                        \
    widget->setFlat(true);                                          \
    widget->setToolTip(tooltip);                                    \
    box->addWidget(widget);                                         \
    connect(widget, SIGNAL(clicked()), receiver, SLOT(callback())); \
    connect(widget, SIGNAL(clicked()), widget, SLOT(setFocus()));

//!
//! create the header content for each control info
//!
QFrame *MEControlParameter::createHeader()
{

    // create a widget with a horizontal layout
    // widget contains title and some global icons
    QFrame *header = new QFrame(this);
    header->setPalette(QPalette(m_node->getColor()));
    header->setFrameStyle(QFrame::Box | QFrame::Raised);
    header->setAutoFillBackground(true);

    // create the layout
    QHBoxLayout *hbox = new QHBoxLayout(header);
    hbox->setMargin(1);
    hbox->setSpacing(1);

    // add some icons to the header box
    // execute, help, show/hide
    if (MEUserInterface::instance()->hasMiniGUI())
    {
        addButton(m_showPB, MEMainHandler::instance()->pm_folderopen, hbox, this, showCB, "Show/hide parameters of this module");
        addButton(m_helpPB, MEMainHandler::instance()->pm_help, hbox, m_node, helpCB, "Open the module help");
    }

    // execute, modulinfo, help,  show/hide
    else
    {
        addButton(m_showPB, MEMainHandler::instance()->pm_folderopen, hbox, this, showCB, "Show/hide all parameters of this module");
        addButton(m_execPB, MEMainHandler::instance()->pm_exec, hbox, m_node, executeCB, "Execute the pipeline starting with this module");
        addButton(m_infoPB, MEMainHandler::instance()->pm_bookclosed, hbox, m_node, bookClick, "Open/Close the module parameter window");
        if (m_node->isBookOpen())
            m_infoPB->setIcon(MEMainHandler::instance()->pm_bookopen);
        connect(m_infoPB, SIGNAL(clicked()), this, SLOT(bookCB()));
        addButton(m_helpPB, MEMainHandler::instance()->pm_help, hbox, m_node, helpCB, "Open the module help");
    }
    m_fopen = true;

    // create title
    m_moduleTitle = new QLabel(m_node->getNodeTitle(), header);
    hbox->addWidget(m_moduleTitle);
    hbox->addStretch(1);

    return (header);
}

//!
//! set a new module title
//!
void MEControlParameter::setNodeTitle(const QString &text)
{
    m_moduleTitle->setText(text);
}

//!
//! switch master/slave state
//!
void MEControlParameter::setMasterState(bool state)
{
    if (m_execPB)
        m_execPB->setEnabled(state);
    m_mainContent->setEnabled(state);
}

//!
//! insert the parameter on the right position as the parameter are defined
//!
void MEControlParameter::insertParameter(MEControlParameterLine *line)
{
    // store parameter line
    m_parameterList.append(line);

    // find the right position
    int index = 0;
    foreach (MEParameterPort *port, m_node->pinlist)
    {
        if (port == line->getPort())
        {
            m_vlist->insertWidget(index, line);
            if (isHidden())
                show();
            return;
        }

        else
        {
            if (port->hasControlLine())
                index++;
        }
    }

    if (isHidden())
        show();
}

//!
//! remove the parameter from the control panel
//!
void MEControlParameter::removeParameter(MEControlParameterLine *line)
{
    // remove
    m_parameterList.remove(m_parameterList.indexOf(line));
    delete line;

    // if last parameter was removed hide window
    if (m_parameterList.isEmpty())
        hide();
}

//!
//! recalculate label size if width is to small
//!
void MEControlParameter::recalculate(int size)
{
    foreach (MEControlParameterLine *line, m_parameterList)
        line->getLabel()->setFixedWidth(size);

    m_labelWidth = size;
    adjustSize();
}

//!
//! open/close book icon if book icon in node has c
//!
void MEControlParameter::bookCB()
{
    if (m_node->isBookOpen())
    {
        m_infoPB->setIcon(MEMainHandler::instance()->pm_bookopen);
        if (m_node->getModuleInfo())
            MEModulePanel::instance()->raise();
    }

    else
        m_infoPB->setIcon(MEMainHandler::instance()->pm_bookclosed);
}

//!
//! show/hide control content for one module
//!
void MEControlParameter::showCB()
{
    if (m_fopen)
    {
        m_showPB->setIcon(MEMainHandler::instance()->pm_folderclosed);
        m_mainContent->hide();
        m_fopen = false;
    }

    else
    {
        m_showPB->setIcon(MEMainHandler::instance()->pm_folderopen);
        m_mainContent->show();
        m_fopen = true;
    }
}
