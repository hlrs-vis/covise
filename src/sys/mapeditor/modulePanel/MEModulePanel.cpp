/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QPushButton>
#include <QAction>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QCloseEvent>
#include <QDebug>

#include "MEModulePanel.h"
#include "MEModuleParameterLine.h"
#include "MEModuleParameter.h"
#include "widgets/MEUserInterface.h"
#include "handler/MEMainHandler.h"
#include "nodes/MENode.h"
#include "ports/MEParameterPort.h"

/*!
    \class MEModulePanel
    \brief Main dialog for the interaction with module parameters

    Provides a scroll area with a tab widget. <br>
    Creates a MEParameterWindow for each module and inserts it into the tab widget.
*/

MEModulePanel::MEModulePanel(QWidget *parent)
    : QDialog(parent)
    , m_firsttime(true)
{
    // set icon & text
    setWindowIcon(MEMainHandler::instance()->pm_logo);
}

MEModulePanel::~MEModulePanel()
{
}

MEModulePanel *MEModulePanel::instance()
{
    static MEModulePanel *singleton = 0;
    if (singleton == 0)
        singleton = new MEModulePanel(MEUserInterface::instance());

    return singleton;
}

#define addButton(pb, text, tooltip, callback, role)        \
    pb = new QPushButton(text, this);                       \
    pb->setToolTip(tooltip);                                \
    connect(pb, SIGNAL(clicked()), this, SLOT(callback())); \
    buttonBox->addButton(pb, role);

//!
//! create layout and push buttons
//!
void MEModulePanel::init()
//------------------------------------------------------------------------
{
    // set icon
    setWindowIcon(MEMainHandler::instance()->pm_logo);
    setWindowTitle("Module Parameters");

    // make the main layout for this windows
    QVBoxLayout *main = new QVBoxLayout();
    main->setMargin(2);
    main->setSpacing(2);

    // make a scrollview main window that contains a tabwidget
    m_tabWidget = new QTabWidget();
    connect(m_tabWidget, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

    QDialogButtonBox *buttonBox = new QDialogButtonBox();

    addButton(m_helpPB, "&Help", "Show help for this module", helpCB, QDialogButtonBox::HelpRole);
    addButton(m_detailPB, "&Details >>", "Show more details for parameters", detailCB, QDialogButtonBox::ActionRole);
    addButton(m_executePB, "&Execute", "Execute pipeline beginning with this module", execCB, QDialogButtonBox::ActionRole);
    addButton(m_cancelPB, "&Discard", "Cancel all parameter modifications since opening this window", cancelCB, QDialogButtonBox::RejectRole);
    addButton(m_okPB, "&Close", "Close this tab", okCB, QDialogButtonBox::AcceptRole);

    m_executePB->setEnabled(!MEMainHandler::instance()->isExecOnChange());

    // user can close the window pressing ESC
    QAction *m_escape_a = new QAction("Escape", this);
    m_escape_a->setShortcut(Qt::Key_Escape);
    connect(m_escape_a, SIGNAL(triggered()), this, SLOT(cancelCB()));
    addAction(m_escape_a);

    QAction *closeAction = new QAction("close", this);
    closeAction->setShortcut(QKeySequence::Close);
    connect(closeAction, SIGNAL(triggered(bool)), this, SLOT(close()));
    this->addAction(closeAction);

    // add widgets & layouts
    main->addWidget(m_tabWidget);
    main->addWidget(buttonBox);

    setLayout(main);

    setMasterState(MEMainHandler::instance()->isMaster());
}

void MEModulePanel::setCancelButtonState(bool state)
{
    m_cancelPB->setEnabled(state);
}

void MEModulePanel::setDetailText(const QString &text)
{
    m_detailPB->setText(text);
}

//!
//! Catch an user close (x)
//!
void MEModulePanel::closeEvent(QCloseEvent *e)
{
    closeWindow();
    m_firsttime = true;
    e->accept();
}

//!
//! Last window was closed, hide dialog window
//!
void MEModulePanel::closeLastWindow()
{
    m_firsttime = true;
    hide();
}

//!
//! Reset push button text & state
//!
void MEModulePanel::resetButtons(MENode *node)
{
    if (node->isShownDetails())
        m_detailPB->setText("<< &Details");
    else
        m_detailPB->setText("&Details >>");

    m_cancelPB->setEnabled(node->hasBeenModified());

    show();
}

//!
//! Add a new module parameter window to the tabs
//!
MEModuleParameter *MEModulePanel::addModuleInfo(MENode *node)
{

    MEModuleParameter *info = new MEModuleParameter(this, node);
    connect(info, SIGNAL(disableDiscard(bool)), this, SLOT(paramChanged(bool)));
    showModuleInfo(info);
    raise();

    int width = info->sizeHint().width() + 70;
    int height = info->sizeHint().height() + 50;
    fitWindow(width, height);

    return info;
}

//!
//! Change the tab text for a module parameter window
//!
void MEModulePanel::changeModuleInfoTitle(MEModuleParameter *info, const QString &text)
{
    m_tabWidget->setTabText(m_tabWidget->indexOf(info), text);
}

//!
//! Switch the master/slave state
//!
void MEModulePanel::setMasterState(bool state)
{
    m_executePB->setEnabled(state);
    m_cancelPB->setEnabled(state);
    m_okPB->setEnabled(state);

    for (unsigned int i = 0; i < (unsigned int)(m_tabWidget->count()); i++)
        m_tabWidget->widget(i)->setEnabled(state);
}

//!
//! Fit window size if reasonable
//!
void MEModulePanel::fitWindow(int w, int h)
{

    w = w + 50;
    h = h + 85; // plus height of buttons

    // show new window with original size
    // but not greater than screen size
    int oldW = m_firsttime ? 0 : width();
    int oldH = m_firsttime ? 0 : height();

    m_firsttime = false;

    int ww = qMax(oldW, w);
    int hh = qMax(oldH, h);
    ww = qMin(ww, MEMainHandler::screenSize.width() - 50);
    hh = qMin(hh, MEMainHandler::screenSize.height() - 50);

    if (width() != ww || height() != hh)
    {
        resize(ww, hh);
        //update();
    }
}

//!
//! Disable/enable local execute mode
//!
void MEModulePanel::enableExecCB(bool state)
{
    if (state)
        m_executePB->setEnabled(false);
    else
        m_executePB->setEnabled(true);
}

//!
//! Close all tabs, close node bookicons
//!
void MEModulePanel::closeWindow()
{
    while (m_tabWidget->currentWidget())
    {
        MEModuleParameter *info = (MEModuleParameter *)m_tabWidget->currentWidget();
        info->getNode()->bookClick();
    }
}

//!
//! Hide the module info, also hide parent window if it contains nothing
//!
void MEModulePanel::hideModuleInfo(MEModuleParameter *info)
{
    // show the right button text & state
    resetButtons(info->getNode());

    // hide info
    m_tabWidget->removeTab(m_tabWidget->indexOf(info));
    if (m_tabWidget->count() == 0)
        closeLastWindow();
}

//!
//! Show the module info
//!
void MEModulePanel::showModuleInfo(MEModuleParameter *info)
{
    // show the right button text & state
    resetButtons(info->getNode());

    // store current values for restore action
    if (!m_tabWidget->isVisible())
        info->getNode()->storeParameter();

    // show info
    m_tabWidget->addTab(info, info->getNode()->getNodeTitle());
    m_tabWidget->setCurrentIndex(m_tabWidget->indexOf(info));
}

//!
//! Disable/enable cancel button
//!
void MEModulePanel::paramChanged(bool state)
{
    setCancelButtonState(state);
}

//!
//! User has selected another tab, reset dialog buttons & state
//!
void MEModulePanel::tabChanged(int index)
{
    if (index < 0)
        return;
    MEModuleParameter *info = (MEModuleParameter *)m_tabWidget->widget(index);
    resetButtons(info->getNode());
}

//!
//! Node help callback
//!
void MEModulePanel::helpCB()
{
    MEModuleParameter *info = (MEModuleParameter *)m_tabWidget->currentWidget();
    info->getNode()->helpCB();
}

//!
//! Close node book icon
//!------------------------------------------------------------------------
void MEModulePanel::okCB()
{
    MEModuleParameter *info = (MEModuleParameter *)m_tabWidget->currentWidget();
    info->getNode()->bookClick();
}

//!
//! Cancel callback, restore old stored values, disable DISCARD button
//!
void MEModulePanel::cancelCB()
{
    MEModuleParameter *info = (MEModuleParameter *)m_tabWidget->currentWidget();
    MENode *node = info->getNode();
    node->restoreParameter();
    node->setModified(false);
    node->bookClick();
}

//!
//! Node execution callback
//!
void MEModulePanel::execCB()
{
    MEModuleParameter *info = (MEModuleParameter *)m_tabWidget->currentWidget();
    MENode *node = info->getNode();
    node->storeParameter();
    node->executeCB();
}

//!
//! Node detail callback, show the second line of a parameter, used for slider and scalar
//!
void MEModulePanel::detailCB()
{
    MEModuleParameter *info = (MEModuleParameter *)m_tabWidget->currentWidget();
    MENode *node = info->getNode();

    if (node)
    {
        if (node->isShownDetails())
        {
            setDetailText("Details >>");
            node->setShowDetails(false);

            foreach (MEParameterPort *port, node->pinlist)
            {
                if (port->getParamType() == MEParameterPort::T_FLOATSLIDER || port->getParamType() == MEParameterPort::T_INTSLIDER || port->getParamType() == MEParameterPort::T_INT || port->getParamType() == MEParameterPort::T_FLOAT)
                {
                    if (port->getSecondLine())
                        port->getSecondLine()->hide();
                }
            }
        }

        else
        {
            setDetailText("<< Details");
            node->setShowDetails(true);

            foreach (MEParameterPort *port, node->pinlist)
            {
                if (port->getParamType() == MEParameterPort::T_FLOATSLIDER || port->getParamType() == MEParameterPort::T_INTSLIDER || port->getParamType() == MEParameterPort::T_INT || port->getParamType() == MEParameterPort::T_FLOAT)
                {
                    if (port->getSecondLine())
                        port->getSecondLine()->show();
                }
            }
        }
    }
}
