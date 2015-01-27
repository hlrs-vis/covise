/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   31.03.2010
**
**************************************************************************/

#include "zoomtool.hpp"

#include "toolmanager.hpp"
#include "src/mainwindow.hpp"

// Qt //
//
#include <QWidget>
#include <QToolBar>
#include <QAction>
#include <QComboBox>
#include <QMenu>
#include <QLabel>

//################//
//                //
// ZoomTool       //
//                //
//################//

ZoomTool::ZoomTool(ToolManager *toolManager)
    : Tool(toolManager)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    QLabel *zoomToLabel = new QLabel("Zoom to: ");

    zoomComboBox_ = new QComboBox();
    QStringList zoomFactors;
    zoomFactors << tr("25%") << tr("50%") << tr("100%") << tr("200%") << tr("400%") << tr("800%");
    zoomComboBox_->addItems(zoomFactors);
    zoomComboBox_->setCurrentIndex(2);
    zoomComboBox_->setStatusTip(tr("Zoom to."));
    zoomComboBox_->setToolTip(tr("Zoom To"));
    connect(zoomComboBox_, SIGNAL(currentIndexChanged(QString)), this, SLOT(zoomTo(QString)));

    zoomInAction_ = new QAction(tr("Zoom &In"), this);
    zoomInAction_->setShortcuts(QKeySequence::ZoomIn);
    zoomInAction_->setStatusTip(tr("Zoom in."));
    connect(zoomInAction_, SIGNAL(triggered()), this, SLOT(zoomIn()));

    zoomOutAction_ = new QAction(tr("Zoom &Out"), this);
    zoomOutAction_->setShortcuts(QKeySequence::ZoomOut);
    zoomOutAction_->setStatusTip(tr("Zoom out."));
    connect(zoomOutAction_, SIGNAL(triggered()), this, SLOT(zoomOut()));

    //	zoomBoxAction = new QAction(tr("Zoom &Box"), this);
    //	//zoomBoxAction->setShortcuts();
    //	zoomBoxAction->setStatusTip(tr("Draw a box to zoom in."));
    //	connect(zoomBoxAction, SIGNAL(triggered()), this, SLOT(zoomBox()));

    viewSelectedAction_ = new QAction(tr("View &Selected"), this);
    viewSelectedAction_->setStatusTip(tr("View selected"));
    connect(viewSelectedAction_, SIGNAL(triggered()), this, SLOT(viewSelected()));

    // Select inverse //
    //
    selectInverseAction_ = new QAction(tr("Select inverse"), this);
    selectInverseAction_->setStatusTip(tr("Select inverse"));
    selectInverseAction_->setShortcut(tr("Ctrl+I"));
    connect(selectInverseAction_, SIGNAL(triggered()), SLOT(selectInverse()));

    // Hiding //
    //
    hideSelectedAction_ = new QAction(tr("Hide selected"), this);
    hideSelectedAction_->setStatusTip(tr("Hide all selected elements."));
    hideSelectedAction_->setShortcut(tr("Ctrl+H"));
    connect(hideSelectedAction_, SIGNAL(triggered()), SLOT(hideSelected()));

    hideSelectedRoadsAction_ = new QAction(tr("Hide selected roads"), this);
    hideSelectedRoadsAction_->setStatusTip(tr("Hide all selected roads."));
    hideSelectedRoadsAction_->setShortcut(tr("Ctrl+R"));
    connect(hideSelectedRoadsAction_, SIGNAL(triggered()), SLOT(hideSelectedRoads()));

    hideDeselectedAction_ = new QAction(tr("Hide deselected"), this);
    hideDeselectedAction_->setStatusTip(tr("Hide all deselected elements."));
    connect(hideDeselectedAction_, SIGNAL(triggered()), SLOT(hideDeselected()));

    unhideAllAction_ = new QAction(tr("Unhide all"), this);
    unhideAllAction_->setStatusTip(tr("Show all hidden elements."));
    unhideAllAction_->setShortcut(tr("Ctrl+Shift+H"));
    connect(unhideAllAction_, SIGNAL(triggered()), SLOT(unhideAll()));

    // Ruler //
    //
    rulerAction_ = new QAction(tr("Show/Hide &Rulers"), this);
    rulerAction_->setStatusTip(tr("Toggle Rulers."));
    rulerAction_->setCheckable(true);
    rulerAction_->setChecked(false);
    connect(rulerAction_, SIGNAL(triggered(bool)), this, SLOT(activateRulers(bool)));

    // Deactivate if no project //
    //
    connect(ODD::instance()->mainWindow(), SIGNAL(hasActiveProject(bool)), this, SLOT(activateProject(bool)));

    // ToolBar //
    //
    QToolBar *zoomToolBar = new QToolBar(tr("Zoom"));
    zoomToolBar->addWidget(zoomToLabel);
    zoomToolBar->addWidget(zoomComboBox_);
    zoomToolBar->addAction(zoomInAction_);
    zoomToolBar->addAction(zoomOutAction_);
    //zoomToolBar->addAction(zoomBoxAction);
    zoomToolBar->addAction(viewSelectedAction_);

    // ToolManager //
    //
    //	toolManager->addToolBar(zoomToolBar);
    ODD::instance()->mainWindow()->addToolBar(zoomToolBar);
    //	ODD::instance()->mainWindow()->addToolBarBreak();

    // View Menu //
    //
    QMenu *viewMenu = ODD::instance()->mainWindow()->getViewMenu();

    viewMenu->addSeparator();
    viewMenu->addAction(zoomInAction_);
    viewMenu->addAction(zoomOutAction_);
    //viewMenu_->addAction(zoomBoxAction);

    viewMenu->addSeparator();
    viewMenu->addAction(selectInverseAction_);

    viewMenu->addSeparator();
    viewMenu->addAction(hideSelectedAction_);
    viewMenu->addAction(hideSelectedRoadsAction_);
    //	viewMenu->addAction(hideDeselectedAction_);
    viewMenu->addAction(unhideAllAction_);

    viewMenu->addSeparator();
    viewMenu->addAction(rulerAction_);
}

//################//
// SLOTS          //
//################//

/*! \brief.
*/
void
ZoomTool::activateProject(bool active)
{
    // Enable/Disable //
    //
    zoomComboBox_->setEnabled(active);
    zoomInAction_->setEnabled(active);
    zoomOutAction_->setEnabled(active);
    viewSelectedAction_->setEnabled(active);
    rulerAction_->setEnabled(active);
    selectInverseAction_->setEnabled(active);

    hideSelectedAction_->setEnabled(active);
    hideSelectedRoadsAction_->setEnabled(active);
    hideDeselectedAction_->setEnabled(active);
    unhideAllAction_->setEnabled(active);

    // Rulers //
    //
    activateRulers(rulerAction_->isChecked());
}

/*! \brief.
*/
void
ZoomTool::zoomTo(const QString &zoomFactor)
{
    ZoomToolAction *action = new ZoomToolAction(zoomFactor);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::zoomIn()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_ZOOMIN);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::zoomOut()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_ZOOMOUT);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::zoomBox()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_ZOOMBOX);
    emit toolAction(action);
    delete action;
}
/*!
*/
void
ZoomTool::viewSelected()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_VIEW_SELECTED);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::selectInverse()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_SELECT_INVERSE);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::hideSelected()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_HIDE_SELECTED);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::hideSelectedRoads()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_HIDE_SELECTED_ROADS);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::hideDeselected()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_HIDE_DESELECTED);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::unhideAll()
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_UNHIDE_ALL);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
ZoomTool::activateRulers(bool active)
{
    ZoomToolAction *action = new ZoomToolAction(ZoomTool::TZM_RULERS, active);
    emit toolAction(action);
    delete action;
}

//################//
//                //
// ZoomToolAction //
//                //
//################//

// Note: This is not a typical Editor/Tool combination since this is not bound to
// a specify editor! So ENO_EDITOR and TNO_TOOL is set (Otherwise an editor would
// be loaded).

ZoomToolAction::ZoomToolAction(ZoomTool::ZoomToolId zoomToolId, bool toggled)
    : ToolAction(ODD::ENO_EDITOR, ODD::TNO_TOOL)
    , zoomToolId_(zoomToolId)
    , zoomFactor_("")
    , toggled_(toggled)
{
}

ZoomToolAction::ZoomToolAction(const QString &zoomFactor)
    : ToolAction(ODD::ENO_EDITOR, ODD::TNO_TOOL)
    , zoomToolId_(ZoomTool::TZM_ZOOMTO)
    , zoomFactor_(zoomFactor)
{
}
