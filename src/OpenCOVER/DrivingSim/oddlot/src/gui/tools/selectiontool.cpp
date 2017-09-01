/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**
**************************************************************************/

#include "selectiontool.hpp"

#include "toolmanager.hpp"
#include "src/mainwindow.hpp"

// Qt //
//
#include <QWidget>
#include <QToolBar>
#include <QAction>
#include <QPushButton>
#include <QIcon>
#include <QLabel>
#include <QKeyEvent>
#include <QButtonGroup>

//################//
//                //
// SelectionTool       //
//                //
//################//

SelectionTool::SelectionTool(ToolManager *toolManager)
    : Tool(toolManager)
{

    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    // ButtonGroup //
    //
    // A button group so only one button can be checked at a time
    QButtonGroup *toolGroup = new QButtonGroup;
    connect(toolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));

    QLabel *selectionLabel = new QLabel("Selection: ");

    selectionBox_ = new QPushButton();
    selectionBox_->setIcon(QIcon(":/tools/boundingbox.png"));
    selectionBox_->setToolTip(tr("Bounding Box"));
    selectionBox_->setCheckable(false);

    toolGroup->addButton(selectionBox_, SelectionTool::TSL_BOUNDINGBOX);

    // Deactivate if no project //
    //
    connect(ODD::instance()->mainWindow(), SIGNAL(hasActiveProject(bool)), this, SLOT(activateProject(bool)));

    // ToolBar //
    //
    QToolBar *selectionToolBar = new QToolBar(tr("Select"));

    selectionToolBar->addWidget(selectionLabel);
    selectionToolBar->addWidget(selectionBox_);

    // ToolManager //
    //
    ODD::instance()->mainWindow()->addToolBar(selectionToolBar);
}

//################//
// SLOTS          //
//################//

/*! \brief.
*/
void
SelectionTool::activateProject(bool active)
{
    // Enable/Disable //
    //
    selectionBox_->setEnabled(active);
}

/*! \brief Gets called when a tool button has been selected.
*
*/
void
SelectionTool::handleToolClick(int id)
{
    selectionToolId_ = (SelectionTool::SelectionToolId)id;

    // Set a tool //
    //
    SelectionToolAction *action = new SelectionToolAction(selectionToolId_);
    emit toolAction(action);
    delete action;
}


void
SelectionTool::keyAction(KeyAction *keyAction)
{
    if (keyAction->getKeyActionType() == KeyAction::ATK_PRESS)
    {
        QKeyEvent *event = keyAction->getEvent();

        switch (event->key())
        {

        case Qt::Key_B:
            selectionBox_->click();
            break;

        default:
            break;
        }
    }
}

//################//
//                //
// SelectionToolAction //
//                //
//################//

// Note: This is not a typical Editor/Tool combination since this is not bound to
// a specify editor! So ENO_EDITOR and TNO_TOOL is set (Otherwise an editor would
// be loaded).

SelectionToolAction::SelectionToolAction(SelectionTool::SelectionToolId selectionToolId)
    : ToolAction(ODD::ENO_EDITOR, ODD::TNO_TOOL)
    , selectionToolId_(selectionToolId)
{
	if (selectionToolId == SelectionTool::TSL_BOUNDINGBOX)
	{
		boundingBoxActive_ = !boundingBoxActive_;
	}
}
