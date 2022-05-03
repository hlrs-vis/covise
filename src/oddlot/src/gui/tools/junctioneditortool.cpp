/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   06.04.2010
 **
 **************************************************************************/

#include "junctioneditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"


// Qt //
//
#include <QButtonGroup>


//################//
//                //
// JunctionEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
JunctionEditorTool::JunctionEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TJE_SELECT)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    // Tool Bar //
    //
    initToolBar();
    initToolWidget();
}

void
JunctionEditorTool::initToolWidget()
{

    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    //ribbonWidget->
    ui = new Ui::JunctionRibbon();
    ui->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    connect(ribbonToolGroup_, SIGNAL(idClicked(int)), this, SLOT(handleRibbonToolClick(int)));
#else
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));
#endif

    ribbonToolGroup_->addButton(ui->connectingLane, ODD::TJE_CREATE_LANE);
    ribbonToolGroup_->addButton(ui->connectingRoad, ODD::TJE_CREATE_ROAD);
    ribbonToolGroup_->addButton(ui->split, ODD::TJE_SPLIT);
    ribbonToolGroup_->addButton(ui->adjust, ODD::TJE_MOVE);

    ribbonToolGroup_->addButton(ui->junctionCreate, ODD::TJE_CREATE_JUNCTION);
    ribbonToolGroup_->addButton(ui->junctionAdd, ODD::TJE_ADD_TO_JUNCTION);
    ribbonToolGroup_->addButton(ui->junctionRemove, ODD::TJE_REMOVE_FROM_JUNCTION);

    ribbonToolGroup_->addButton(ui->linkSelected, ODD::TJE_LINK_ROADS);
    ribbonToolGroup_->addButton(ui->unlinkSelected, ODD::TJE_UNLINK_ROADS);
    ribbonToolGroup_->addButton(ui->cuttingCircle, ODD::TJE_CIRCLE);

    ribbonToolGroup_->addButton(ui->select, ODD::TJE_SELECT);

    /*   connect(ui->radiusEdit, SIGNAL(editingFinished()), this, SLOT(setRRadius()));
       connect(ui->radiusEdit, SIGNAL(valueChanged(double)), this, SLOT(setRRadius(double))); */

    toolManager_->addRibbonWidget(ribbonWidget, tr("Junction"), ODD::EJE);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
JunctionEditorTool::initToolBar()
{
    // no tool bar for me
}

//################//
// SLOTS          //
//################//


/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
JunctionEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::EJE);
    JunctionEditorToolAction *junctionEditorToolAction = dynamic_cast<JunctionEditorToolAction *>(action);

    ribbonToolGroup_->button(action->getToolId())->click();


}


/*! \brief Gets called when a tool button has been selected.
*
*/
void
JunctionEditorTool::handleRibbonToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    JunctionEditorToolAction *action = new JunctionEditorToolAction(toolId_);
    emit toolAction(action);
    //   delete action;
}



//################//
//                //
// JunctionEditorToolAction //
//                //
//################//

JunctionEditorToolAction::JunctionEditorToolAction(ODD::ToolId toolId, ODD::ToolId paramToolId)
    : ToolAction(ODD::EJE, toolId, paramToolId)
{
}
