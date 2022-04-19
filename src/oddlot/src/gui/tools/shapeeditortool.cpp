/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 /**************************************************************************
 ** ODD: OpenDRIVE Designer
 **   Frank Naegele (c) 2010
 **   <mail@f-naegele.de>
 **   14.07.2010
 **
 **************************************************************************/

#include "shapeeditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"


 // Qt //
 //
#include <QButtonGroup>


//################//
//                //
// ShapeEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
ShapeEditorTool::ShapeEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
    , toolId_(ODD::TRS_SELECT)
{
    // Connect emitted ToolActions to ToolManager //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    // Tool Bar //
    //
    initToolBar();
    initToolWidget();
}

void
ShapeEditorTool::initToolWidget()
{

    // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    ui_ = new Ui::ShapeRibbon();
    ui_->setupUi(ribbonWidget);

    ribbonToolGroup_ = new QButtonGroup(toolManager_);
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleRibbonToolClick(int)));


    ribbonToolGroup_->addButton(ui_->select, ODD::TRS_SELECT);
    ribbonToolGroup_->addButton(ui_->shapeAdd, ODD::TRS_ADD);
    ribbonToolGroup_->addButton(ui_->shapeDelete, ODD::TRS_DEL);

    toolManager_->addRibbonWidget(ribbonWidget, tr("RoadShape"), ODD::ERS);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
}

void
ShapeEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//


/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
ShapeEditorTool::activateRibbonEditor()
{
    ToolAction *action = toolManager_->getLastToolAction(ODD::ERS);

    ribbonToolGroup_->button(action->getToolId())->click();

}


void
ShapeEditorTool::handleRibbonToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    ShapeEditorToolAction *action = new ShapeEditorToolAction(toolId_);
    emit toolAction(action);
    // delete action;
}


//################//
//                //
// ShapeEditorToolAction //
//                //
//################//

ShapeEditorToolAction::ShapeEditorToolAction(ODD::ToolId toolId)
    : ToolAction(ODD::ERS, toolId)
{
}

