/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#include "osceditortool.hpp"

#include "toolmanager.hpp"
#include "toolwidget.hpp"

#include "src/mainwindow.hpp"

// Qt  //
//
#include <QLayout>
#include <QButtonGroup>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>

//################//
//                //
// OpenScenarioEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
OpenScenarioEditorTool::OpenScenarioEditorTool(ToolManager *toolManager)
    : Tool(toolManager)
    , toolId_(ODD::TOS_SELECT)
	, ui(new Ui::OSCRibbon)
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
OpenScenarioEditorTool::initToolWidget()
{
	 // Ribbon //
    //

    ToolWidget *ribbonWidget = new ToolWidget();
    ui->setupUi(ribbonWidget);

	ui->catalogComboBox->addItem("vehicleCatalog");
	ui->catalogComboBox->addItem("driverCatalog");
	ui->catalogComboBox->addItem("observerCatalog");
	ui->catalogComboBox->addItem("pedestrianCatalog");
	ui->catalogComboBox->addItem("miscObjectCatalog");
	ui->catalogComboBox->addItem("entityCatalog");
	ui->catalogComboBox->addItem("environmentCatalog");
	ui->catalogComboBox->addItem("maneuverCatalog");
	ui->catalogComboBox->addItem("routingCatalog");
   
    connect(ui->catalogComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleCatalogSelection(int)));
    ui->catalogComboBox->setCurrentIndex(0); // this doesn't trigger an event...
    handleCatalogSelection(0); // ... so do it yourself

    
    QButtonGroup *ribbonToolGroup = new QButtonGroup;
    connect(ribbonToolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));
    
    ribbonToolGroup->addButton(ui->oscSave, ODD::TOS_SAVE_CATALOG); 
    
    toolManager_->addRibbonWidget(ribbonWidget, tr("OpenScenario"));
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateEditor()));
}

void
	OpenScenarioEditorTool::initToolBar()
{
    // no toolbar for me //
}

//################//
// SLOTS          //
//################//

/*! \brief Gets called when this widget (tab) has been activated.
*/
void
OpenScenarioEditorTool::activateEditor()
{
    OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(toolId_, "");
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when a tool has been selected.
*/
void
OpenScenarioEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    // Set a tool //
    //
    OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(toolId_, "");
    emit toolAction(action);
    delete action;
}

/*! \brief Gets called when a tool has been selected.
*/
void
OpenScenarioEditorTool::handleCatalogSelection(int id)
{
	if (id > 0)
	{
		toolId_ = ODD::TOS_CREATE_CATALOG;
		const QString selectedText = ui->catalogComboBox->itemText(id);

		// Set a tool //
		//
		OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(toolId_, selectedText);
		emit toolAction(action);
		delete action;
	}
}

//################//
//                //
// OpenScenarioEditorToolAction //
//                //
//################//

OpenScenarioEditorToolAction::OpenScenarioEditorToolAction(ODD::ToolId toolId, const QString &text)
    : ToolAction(ODD::EOS, toolId)
	, text_(text)
{
}

