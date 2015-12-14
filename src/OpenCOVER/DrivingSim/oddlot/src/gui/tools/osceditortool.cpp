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
	// Widget/Layout //
    //
	 QGridLayout *toolLayout = new QGridLayout;

    // ButtonGroup //
    //
    // A button group so only one button can be checked at a time
    QButtonGroup *toolGroup = new QButtonGroup;
    connect(toolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));

    // Tools //
    //
    QPushButton *toolButton;
    int row = -1; // button row

    // Link Roads by Handles//
    //
    catalogComboBox_ = new QComboBox(); 
	catalogComboBox_->addItem("Choose new catalog ...");
	catalogComboBox_->addItem("vehicleCatalog");
	catalogComboBox_->addItem("driverCatalog");
	catalogComboBox_->addItem("observerCatalog");
	catalogComboBox_->addItem("pedestrianCatalog");
	catalogComboBox_->addItem("miscObjectCatalog");
	catalogComboBox_->addItem("entityCatalog");
	catalogComboBox_->addItem("environmentCatalog");
	catalogComboBox_->addItem("maneuverCatalog");
	catalogComboBox_->addItem("routingCatalog");
   
    connect(catalogComboBox_, SIGNAL(currentIndexChanged(int)), this, SLOT(handleCatalogSelection(int)));
    catalogComboBox_->setCurrentIndex(0); // this doesn't trigger an event...
    handleCatalogSelection(0); // ... so do it yourself
    toolLayout->addWidget(catalogComboBox_, ++row, 0);


    // Link Roads //
    //
    toolButton = new QPushButton(tr("Save Catalog"));
    toolButton->setCheckable(false);
    toolLayout->addWidget(toolButton, ++row, 0);
    toolGroup->addButton(toolButton, ODD::TOS_SAVE_CATALOG); // button, id

    // Finish Layout //
    //
    toolLayout->setRowStretch(++row, 1); // last row fills the rest of the availlable space
    toolLayout->setColumnStretch(1, 1); // column 1 fills the rest of the availlable space

    // Widget/Layout //
    //
    ToolWidget *toolWidget = new ToolWidget();
    toolWidget->setLayout(toolLayout);
	toolManager_->addToolBoxWidget(toolWidget, tr("OpenScenario Editor"));
    connect(toolWidget, SIGNAL(activated()), this, SLOT(activateEditor()));
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
		const QString selectedText = catalogComboBox_->itemText(id);

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

