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
#include <QSignalMapper>

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

	for (int i = 0; i < ODD::CATALOGLIST.size(); i++)
	{
		ui->catalogComboBox->addItem(QString::fromStdString(ODD::CATALOGLIST.at(i)));
	}
  
    connect(ui->catalogComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleCatalogSelection(int)));
    ui->catalogComboBox->setCurrentIndex(0); // this doesn't trigger an event...
    handleCatalogSelection(0); // ... so do it yourself

    
    ribbonToolGroup_ = new QButtonGroup;
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));
    
    ribbonToolGroup_->addButton(ui->oscSave, ODD::TOS_SAVE_CATALOG); 
	ribbonToolGroup_->addButton(ui->select, ODD::TOS_SELECT);
	ribbonToolGroup_->addButton(ui->invisibleButton, ODD::TOS_NONE);
	ribbonToolGroup_->addButton(ui->fileHeaderButton, ODD::TOS_BASE);
	ribbonToolGroup_->addButton(ui->roadNetworkButton, ODD::TOS_BASE);
	ribbonToolGroup_->addButton(ui->entitiesButton, ODD::TOS_BASE);
	ribbonToolGroup_->addButton(ui->storyboardButton, ODD::TOS_BASE);
	ui->invisibleButton->hide();

//	ribbonToolGroup_->addButton(ui->graphEditButton, ODD::TOS_GRAPHELEMENT);
	connect(ui->graphEditButton, SIGNAL(clicked(bool)), this, SLOT(handleGraphState(bool)));

   
    toolManager_->addRibbonWidget(ribbonWidget, tr("OpenScenario"));
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateEditor()));
}

void
	OpenScenarioEditorTool::initToolBar()
{
    // no toolbar for me //
}

void OpenScenarioEditorTool::objectSelection(bool state)
{
	ui->invisibleButton->setChecked(!state);
}

//################//
// SLOTS          //
//################//

/*! \brief Gets called when this widget (tab) has been activated.
*/
void
OpenScenarioEditorTool::activateEditor()
{
    ui->graphEditButton->setEnabled(false);
    ui->graphEditButton->setVisible(false);
	ui->graphEditButton->setChecked(false); 
	graphEdit_ = false;

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

	if (graphEdit_)
	{
		handleGraphState(false);
		ui->graphEditButton->setChecked(false);
	}

	OpenScenarioEditorToolAction *action;


	if (toolId_ == ODD::TOS_BASE)
	{
		action = new OpenScenarioEditorToolAction(toolId_, ribbonToolGroup_->checkedButton()->text());
	}
	else if (toolId_ != ODD::TOS_NONE)
	{
		action = new OpenScenarioEditorToolAction(toolId_, "");
		if (ui->graphEditButton->isVisible())
		{
			enableGraphEdit(false);
		}
	}
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


void
OpenScenarioEditorTool::enableGraphEdit(bool state)
{
    if (state || !ui->graphEditButton->isChecked()) // if hidden and deselected still visible
    {
        ui->graphEditButton->setEnabled(state);
        ui->graphEditButton->setVisible(state);
    } 
}

void
OpenScenarioEditorTool::handleGraphState(bool state)
{
	if (state)
	{
		ui->graphEditButton->setText("Editing Finished");
		graphEdit_ = true;
	}
	else
	{
		ui->graphEditButton->setText("Edit Graph");
		graphEdit_ = false;
	}

     // Set a tool //
    //
    OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(ODD::TOS_GRAPHELEMENT, state);
    emit toolAction(action);
    delete action;
}

void
OpenScenarioEditorTool::setButtonColor(const QString &name, QColor color)
{
	QPushButton *button = ui->baseTools->findChild<QPushButton *>(name);
	if (button)
	{
		button->setStyleSheet("color: rgb(" + QString::number(color.red()) + "," + QString::number(color.green()) + "," + QString::number(color.blue()) + ")");
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

OpenScenarioEditorToolAction::OpenScenarioEditorToolAction(ODD::ToolId toolId, bool state)
    : ToolAction(ODD::EOS, toolId)
	, state_(state)
{
}

