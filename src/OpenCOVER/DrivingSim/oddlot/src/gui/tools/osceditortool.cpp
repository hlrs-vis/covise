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

    
    QButtonGroup *ribbonToolGroup = new QButtonGroup;
    connect(ribbonToolGroup, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));
    
    ribbonToolGroup->addButton(ui->oscSave, ODD::TOS_SAVE_CATALOG); 

    connect(ui->graphEditButton, SIGNAL(clicked(bool)), this, SLOT(handleGraphState(bool)));
 //   ribbonToolGroup->addButton(ui->graphEditButton, ODD::TOS_GRAPHELEMENT);

	// add all members of OpenScenarioBase as buttons
	//
	//QButtonGroup *memberToolGroup = new QButtonGroup;

	// Signal Mapper for the objects //
	//
	QSignalMapper *signalPushMapper = new QSignalMapper(this);
	connect(signalPushMapper, SIGNAL(mapped(QString)), this, SLOT(onPushButtonPressed(QString)));

	QList<QString> openScenarioBaseObjects;
	openScenarioBaseObjects.append("FileHeader");
	openScenarioBaseObjects.append("RoadNetwork");
	openScenarioBaseObjects.append("Entities");
	openScenarioBaseObjects.append("Storyboard");
	
	int column = 0;
	for (int i = 0; i < openScenarioBaseObjects.size(); i++)
	{
		QPushButton *oscPushButton = new QPushButton(ui->baseTools);
		oscPushButton->setText(openScenarioBaseObjects.at(i));
		oscPushButton->setObjectName(openScenarioBaseObjects.at(i));
		//			memberWidgets_.insert(memberName, oscPushButton);
		ui->baseToolsLayout->addWidget(oscPushButton, i%2, column, 1, 1);
		connect(oscPushButton, SIGNAL(pressed()), signalPushMapper, SLOT(map()));
		signalPushMapper->setMapping(oscPushButton, openScenarioBaseObjects.at(i));
		if (i%2 == 1)
		{
			column++;
		}
	}  

   
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
    ui->graphEditButton->setEnabled(false);
    ui->graphEditButton->setVisible(false);
	ui->graphEditButton->setChecked(false);

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

void 
OpenScenarioEditorTool::onPushButtonPressed(QString name)
{
	if (name != "")
	{
		toolId_ = ODD::TOS_BASE;

		// Set a tool //
		//
		OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(toolId_, name);
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

