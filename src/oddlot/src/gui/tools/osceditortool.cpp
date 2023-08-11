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


 // Qt  //
 //
#include <QButtonGroup>
#include <QtGlobal>

//################//
//                //
// OpenScenarioEditorTool //
//                //
//################//

/*! \todo Ownership/destructor
*/
OpenScenarioEditorTool::OpenScenarioEditorTool(ToolManager *toolManager)
    : EditorTool(toolManager)
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


    ribbonToolGroup_ = new QButtonGroup(toolManager_);
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    connect(ribbonToolGroup_, SIGNAL(idClicked(int)), this, SLOT(handleToolClick(int)));
#else
    connect(ribbonToolGroup_, SIGNAL(buttonClicked(int)), this, SLOT(handleToolClick(int)));
#endif

    ribbonToolGroup_->addButton(ui->oscSave, ODD::TOS_SAVE_CATALOG);
    ribbonToolGroup_->addButton(ui->select, ODD::TOS_SELECT);
    ribbonToolGroup_->addButton(ui->invisibleButton, ODD::TOS_NONE);
    ribbonToolGroup_->addButton(ui->fileHeaderButton, ODD::TOS_FILEHEADER);
    ribbonToolGroup_->addButton(ui->roadNetworkButton, ODD::TOS_ROADNETWORK);
    ribbonToolGroup_->addButton(ui->entitiesButton, ODD::TOS_ENTITIES);
    ribbonToolGroup_->addButton(ui->storyboardButton, ODD::TOS_STORYBOARD);
    ui->invisibleButton->hide();

    // ribbonToolGroup_->addButton(ui->graphEditButton, ODD::TOS_GRAPHELEMENT);
    connect(ui->graphEditButton, SIGNAL(clicked(bool)), this, SLOT(handleGraphState(bool)));


    toolManager_->addRibbonWidget(ribbonWidget, tr("OpenScenario"), ODD::EOS);
    connect(ribbonWidget, SIGNAL(activated()), this, SLOT(activateRibbonEditor()));
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

/*! \brief Is called by the toolmanager to initialize the UI */
/* UI sets the values of the current project */
void
OpenScenarioEditorTool::activateRibbonEditor()
{
    ui->graphEditButton->setEnabled(false);
    ui->graphEditButton->setVisible(false);
    ui->graphEditButton->setChecked(false);
    graphEdit_ = false;

    ToolAction *action = toolManager_->getLastToolAction(ODD::EOS);

    ribbonToolGroup_->button(action->getToolId())->click();

}

/*! \brief Gets called when a tool has been selected.
*/
void
OpenScenarioEditorTool::handleToolClick(int id)
{
    toolId_ = (ODD::ToolId)id;

    if (toolId_ == ODD::TOS_NONE)
    {
        return;
    }

    if (graphEdit_)
    {
        handleGraphState(false);
        ui->graphEditButton->setChecked(false);
    }

    OpenScenarioEditorToolAction *action;

    switch (toolId_)
    {
    case ODD::TOS_ENTITIES:
    case ODD::TOS_ROADNETWORK:
    case ODD::TOS_STORYBOARD:
    case ODD::TOS_FILEHEADER:
        action = new OpenScenarioEditorToolAction(toolId_, ribbonToolGroup_->checkedButton()->text());
        break;
    default:
        action = new OpenScenarioEditorToolAction(toolId_, "");
        if (ui->graphEditButton->isVisible())
        {
            enableGraphEdit(false);
        }
        break;
    }
    emit toolAction(action);
    // delete action;
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
    : ToolAction(ODD::EOS, toolId, ODD::TNO_TOOL)
    , text_(text)
{
}

OpenScenarioEditorToolAction::OpenScenarioEditorToolAction(ODD::ToolId toolId, bool state)
    : ToolAction(ODD::EOS, toolId, ODD::TNO_TOOL)
    , state_(state)
{
}

