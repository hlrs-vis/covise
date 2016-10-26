/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/12/2010
**
**************************************************************************/

#include "projectsettings.hpp"

#include "src/gui/projectwidget.hpp"

#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
//#include "src/data/commands/projectdatacommands.hpp"

// Settings //
//
#include "projectsettingsvisitor.hpp"
#include "settingselement.hpp"
#include "ui_errorMessageTree.h"

// Qt //
//
#include <QGridLayout>
#include <QVBoxLayout>
#include <QStatusBar>
#include <QLabel>

//################//
// CONSTRUCTOR    //
//################//

ProjectSettings::ProjectSettings(ProjectWidget *projectWidget, ProjectData *projectData)
    : QWidget(projectWidget)
    , projectWidget_(projectWidget)
    , projectData_(projectData)
    , settingsElement_(NULL)
	, ui(new Ui::ErrorMessageTree)
{
    // Observer //
    //
    projectData_->attachObserver(this);

    // Buttons //
    //
    //	QGridLayout * buttonsLayout = new QGridLayout;

    // Layout //
    //
    settingsLayout_ = new QVBoxLayout;
    settingsLayout_->setContentsMargins(0,0,0,0);

    ////	settingsLayout->addChildLayout(buttonsLayout);

    setLayout(settingsLayout_);

	// Error Message Widget for errorDock_ //
	//
	QWidget *errorMessageWidget = new QWidget(this);
	ui->setupUi(errorMessageWidget);

	ODD::mainWindow()->setErrorMessageTree(errorMessageWidget);
}

ProjectSettings::~ProjectSettings()
{
    // Observer //
    //
    projectData_->detachObserver(this);
}

//################//
// FUNCTIONS      //
//################//

/*! \brief Register item to be removed later.
*/
void
ProjectSettings::addToGarbage(SettingsElement *item)
{
    if (!item)
    {
        return; // NULL pointer
    }

    if (!garbageList_.contains(item))
    {
        foreach (SettingsElement *garbageItem, garbageList_) // the list should not be too long...
        {
            if (garbageItem->isAncestorOf(item))
            {
                // Item will be destroyed by parent item that is already in garbage //
                //
                return;
            }
        }
        garbageList_.append(item);
        if (settingsElement_ == item)
        {
            settingsElement_ = NULL;
        }
    }
}

bool
ProjectSettings::executeCommand(DataCommand *command)
{
    ODD::mainWindow()->statusBar()->showMessage(command->text(), 4000);
    if (command->isValid())
    {
        getProjectData()->getUndoStack()->push(command);
        return true;
    }
    else
    {
        delete command;
        return false;
    }
}

void
ProjectSettings::printErrorMessage(const QString &text)
{
	if (ui->errorVerticalLayout->count() > 3)
	{
		QLayoutItem *item = ui->errorVerticalLayout->itemAt(1);
		ui->errorVerticalLayout->removeItem(item);
		delete item;
	}

	QLabel *label = new QLabel(text);
	label->setWordWrap(true);
	ui->errorVerticalLayout->insertWidget(ui->errorVerticalLayout->count()-1, label); 
}

void
ProjectSettings::updateWidget()
{
    // Create new one //
    //
    DataElement *element = projectData_->getActiveElement();
    if (!element)
    {
        element = projectData_;
    }

    if ((settingsElement_)
        && (element == settingsElement_->getDataElement()))
    {
        return; // do nothing, element already displayed
    }

    // Delete old one //
    //
    if (settingsElement_)
    {
        settingsElement_->registerForDeletion();
        //		addToGarbage(settingsElement_);
        settingsElement_ = NULL;
    }

    // TODO: find out, if the current element has been deselected, than don't create a Project Setting

    ProjectSettingsVisitor *projectSettingsVisitor = new ProjectSettingsVisitor(this);
    element->accept(projectSettingsVisitor);
    settingsElement_ = projectSettingsVisitor->getSettingsElement();

    // Add to layout //
    //
    if (!settingsElement_)
    {
        //		settingsLayout_->addWidget(new QWidget());
    }
    else
    {
        settingsLayout_->addWidget(settingsElement_);
    }
}

//################//
// EVENTS         //
//################//

//##################//
// SLOTS            //
//##################//

/*! \brief This slot is called when the ProjectSettings's project has been activated or deactivated.
*
*/
void
ProjectSettings::projectActivated(bool active)
{
    if (active)
    {
        updateWidget();
    }
}

/*! \brief Remove all registered items.
*/
void
ProjectSettings::garbageDisposal()
{
    foreach (SettingsElement *item, garbageList_)
    {
        delete item;
    }
    garbageList_.clear();
}

//##################//
// Observer Pattern //
//##################//

void
ProjectSettings::updateObserver()
{
    // CrossfallSection //
    //
    int changes = projectData_->getProjectDataChanges();

    if (changes & ProjectData::CPD_SelectedElementsChanged)
    {
        updateWidget();
    }
}
