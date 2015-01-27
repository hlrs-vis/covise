/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/19/2010
**
**************************************************************************/

#include "settingselement.hpp"

// Data //
//
#include "src/data/dataelement.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// Qt //
//

//################//
// CONSTRUCTOR    //
//################//

SettingsElement::SettingsElement(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, DataElement *dataElement)
    : QWidget(parentSettingsElement)
    , Observer()
    , projectSettings_(projectSettings)
    , parentSettingsElement_(parentSettingsElement)
    , dataElement_(dataElement)
    , isInGarbage_(false)
{
    init();
}

SettingsElement::~SettingsElement()
{
    // Observer //
    //
    dataElement_->detachObserver(this);
}

void
SettingsElement::init()
{
    // Observer //
    //
    dataElement_->attachObserver(this);
}

void
SettingsElement::registerForDeletion()
{
    getProjectSettings()->addToGarbage(this);

    notifyDeletion();
}

void
SettingsElement::notifyDeletion()
{
    isInGarbage_ = true;
}

ProjectData *
SettingsElement::getProjectData() const
{
    return dataElement_->getProjectData();
}

//##################//
// ProjectSettings  //
//##################//

/*! \brief Returns the ProjectSettings this DataElement belongs to.
*
*/
ProjectSettings *
SettingsElement::getProjectSettings() const
{
    return projectSettings_;
}

//##################//
// Parent Element   //
//##################//

//void
//	SettingsElement
//	::setParentSettingsElement(SettingsElement * parentSettingsElement)
//{
//	parentSettingsElement_ = parentSettingsElement;
//}

//################//
// OBSERVER       //
//################//

void
SettingsElement::updateObserver()
{

    // Get change flags //
    //
    int changes = dataElement_->getDataElementChanges();

    // Deletion //
    //
    if ((changes & DataElement::CDE_DataElementDeleted)
        || (changes & DataElement::CDE_DataElementRemoved))
    {
        registerForDeletion();
        return;
    }

    // Hiding //
    //
    if ((changes & DataElement::CDE_HidingChange))
    {
        //		setVisible(!dataElement_->isElementHidden());
    }

    // Selection //
    //
    if (changes & DataElement::CDE_SelectionChange)
    {
        //		// Selection //
        //		//
        //		if(isSelected() != dataElement_->isElementSelected())
        //		{
        //			// DO NOT LET THE OBSERVER CALL itemChange() WHEN YOU ARE ALREADY IN IT!
        //			setSelected(dataElement_->isElementSelected());
        //		}

        //		// Highlighting //
        //		//
        //		updateHighlightingState();
    }
}

//################//
// SLOTS          //
//################//

//void
//	SettingsElement
//	::hideSettingsElement()
//{
//	QList<DataElement *> elements;
//	elements.append(dataElement_);

//	HideDataElementCommand * command = new HideDataElementCommand(elements, NULL);
//	if(command->isValid())
//	{
//		dataElement_->getUndoStack()->push(command);
//	}
//	else
//	{
//		delete command;
//	}

//}

//################//
// EVENTS         //
//################//
