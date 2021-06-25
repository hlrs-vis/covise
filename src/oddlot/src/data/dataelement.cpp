/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.03.2010
**
**************************************************************************/

#include "dataelement.hpp"

#include "src/data/changemanager.hpp"

// Data //
//
#include "src/data/projectdata.hpp"

//################//
// CONSTRUCTOR    //
//################//

/*! \brief .
*
* Sets the change flags to 0x1 (Element created).
*/
DataElement::DataElement()
    : Acceptor()
    , Subject()
    , dataElementChanges_(0x0)
    , projectData_(NULL)
    , parentElement_(NULL)
    , selected_(false)
    , childSelected_(false)
    , hidden_(false)
{
}

DataElement::~DataElement()
{
    // Deselect //
    //
    if (parentElement_)
    {
        parentElement_->delChild(this);
        parentElement_->delSelectedChild(this);
    }
}

//##################//
// ProjectData      //
//##################//

/*! \brief Returns the ProjectData this DataElement belongs to.
*
*/
ProjectData *
DataElement::getProjectData()
{
    return projectData_;
}

/*! \brief Convenience function to get the QUndoStack.
*
*/
QUndoStack *
DataElement::getUndoStack()
{
    if (projectData_)
    {
        return projectData_->getUndoStack();
    }
    else
    {
        return NULL;
    }
}

/*! \brief Convenience function to get the ChangeManager.
*
*/
ChangeManager *
DataElement::getChangeManager()
{
    if (projectData_)
    {
        return projectData_->getChangeManager();
    }
    else
    {
        return NULL;
    }
}

void
DataElement::linkToProject(ProjectData *projectData)
{
    // Check //
    //
    if (!projectData)
    {
        qDebug("ERROR 1010191404! Link to project: NULL...");
    }

    // Set //
    //
    projectData_ = projectData;
    addDataElementChanges(DataElement::CDE_LinkedToProject);

    // Children //
    //
    foreach (DataElement *element, childElements_)
    {
        element->linkToProject(projectData_);
    }

    // Selection/Hiding //
    //
    if (isElementSelected())
    {
        projectData_->addSelectedElement(this);
    }
    if (isElementHidden())
    {
        projectData_->addHiddenElement(this);
    }
}

void
DataElement::unlinkFromProject()
{
    // Selection/Hiding //
    //
    if (isElementSelected())
    {
        projectData_->removeSelectedElement(this);
    }
    if (isElementHidden())
    {
        projectData_->removeHiddenElement(this);
    }

    // Set //
    //
    addDataElementChanges(DataElement::CDE_UnlinkedFromProject);
    addDataElementChanges(DataElement::CDE_DataElementRemoved); // for legacy
    projectData_ = NULL;

    // Children //
    //
    foreach (DataElement *element, childElements_)
    {
        element->unlinkFromProject();
    }
}

bool
DataElement::isLinkedToProject()
{
    if (projectData_)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//##################//
// Parent Element   //
//##################//

void
DataElement::setParentElement(DataElement *newParentElement)
{
    if (!newParentElement)
    {
        // Remove element //
        //
        if (parentElement_)
        {
            parentElement_->delSelectedChild(this);
            parentElement_->delChild(this);
        }

        unlinkFromProject();

        parentElement_ = NULL;
    }
    else
    {
        // Remove element //
        //
        if (parentElement_)
        {
            parentElement_->delSelectedChild(this);
            parentElement_->delChild(this);
        }

        // Add element //
        //
        newParentElement->addChild(this);
        if (isElementSelected())
        {
            newParentElement->addSelectedChild(this);
        }

        if (newParentElement->getProjectData())
        {
            linkToProject(newParentElement->getProjectData());
        }

        parentElement_ = newParentElement; // save first
        addDataElementChanges(DataElement::CDE_DataElementAdded);
    }
}

//##################//
// Selection        //
//##################//

/*! \brief Set the selection state of this element.
*
* The selection state of the parent element will also be set.
*/
void
DataElement::setElementSelected(bool selected)
{
    if (selected != selected_)
    {
        // This //
        //
        selected_ = selected;
        addDataElementChanges(DataElement::CDE_SelectionChange);

        // Parent //
        //
        if (parentElement_)
        {
            if (selected)
            {
                parentElement_->addSelectedChild(this);
            }
            else
            {
                parentElement_->delSelectedChild(this);
            }
        }
    }

    // Active Element //
    //
    if (getProjectData())
    {
        if (selected_)
        {
            getProjectData()->addSelectedElement(this);
        }
        else
        {
            getProjectData()->removeSelectedElement(this);
        }
    }
}

/*! \brief Adds an element to the list of children.
*/
void
DataElement::addChild(DataElement *element)
{
    if (!childElements_.contains(element))
    {
        childElements_.append(element);
    }
    addDataElementChanges(DataElement::CDE_ChildChange);
}

/*! \brief Removes a child element from the list of selected elements.
*/
void
DataElement::delChild(DataElement *element)
{
    childElements_.removeOne(element); // this shouldn't be too slow since list is rather small at most times
    addDataElementChanges(DataElement::CDE_ChildChange);
}

/*! \brief Adds a child element to the list of selected elements.
*/
void
DataElement::addSelectedChild(DataElement *element)
{
    if (!selectedChildElements_.contains(element))
    {
        selectedChildElements_.append(element);
    }
    childSelected_ = true;
    addDataElementChanges(DataElement::CDE_ChildSelectionChange);
}

/*! \brief Removes a child element from the list of selected elements.
*/
void
DataElement::delSelectedChild(DataElement *element)
{
    selectedChildElements_.removeOne(element); // this shouldn't be too slow since list is rather small at most times
    if (selectedChildElements_.size() == 0)
    {
        childSelected_ = false;
        addDataElementChanges(DataElement::CDE_ChildSelectionChange);
    }
}

//##################//
// Hiding           //
//##################//

/*! \brief Set the hiding state of this element.
*
*/
void
DataElement::setElementHidden(bool hidden)
{
    if (hidden != hidden_)
    {
        hidden_ = hidden;
        addDataElementChanges(DataElement::CDE_HidingChange);
    }

    if (getProjectData())
    {
        if (hidden_)
        {
            getProjectData()->addHiddenElement(this);
        }
        else
        {
            getProjectData()->removeHiddenElement(this);
        }
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
DataElement::notificationDone()
{
    dataElementChanges_ = 0x0;
}

/*! \brief Add one or more change flags.
*
*/
void
DataElement::addDataElementChanges(int changes)
{
    if (changes)
    {
        dataElementChanges_ |= changes;
        notifyObservers();
    }
}
