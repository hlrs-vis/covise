/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/6/2010
**
**************************************************************************/

#include "projecttreeitem.hpp"

// Data //
//
#include "src/data/dataelement.hpp"
#include "src/data/projectdata.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// Tree //
//
#include "src/tree/projecttree.hpp"
#include "src/tree/items/roadtreeitem.hpp"

//################//
// CONSTRUCTOR    //
//################//

ProjectTreeItem::ProjectTreeItem(ProjectTreeItem *parent, DataElement *dataElement, QTreeWidgetItem *fosterParent)
    : QObject(parent)
    , QTreeWidgetItem()
    , Observer()
    , parentProjectTreeItem_(parent)
    , dataElement_(dataElement)
    , isInGarbage_(false)
{
    // Append to fosterParent if given, parent otherwise //
    //
    if (fosterParent)
    {
        fosterParent->addChild(this);
    }
    else
    {
        parent->addChild(this);
    }

    // Init //
    //
    init();
}

ProjectTreeItem::~ProjectTreeItem()
{
    // Observer //
    //
    dataElement_->detachObserver(this);
}

//################//
// FUNCTIONS      //
//################//

void
ProjectTreeItem::init()
{
    // Observer //
    //
    dataElement_->attachObserver(this);

    // Default settings //
    //
    setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    if (dataElement_->isElementSelected())
    {
        setSelected(true);
    }

    //	if(dataElement_->isElementHidden())
    //	{
    ////		setHidden(true);
    //		setDisabled(true);
    //	}
}

void
ProjectTreeItem::registerForDeletion()
{
    getProjectTree()->addToGarbage(this);

    notifyDeletion();
}

void
ProjectTreeItem::notifyDeletion()
{
    isInGarbage_ = true;
}

/*! \brief Returns the ProjectTree this item belongs to.
*
* Returns the ProjectTree of its parent item. Only root nodes
* like the RoadSystemTreeItem actually save and return it directly.
*/
ProjectTree *
ProjectTreeItem::getProjectTree() const
{
    if (parentProjectTreeItem_)
    {
        return parentProjectTreeItem_->getProjectTree();
    }
    else
    {
        return NULL;
    }
}

bool
ProjectTreeItem::isDescendantOf(ProjectTreeItem *projectTreeItem)
{
    if (parentProjectTreeItem_)
    {
        if (parentProjectTreeItem_ == projectTreeItem)
        {
            return true;
        }
        else
        {
            return parentProjectTreeItem_->isDescendantOf(projectTreeItem);
        }
    }
    else
    {
        return false;
    }
}

void
ProjectTreeItem::setData(int column, int role, const QVariant &value)
{
    //	qDebug() << "role: " << role;

    // View: selected ->  Data: select //
    //
    if (role == Qt::UserRole + ProjectTree::PTR_Selection)
    {
        if (isInGarbage() | (column != 0)) // only first column
        {
            return;
        }

        if (value.toBool())
        {
            if (!dataElement_->isElementSelected())
            {
                SelectDataElementCommand *command = new SelectDataElementCommand(dataElement_);
                if (command->isValid())
                {
                    dataElement_->getUndoStack()->push(command);
                }
            }
        }
        else
        {
            if (dataElement_->isElementSelected())
            {
                DeselectDataElementCommand *command = new DeselectDataElementCommand(dataElement_);
                if (command->isValid())
                {
                    dataElement_->getUndoStack()->push(command);
                }
            }
        }
    }

    QTreeWidgetItem::setData(column, role, value);
}

//################//
// EVENTS         //
//################//

//################//
// STATIC         //
//################//

//################//
// OBSERVER       //
//################//

void
ProjectTreeItem::updateObserver()
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
        // NOT CALLED?!? WHY?
        //		if(isHidden() != dataElement_->isElementHidden())
        //		{
        //			setHidden(dataElement_->isElementHidden());
        //			setDisabled(dataElement_->isElementHidden());
        //		}
    }

    // Selection //
    //
    if ((changes & DataElement::CDE_SelectionChange))
    {
        if (isSelected() != dataElement_->isElementSelected())
        {
            setSelected(dataElement_->isElementSelected());
        }

        if (isSelected())
        {
            if (dataElement_->getProjectData()->getSelectedElements().count() > 1)
            {
                treeWidget()->scrollToItem(this);
            }
            else
            {
                QTreeWidgetItem *parent = this;
                do
                {
                    parent = parent->parent();
                    if (parent)
                        parent->setExpanded(true);
                } while (parent);
                setExpanded(true);
                treeWidget()->scrollTo(this->treeWidget()->selectionModel()->selectedIndexes()[0]);
            }
        }
    }
}
