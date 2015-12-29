/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/5/2010
**
**************************************************************************/

#include "projecttree.hpp"

#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
//#include "src/data/commands/projectdatacommands.hpp"

// Tree //
//
#include "projecttreewidget.hpp"
#include "items/projecttreeitem.hpp"

// Qt //
//
#include <QGridLayout>
#include <QVBoxLayout>

//################//
// CONSTRUCTOR    //
//################//

ProjectTree::ProjectTree(ProjectWidget *projectWidget, ProjectData *projectData)
    : QWidget(projectWidget)
    , projectWidget_(projectWidget)
    , projectData_(projectData)
{

    // Model //
    //
    projectTreeWidget_ = new ProjectTreeWidget(this, projectData_);
	projectTreeWidget_->setIndentation(6);

    // Buttons //
    //
    //	QGridLayout * buttonsLayout = new QGridLayout;

    // Layout //
    //
    QVBoxLayout *treeLayout = new QVBoxLayout;
    treeLayout->addWidget(projectTreeWidget_);
    //	treeLayout->addChildLayout(buttonsLayout);
    
    treeLayout->setContentsMargins(0,0,0,0);
    setLayout(treeLayout);
}

//################//
// FUNCTIONS      //
//################//

/*! \brief Register item to be removed later.
*/
void
ProjectTree::addToGarbage(ProjectTreeItem *item)
{
    if (!item)
    {
        return; // NULL pointer
    }

    if (!garbageList_.contains(item))
    {
        foreach (ProjectTreeItem *garbageItem, garbageList_) // the list should not be too long...
        {
            if (item->isDescendantOf(garbageItem))
            {
                // Item will be destroyed by parent item that is already in garbage //
                //
                return;
            }
        }
        garbageList_.append(item);
    }
}

//################//
// EVENTS         //
//################//

//##################//
// SLOTS            //
//##################//

/*! \brief This slot is called when the ProjectTree's project has been activated or deactivated.
*
*/
void
ProjectTree::projectActivated(bool active)
{
    if (active)
    {
    }
}

/*! \brief Remove all registered items.
*/
void
ProjectTree::garbageDisposal()
{
    foreach (ProjectTreeItem *item, garbageList_)
    {
        delete item;
    }
    garbageList_.clear();
}
