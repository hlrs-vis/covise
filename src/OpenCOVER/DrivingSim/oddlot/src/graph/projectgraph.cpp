/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#include "projectgraph.hpp"

#include "src/gui/projectwidget.hpp"

#include "src/data/projectdata.hpp"

#include "src/data/commands/datacommand.hpp"
#include "src/mainwindow.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"
#include "src/gui/tools/maptool.hpp"

// Graph //
//
#include "src/graph/items/graphelement.hpp"

// Qt //
//
#include <QtGui>
#include <QStatusBar>

//################//
// Constructors   //
//################//

ProjectGraph::ProjectGraph(ProjectWidget *projectWidget, ProjectData *projectData)
    : QWidget(projectWidget)
    , Observer()
    , projectWidget_(projectWidget)
    , projectData_(projectData)
{
    // Observer //
    //
    projectData_->attachObserver(this);
    numPostpones = 0;
}

ProjectGraph::~ProjectGraph()
{
    // Observer //
    //
    projectData_->detachObserver(this);
}

/*! \brief Register item to be removed later.
*/
void
ProjectGraph::addToGarbage(QGraphicsItem *item)
{
    if (!item)
    {
        return; // NULL pointer
    }

    if (!garbageList_.contains(item))
    {
        foreach (QGraphicsItem *garbageItem, garbageList_) // the list should not be too long...
        {
            if (garbageItem->isAncestorOf(item))
            {
                // Item will be destroyed by parent item //
                //
                return;
            }

            //if(item->isAncestorOf(garbageItem))
            //{
            //	// gargabeItem is a child of item but will be deleted first anyway
            //}
        }
        garbageList_.append(item);
    }
}

bool
ProjectGraph::executeCommand(DataCommand *command)
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

//################//
// SLOTS          //
//################//

/*! \brief .
*
*/
void
ProjectGraph::toolAction(ToolAction * /*toolAction*/)
{
}

void
ProjectGraph::mouseAction(MouseAction * /*mouseAction*/)
{
    //	graphScene_->mouseAction(mouseAction);
}

void
ProjectGraph::keyAction(KeyAction * /*keyAction*/)
{
    //	graphScene_->keyAction(keyAction);
}

/*! \brief Called right before the editor will be changed.
*
*/
void
ProjectGraph::preEditorChange()
{
    //	graphScene_->clearSelection();
}

/*! \brief Called right after the editor has been changed.
*
*/
void
ProjectGraph::postEditorChange()
{
}

/*! \brief Remove all registered items.
*/
void
ProjectGraph::garbageDisposal()
{
    if (numPostpones <= 0)
    {
        foreach (QGraphicsItem *item, garbageList_)
        {
            //graphScene_->removeItem(item);
            delete item;
        }
        garbageList_.clear();
    }
}

void ProjectGraph::postponeGarbageDisposal()
{
    numPostpones++;
}

void ProjectGraph::finishGarbageDisposal()
{
    numPostpones--;
    if (numPostpones <= 0)
    {
        foreach (QGraphicsItem *item, garbageList_)
        {
            //graphScene_->removeItem(item);
            delete item;
        }
        garbageList_.clear();
    }
}

//################//
// OBSERVER       //
//################//

void
ProjectGraph::updateObserver()
{

    // Get change flags //
    //
    //int changes = projectData_->getProjectDataChanges();

    // xyz //
    //
    //	if((changes & ProjectData::CPD_))
    //	{

    //	}
}
