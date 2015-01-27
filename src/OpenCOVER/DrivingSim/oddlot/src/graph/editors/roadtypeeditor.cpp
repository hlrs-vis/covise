/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.03.2010
**
**************************************************************************/

#include "roadtypeeditor.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/roadsystem.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/graphview.hpp"

#include "src/graph/items/roadsystem/roadtype/roadtyperoadsystemitem.hpp"
#include "src/graph/items/roadsystem/roadtype/typesectionitem.hpp"

#include "src/graph/items/roadsystem/sections/sectionhandle.hpp"

// Tools //
//
#include "src/gui/tools/typeeditortool.hpp"

// Visitor //
//
#include "src/graph/visitors/roadmarkvisitor.hpp"

// Qt //
//
#include <QGraphicsItem>
#include <QUndoStack>

//################//
// CONSTRUCTORS   //
//################//

RoadTypeEditor::RoadTypeEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : ProjectEditor(projectWidget, projectData, topviewGraph)
    , roadSystemItem_(NULL)
    , insertSectionHandle_(NULL)
    , currentRoadType_(TypeSection::RTP_UNKNOWN)
{
}

RoadTypeEditor::~RoadTypeEditor()
{
    kill();
}

//################//
// FUNCTIONS      //
//################//

/**
*
*/
void
RoadTypeEditor::init()
{
    if (!roadSystemItem_)
    {
        // Root item //
        //
        roadSystemItem_ = new RoadTypeRoadSystemItem(getTopviewGraph(), getProjectData()->getRoadSystem());
        getTopviewGraph()->getScene()->addItem(roadSystemItem_);

        // Section Handle //
        //
        insertSectionHandle_ = new SectionHandle(roadSystemItem_);
        insertSectionHandle_->hide();
    }
}

/*!
*/
void
RoadTypeEditor::kill()
{
    delete roadSystemItem_;
    roadSystemItem_ = NULL;
}

SectionHandle *
RoadTypeEditor::getInsertSectionHandle() const
{
    if (!insertSectionHandle_)
    {
        qDebug("ERROR 1003281422! RoadTypeEditor not yet initialized.");
    }
    return insertSectionHandle_;
}

void
RoadTypeEditor::setCurrentRoadType(TypeSection::RoadType roadType)
{
    currentRoadType_ = roadType;
}

//################//
// TOOL           //
//################//

/*! \brief .
*
*/
void
RoadTypeEditor::toolAction(ToolAction *toolAction)
{
    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    if (getCurrentTool() == ODD::TRT_SELECT)
    {
        // does nothing //
    }
    else if (getCurrentTool() == ODD::TRT_ADD)
    {
        // does nothing //
    }
    else if (getCurrentTool() == ODD::TRT_DEL)
    {
        // Problem: The ToolAction is resent, after a warning message has been clicked away. (Due to resend on getting the focus back?)

        //		QList<QGraphicsItem *> selectedItems = getTopviewGraph()->graphScene()->selectedItems();
        //
        //		// Macro Command //
        //		//
        //		int numberOfSelectedItems = selectedItems.size();
        //		if(numberOfSelectedItems > 1)
        //		{
        //			getProjectData()->getUndoStack()->beginMacro(QObject::tr("Delete Road Type Sections"));
        //		}
        //
        //		// Delete selected items //
        //		//
        //		foreach(QGraphicsItem * item, getTopviewGraph()->graphScene()->selectedItems())
        //		{
        //			TypeSectionItem * typeSectionItem = dynamic_cast<TypeSectionItem *>(item);
        //			if(typeSectionItem)
        //			{
        //				typeSectionItem->setSelected(false);
        //				typeSectionItem->deleteTypeSection();
        //			}
        //		}
        //
        //		// Macro Command //
        //		//
        //		if(numberOfSelectedItems > 1)
        //		{
        //			getProjectData()->getUndoStack()->endMacro();
        //		}
    }

    // RoadType //
    //
    TypeEditorToolAction *typeEditorToolAction = dynamic_cast<TypeEditorToolAction *>(toolAction);
    if (typeEditorToolAction)
    {
        // Set RoadType //
        //
        TypeSection::RoadType roadType = typeEditorToolAction->getRoadType();
        if (roadType != TypeSection::RTP_NONE)
        {
            if (typeEditorToolAction->isApplyingRoadType())
            {
                QList<QGraphicsItem *> selectedItems = getTopviewGraph()->getScene()->selectedItems();

                // Macro Command //
                //
                int numberOfSelectedItems = selectedItems.size();
                if (numberOfSelectedItems > 1)
                {
                    getProjectData()->getUndoStack()->beginMacro(QObject::tr("Set Road Type"));
                }

                // Change types of selected items //
                //
                foreach (QGraphicsItem *item, selectedItems)
                {
                    TypeSectionItem *typeSectionItem = dynamic_cast<TypeSectionItem *>(item);
                    if (typeSectionItem)
                    {
                        typeSectionItem->changeRoadType(roadType);
                    }
                }

                // Macro Command //
                //
                if (numberOfSelectedItems > 1)
                {
                    getProjectData()->getUndoStack()->endMacro();
                }
            }
            else
            {
                setCurrentRoadType(roadType);
            }
        }
    }
}
