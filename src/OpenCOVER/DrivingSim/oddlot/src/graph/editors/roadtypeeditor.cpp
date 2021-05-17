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

#include "src/mainwindow.hpp"

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

// GUI //
//
#include "src/gui/parameters/toolvalue.hpp"
#include "src/gui/parameters/toolparametersettings.hpp"

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
    if (tool_)
    {
        delToolParameters();
        ODD::mainWindow()->showParameterDialog(false);
    }

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
    if (tool_ && !tool_->containsToolId(toolAction->getToolId()))
    {
        delToolParameters();
        ODD::mainWindow()->showParameterDialog(false);
    }

    // Parent //
    //
    ProjectEditor::toolAction(toolAction);

    // RoadType //
//
    TypeEditorToolAction *typeEditorToolAction = dynamic_cast<TypeEditorToolAction *>(toolAction);
    if (typeEditorToolAction)
    {
        if (getCurrentTool() == ODD::TRT_SELECT)
        {
            // does nothing //
        }
        else if (getCurrentTool() == ODD::TRT_ADD)
        {
            ODD::ToolId paramTool = getCurrentParameterTool();

            if ((paramTool == ODD::TNO_TOOL) && !tool_)
            {
                ToolValue<int> *param = new ToolValue<int>(ODD::TRT_ADD, ODD::TPARAM_VALUE, 0, ToolParameter::ParameterTypes::ENUM, "color, unknown, rural, motorway, town, lowspeed, pedestrian");
                param->setValue(setRoadTypeSelection(currentRoadType_));
                tool_ = new Tool(ODD::TRT_ADD, 1);
                tool_->readParams(param);

                createToolParameterSettings(tool_, ODD::ERT);
                ODD::mainWindow()->showParameterDialog(true, "Road Type", "Choose the type of the roadsection to be created.");
            }
        }
        else if (getCurrentTool() == ODD::TRT_DEL)
        {
            // Problem: The ToolAction is resent, after a warning message has been clicked away. (Due to resend on getting the focus back?)

            //  QList<QGraphicsItem *> selectedItems = getTopviewGraph()->graphScene()->selectedItems();
            //
            //  // Macro Command //
            //  //
            //  int numberOfSelectedItems = selectedItems.size();
            //  if(numberOfSelectedItems > 1)
            //  {
            //   getProjectData()->getUndoStack()->beginMacro(QObject::tr("Delete Road Type Sections"));
            //  }
            //
            //  // Delete selected items //
            //  //
            //  foreach(QGraphicsItem * item, getTopviewGraph()->graphScene()->selectedItems())
            //  {
            //   TypeSectionItem * typeSectionItem = dynamic_cast<TypeSectionItem *>(item);
            //   if(typeSectionItem)
            //   {
            //    typeSectionItem->setSelected(false);
            //    typeSectionItem->deleteTypeSection();
            //   }
            //  }
            //
            //  // Macro Command //
            //  //
            //  if(numberOfSelectedItems > 1)
            //  {
            //   getProjectData()->getUndoStack()->endMacro();
            //  }
        }
    }
    else
    {
        if (toolAction->getToolId() == ODD::TRT_ADD)
        {
            if (toolAction->getParamToolId() == ODD::TPARAM_VALUE)
            {
                ParameterToolAction *action = dynamic_cast<ParameterToolAction *>(toolAction);
                if (action)
                {
                    int index = action->getParamId();
                    setRoadType(index);
                    ToolParameter *p = tool_->getLastParam(settings_->getCurrentParameterID());
                    ToolValue<int> *v = dynamic_cast<ToolValue<int> *>(p);
                    v->setValue(index);
                }
            }
        }
    }
}

/*! \brief
*
*/
void
RoadTypeEditor::setRoadType(int id)
{
    if (id == 0)
    {
        currentRoadType_ = TypeSection::RTP_UNKNOWN;
    }
    else if (id == 1)
    {
        currentRoadType_ = TypeSection::RTP_RURAL;
    }
    else if (id == 2)
    {
        currentRoadType_ = TypeSection::RTP_MOTORWAY;
    }
    else if (id == 3)
    {
        currentRoadType_ = TypeSection::RTP_TOWN;
    }
    else if (id == 4)
    {
        currentRoadType_ = TypeSection::RTP_LOWSPEED;
    }
    else
    {
        currentRoadType_ = TypeSection::RTP_PEDESTRIAN;
    }
}

int
RoadTypeEditor::setRoadTypeSelection(TypeSection::RoadType type)
{
    if (type == TypeSection::RTP_UNKNOWN)
    {
        return 0;
    }
    else if (type == TypeSection::RTP_RURAL)
    {
        return 1;
    }
    else if (type == TypeSection::RTP_MOTORWAY)
    {
        return 2;
    }
    else if (type == TypeSection::RTP_TOWN)
    {
        return 3;
    }
    else if (type == TypeSection::RTP_LOWSPEED)
    {
        return 4;
    }
    else
    {
        return 5;
    }

}

void
RoadTypeEditor::reject()
{
    ProjectEditor::reject();

    delToolParameters();
    ODD::mainWindow()->showParameterDialog(false);
}

