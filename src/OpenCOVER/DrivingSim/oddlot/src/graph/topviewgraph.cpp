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

#include "topviewgraph.hpp"

#include "src/gui/projectwidget.hpp"

#include "src/data/projectdata.hpp"
//#include "src/data/commands/projectdatacommands.hpp"

#include "src/data/commands/dataelementcommands.hpp"

#include "graphscene.hpp"
#include "graphview.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"
#include "src/gui/tools/maptool.hpp"

// Graph //
//
#include "src/graph/items/graphelement.hpp"
#include "src/graph/items/roadsystem/signal/signalitem.hpp"
#include "src/graph/editors/signaleditor.hpp"

// Qt //
//
#include <QtGui>
#include <QGridLayout>

//################//
// Constructors   //
//################//

TopviewGraph::TopviewGraph(ProjectWidget *projectWidget, ProjectData *projectData)
    : ProjectGraph(projectWidget, projectData)
{
    // Qt Scene //
    //
    //	graphScene_ = new GraphScene(QRectF(-1150.0, -480.0, 3850.0, 3780.0), this);
    //	graphScene_ = new GraphScene(QRectF(-10000.0, -10000.0, 20000.0, 20000.0), this); // x, y, sizeX, sizeY
    graphScene_ = new GraphScene(QRectF(-100.0, -1000.0, 20000.0, 20000.0), this); // x, y, sizeX, sizeY, south, west, north-south, east-west

    updateSceneSize();

    connect(graphScene_, SIGNAL(mouseActionSignal(MouseAction *)), this, SIGNAL(mouseActionSignal(MouseAction *)));
    connect(graphScene_, SIGNAL(keyActionSignal(KeyAction *)), this, SIGNAL(keyActionSignal(KeyAction *)));
    connect(graphScene_, SIGNAL(wheelActionSignal(WheelAction *)), this, SLOT(wheelAction(WheelAction *)));

    // Qt View //
    //
    graphView_ = new GraphView(graphScene_, this);
    graphView_->setObjectName("TopviewGraphView");
    graphView_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    graphView_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    //graphView_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    //graphView_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    graphView_->setRenderHints(QPainter::Antialiasing);

    graphView_->resetViewTransformation();

    //graphView_->setAlignment(Qt::AlignLeft | Qt::AlignTop); // place image at top left, center otherwise
    //graphView_->setCacheMode(QGraphicsView::CacheBackground); // speedup for tiles / many small pixmaps in viewport

    // Layout //
    //
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(graphView_);
    setLayout(layout);

    updateSceneSize();
}

TopviewGraph::~TopviewGraph()
{
    delete graphView_;
    delete graphScene_;
}

/*! \brief Updates the scene size
*/
void
TopviewGraph::updateSceneSize()
{
    graphScene_->setSceneRect(getProjectData()->getWest(), getProjectData()->getSouth(), getProjectData()->getEast() - getProjectData()->getWest(), getProjectData()->getNorth() - getProjectData()->getSouth());
}

void
    TopviewGraph::notifySignals()
{
    SignalEditor * signalEditor = dynamic_cast<SignalEditor *>(getProjectWidget()->getProjectEditor());
    if (signalEditor)
    {
        QList<QGraphicsItem *> items = graphView_->items();
        foreach (QGraphicsItem * item, items)
        {
            SignalItem * signalItem = dynamic_cast<SignalItem *>(item);
            if (signalItem)
            {
                signalItem->zoomAction();
            }
        }
    }
}

//################//
// SLOTS          //
//################//

/*! \brief .
*
*/
void
TopviewGraph::toolAction(ToolAction *toolAction)
{

    graphView_->toolAction(toolAction);

    // ViewTools //
    //
    ZoomToolAction *zoomToolAction = dynamic_cast<ZoomToolAction *>(toolAction);
    if (zoomToolAction)
    {
        // Canvas Size //
        //
        ZoomTool::ZoomToolId id = zoomToolAction->getZoomToolId();

        // Hiding //
        //
        if (id == ZoomTool::TZM_HIDE_SELECTED)
        {

            QList<QGraphicsItem *> selectedItems = getScene()->selectedItems();

            // Macro Command //
            //
            if (selectedItems.size() > 1)
            {
                getProjectData()->getUndoStack()->beginMacro(QObject::tr("Hide Elements"));
            }

            // Hide selected items //
            //
            foreach (QGraphicsItem *item, selectedItems)
            {
                GraphElement *graphElement = dynamic_cast<GraphElement *>(item);
                if (graphElement)
                {
                    graphElement->hideGraphElement();
                }
            }

            // Macro Command //
            //
            if (selectedItems.size() > 1)
            {
                getProjectData()->getUndoStack()->endMacro();
            }
        }
        if (id == ZoomTool::TZM_HIDE_SELECTED_ROADS)
        {
            QList<QGraphicsItem *> selectedItems = getScene()->selectedItems();

            // Macro Command //
            //
            if (selectedItems.size() > 1)
            {
                getProjectData()->getUndoStack()->beginMacro(QObject::tr("Hide Roads"));
            }

            // Hide selected items //
            //
            foreach (QGraphicsItem *item, selectedItems)
            {
                GraphElement *graphElement = dynamic_cast<GraphElement *>(item);
                if (graphElement)
                {
                    graphElement->hideRoads();
                }
            }

            // Macro Command //
            //
            if (selectedItems.size() > 1)
            {
                getProjectData()->getUndoStack()->endMacro();
            }
        }
        else if (id == ZoomTool::TZM_SELECT_INVERSE)
        {
            QList<QGraphicsItem *> selectedItems = getScene()->selectedItems();
            QList<QGraphicsItem *> items = getScene()->items();

            // Select inverse //
            //
            foreach (QGraphicsItem *item, items)
            {
                if (!selectedItems.contains(item))
                {
                    item->setSelected(true);
                }
                else
                {
                    item->setSelected(false);
                }
            }
        }
        //		else if(id == ZoomTool::TZM_HIDE_DESELECTED)
        //		{

        //		}
        else if (id == ZoomTool::TZM_UNHIDE_ALL)
        {
            UnhideDataElementCommand *command = new UnhideDataElementCommand(getProjectData()->getHiddenElements(), NULL);
            executeCommand(command);
        }
        else if ((id == ZoomTool::TZM_ZOOMIN) || (id == ZoomTool::TZM_ZOOMOUT) || (id == ZoomTool::TZM_ZOOMTO))
        {
            notifySignals();
        }
    }
}


void
TopviewGraph::mouseAction(MouseAction *mouseAction)
{
    graphScene_->mouseAction(mouseAction);
}

void
TopviewGraph::keyAction(KeyAction *keyAction)
{
    graphScene_->keyAction(keyAction);
}

void 
    TopviewGraph::wheelAction(WheelAction *wheelAction)
{
    notifySignals();
}

/*! \brief Called right before the editor will be changed.
*
* Deselect all items.
*/
void
TopviewGraph::preEditorChange()
{
    graphScene_->clearSelection();
}

/*! \brief Called right after the editor has been changed.
*
*/
void
TopviewGraph::postEditorChange()
{
}

//################//
// OBSERVER       //
//################//

void
TopviewGraph::updateObserver()
{

    // Get change flags //
    //
    int changes = getProjectData()->getProjectDataChanges();

    // Scene Size //
    //
    if ((changes & ProjectData::CPD_SizeChange))
    {
        updateSceneSize();
    }

    //	if((changes & ProjectData::CPD_ActiveElementChange))
    //	{

    //	}
}
