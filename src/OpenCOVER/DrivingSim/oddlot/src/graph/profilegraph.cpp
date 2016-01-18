/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.06.2010
**
**************************************************************************/

#include "profilegraph.hpp"

#include "src/gui/projectwidget.hpp"

#include "src/data/projectdata.hpp"

// Graph //
//
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/profilegraphview.hpp"
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/editors/elevationeditor.hpp"

// Qt //
//
#include <QGridLayout>

ProfileGraph::ProfileGraph(ProjectWidget *projectWidget, ProjectData *projectData)
    : ProjectGraph(projectWidget, projectData)
{
    // Qt Scene //
    //
    scene_ = new ProfileGraphScene(QRectF(-1000.0, -1000.0, 20000.0, 2000.0), this); // x, y, sizeX, sizeY
    qDebug("TODO ProfileGraph Scene size");
    // TODO!!!
    //	scene_ = new ProfileGraphScene(projectData_->getRoadSystem()->getRectF(), this);
    // may be not ready yet

    //	connect(scene_, SIGNAL(mouseActionSignal(MouseAction*)), this, SIGNAL(mouseActionSignal(MouseAction*)));
    //	connect(scene_, SIGNAL(keyActionSignal(KeyAction*)), this, SIGNAL(keyActionSignal(KeyAction*)));

    // Qt View //
    //
    view_ = new ProfileGraphView(scene_, this);
    view_->setObjectName("ProfileGraphView");
    view_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    view_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    view_->setRenderHints(QPainter::Antialiasing);

    //	view_->resetMatrix();
    view_->resetViewTransformation();
    view_->centerOn(0.0, 0.0);

    //view_->setAlignment(Qt::AlignLeft | Qt::AlignTop); // place image at top left, center otherwise
    //view_->setCacheMode(QGraphicsView::CacheBackground); // speedup for tiles / many small pixmaps in viewport

    // Layout //
    //
    QGridLayout *layout = new QGridLayout();
    layout->setContentsMargins(0,0,0,0);
    layout->addWidget(view_);
    setLayout(layout);
}

ProfileGraph::ProfileGraph(ProjectWidget *projectWidget, ProjectData *projectData, qreal height)
    : ProjectGraph(projectWidget, projectData)
{
    // Qt Scene //
    //
    scene_ = new ProfileGraphScene(QRectF(-1000.0, -1.0, 20000.0, height), this); // x, y, sizeX, sizeY
    qDebug("TODO ProfileGraph Scene size");
    // TODO!!!
    //	scene_ = new ProfileGraphScene(projectData_->getRoadSystem()->getRectF(), this);
    // may be not ready yet

    //	connect(scene_, SIGNAL(mouseActionSignal(MouseAction*)), this, SIGNAL(mouseActionSignal(MouseAction*)));
    //	connect(scene_, SIGNAL(keyActionSignal(KeyAction*)), this, SIGNAL(keyActionSignal(KeyAction*)));

    // Qt View //
    //
    view_ = new ProfileGraphView(scene_, this);
    view_->setObjectName("ProfileGraphView");
    view_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    view_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    view_->setRenderHints(QPainter::Antialiasing);

    //	view_->resetMatrix();
    view_->resetViewTransformation();
    view_->centerOn(0.0, 0.0);

    //view_->setAlignment(Qt::AlignLeft | Qt::AlignTop); // place image at top left, center otherwise
    //view_->setCacheMode(QGraphicsView::CacheBackground); // speedup for tiles / many small pixmaps in viewport

    // Layout //
    //
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(view_);
    layout->setContentsMargins(0,0,0,0);
    setLayout(layout);
}

/*! \brief The Destructor deletes the Qt Scene and Qt View.
*
*/
ProfileGraph::~ProfileGraph()
{
    delete view_;
    delete scene_;
}

void
ProfileGraph::updateBoundingBox()
{
    // Show all selected roads in ProfileGraph //
    //

    ProjectEditor *editor = getProjectWidget()->getProjectEditor();

    ElevationEditor *elevationEditor = dynamic_cast<ElevationEditor *>(editor);
    if (elevationEditor)
    {
        elevationEditor->initBox();
        QMap<RSystemElementRoad *, ElevationRoadPolynomialItem *> selectedElevationItems = elevationEditor->getSelectedElevationItems();
        QMap<RSystemElementRoad *, ElevationRoadPolynomialItem *>::const_iterator iterator = selectedElevationItems.begin();
        while (iterator != selectedElevationItems.end())
        {
            if (iterator.value())
            {
                elevationEditor->addSelectedRoad(iterator.value());
                iterator++;
            }
        }
        elevationEditor->fitView();
    }
}

///*! \brief Remove all registered items.
//*/
//void
//	ProfileGraph
//	::garbageDisposal()
//{
//	foreach(QGraphicsItem * item, garbageList_)
//	{
////		if(item->scene())
////		{
////			item->scene()->removeItem(item);
////		}
////		else
////		{
//////			qDebug("WARNING 1006251515! Garbage disposal: Item has no scene.");
////		}
//		delete item;
//	}

//	garbageList_.clear();
//}

//################//
// SLOTS          //
//################//

/*! \brief .
*
*/
void
ProfileGraph::toolAction(ToolAction *toolAction)
{
    view_->toolAction(toolAction);
}

void
ProfileGraph::mouseAction(MouseAction * /*mouseAction*/)
{
    //	graphScene_->mouseAction(mouseAction);
}

void
ProfileGraph::keyAction(KeyAction * /*keyAction*/)
{
    //	graphScene_->keyAction(keyAction);
}

//################//
// OBSERVER       //
//################//

void
ProfileGraph::updateObserver()
{

    // Get change flags //
    //
    int changes = getProjectData()->getProjectDataChanges();

    // Selection Change //
    //
    if ((changes & ProjectData::CPD_SelectedElementsChanged))
    {
        updateBoundingBox();
    }
}