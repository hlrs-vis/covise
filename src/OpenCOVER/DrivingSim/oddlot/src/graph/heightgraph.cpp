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

#include "heightgraph.hpp"

#include "src/gui/projectwidget.hpp"

#include "src/data/projectdata.hpp"

// Graph //
//
#include "src/graph/profilegraphscene.hpp"
#include "src/graph/profilegraphview.hpp"

// Qt //
//
#include <QGridLayout>

HeightGraph::HeightGraph(QWidget *parent, ProjectWidget *projectWidget, ProjectData *projectData)
    : ProjectGraph(projectWidget, projectData)
{
    setParent(parent);
    // Qt Scene //
    //
    scene_ = new ProfileGraphScene(QRectF(-1000.0, 0.0, 20000.0, 20.0), this); // x, y, sizeX, sizeY
    qDebug("TODO HeightGraph Scene size");
    // TODO!!!
    //	scene_ = new HeightGraphScene(projectData_->getRoadSystem()->getRectF(), this);
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
    setLayout(layout);
}

/*! \brief The Destructor deletes the Qt Scene and Qt View.
*
*/
HeightGraph::~HeightGraph()
{
    delete view_;
    delete scene_;
}

///*! \brief Remove all registered items.
//*/
//void
//	HeightGraph
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
