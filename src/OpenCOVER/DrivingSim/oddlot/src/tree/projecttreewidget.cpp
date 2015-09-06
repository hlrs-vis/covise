/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/11/2010
**
**************************************************************************/

#include "projecttreewidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"

// Tree //
//
#include "projecttree.hpp"
#include "items/roadsystemtreeitem.hpp"
#include "items/tilesystemtreeitem.hpp"
#include "items/vehiclesystem/vehiclesystemtreeitem.hpp"
#include "items/pedestriansystem/pedestriansystemtreeitem.hpp"
#include "items/scenerysystemtreeitem.hpp"

//################//
// CONSTRUCTOR    //
//################//

ProjectTreeWidget::ProjectTreeWidget(ProjectTree *projectTree, ProjectData *projectData)
    : QTreeWidget(projectTree)
    , projectTree_(projectTree)
    , projectData_(projectData)
{
    init();
}

ProjectTreeWidget::~ProjectTreeWidget()
{
    delete tileSystemTreeItem_;
    delete roadSystemTreeItem_;
    delete vehicleSystemTreeItem_;
    delete pedestrianSystemTreeItem_;
    delete scenerySystemTreeItem_;
}

//################//
// FUNCTIONS      //
//################//

void
ProjectTreeWidget::init()
{
    setSelectionMode(QAbstractItemView::ExtendedSelection);
    setUniformRowHeights(true);

    // Labels //
    //
    setHeaderLabels(QStringList() << tr(""));
    // Root Items //
    //
    QTreeWidgetItem *rootItem = invisibleRootItem();

    tileSystemTreeItem_ = new TileSystemTreeItem(projectTree_, projectData_->getTileSystem(), rootItem);
    roadSystemTreeItem_ = new RoadSystemTreeItem(projectTree_, projectData_->getRoadSystem(), rootItem);
    vehicleSystemTreeItem_ = new VehicleSystemTreeItem(projectTree_, projectData_->getVehicleSystem(), rootItem);
    pedestrianSystemTreeItem_ = new PedestrianSystemTreeItem(projectTree_, projectData_->getPedestrianSystem(), rootItem);
    scenerySystemTreeItem_ = new ScenerySystemTreeItem(projectTree_, projectData_->getScenerySystem(), rootItem);
}

//################//
// EVENTS         //
//################//

void
ProjectTreeWidget::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
    //	qDebug() << "selected: " << selected.indexes().count();
    //	qDebug() << "deselected: " << deselected.indexes().count();

    // Set the data of the item, so the item notices it's selection
    //
    foreach (QModelIndex index, deselected.indexes())
    {
        model()->setData(index, false, Qt::UserRole + ProjectTree::PTR_Selection);
    }

    foreach (QModelIndex index, selected.indexes())
    {
        model()->setData(index, true, Qt::UserRole + ProjectTree::PTR_Selection);
    }

    QTreeWidget::selectionChanged(selected, deselected);
}
