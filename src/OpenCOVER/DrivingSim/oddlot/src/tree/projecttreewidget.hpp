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

#ifndef PROJECTTREEWIDGET_HPP
#define PROJECTTREEWIDGET_HPP

#include <QTreeWidget>

class ProjectData;

class ProjectTree;

class RoadSystemTreeItem;
class TileSystemTreeItem;
class VehicleSystemTreeItem;
class PedestrianSystemTreeItem;
class ScenerySystemTreeItem;

class ProjectTreeWidget : public QTreeWidget
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectTreeWidget(ProjectTree *projectTree, ProjectData *projectData);
    virtual ~ProjectTreeWidget();

    ProjectTree *getProjectTree() const
    {
        return projectTree_;
    }

    ProjectData *getProjectData() const
    {
        return projectData_;
    }

protected:
private:
    ProjectTreeWidget(); /* not allowed */
    ProjectTreeWidget(const ProjectTreeWidget &); /* not allowed */
    ProjectTreeWidget &operator=(const ProjectTreeWidget &); /* not allowed */

    void init();

    //################//
    // EVENTS         //
    //################//

public:
    void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

    //################//
    // PROPERTIES     //
    //################//

private:
    ProjectTree *projectTree_; // linked

    ProjectData *projectData_; // Model, linked

    RoadSystemTreeItem *roadSystemTreeItem_; // Tree, owned
    TileSystemTreeItem *tileSystemTreeItem_; // Tree, owned
    VehicleSystemTreeItem *vehicleSystemTreeItem_; // Tree, owned
    PedestrianSystemTreeItem *pedestrianSystemTreeItem_; // Tree, owned
    ScenerySystemTreeItem *scenerySystemTreeItem_; // Tree, owned
};

#endif // PROJECTTREEWIDGET_HPP
