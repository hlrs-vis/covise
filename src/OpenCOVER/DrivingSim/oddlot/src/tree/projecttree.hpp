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

#ifndef PROJECTTREE_HPP
#define PROJECTTREE_HPP

#include <QWidget>

class ProjectWidget;

class ProjectData;

class ProjectTreeWidget;

class ProjectTreeItem;

class ProjectTree : public QWidget
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    enum Roles
    {
        PTR_Selection = 1,
        PTR_SelectInViews = 2,
        PTR_DeselectInViews = 3
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectTree(ProjectWidget *projectWidget, ProjectData *projectData);
    virtual ~ProjectTree()
    { /* does nothing */
    }

    ProjectWidget *getProjectWidget() const
    {
        return projectWidget_;
    }
    ProjectData *getProjectData() const
    {
        return projectData_;
    }

    ProjectTreeWidget *getProjectTreeWidget() const
    {
        return projectTreeWidget_;
    }

    void addToGarbage(ProjectTreeItem *);

protected:
private:
    ProjectTree(); /* not allowed */
    ProjectTree(const ProjectTree &); /* not allowed */
    ProjectTree &operator=(const ProjectTree &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

public:
//################//
// SIGNALS        //
//################//

signals:

    //################//
    // SLOTS          //
    //################//

public slots:

    void projectActivated(bool active);

    void garbageDisposal();

    //################//
    // PROPERTIES     //
    //################//

private:
    ProjectWidget *projectWidget_; // Project, linked

    ProjectData *projectData_; // Model, linked

    ProjectTreeWidget *projectTreeWidget_; // owned

    QList<ProjectTreeItem *> garbageList_;
};

#endif // PROJECTTREE_HPP
