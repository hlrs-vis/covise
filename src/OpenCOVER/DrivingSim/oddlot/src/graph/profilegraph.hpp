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

#ifndef PROFILEGRAPH_HPP
#define PROFILEGRAPH_HPP

#include "projectgraph.hpp"

class ProfileGraphScene;
class ProfileGraphView;

class ProfileGraph : public ProjectGraph
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProfileGraph(ProjectWidget *projectWidget, ProjectData *projectData);
    explicit ProfileGraph(ProjectWidget *projectWidget, ProjectData *projectData, qreal height);
    virtual ~ProfileGraph();

    ProfileGraphScene *getScene() const
    {
        return scene_;
    }
    ProfileGraphView *getView() const
    {
        return view_;
    }

    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    ProfileGraph(); /* not allowed */
    ProfileGraph(const ProfileGraph &); /* not allowed */
    ProfileGraph &operator=(const ProfileGraph &); /* not allowed */

    void updateBoundingBox();

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // SLOTS          //
    //################//

public slots:

    // Tools, Mouse & Key //
    //
    void toolAction(ToolAction *);
    void mouseAction(MouseAction *);
    void keyAction(KeyAction *);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    ProfileGraphScene *scene_;
    ProfileGraphView *view_;
};

#endif // PROFILEGRAPH_HPP
