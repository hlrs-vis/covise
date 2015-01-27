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

#ifndef HEIGHTGRAPH_HPP
#define HEIGHTGRAPH_HPP

#include "projectgraph.hpp"

class ProfileGraphScene;
class ProfileGraphView;

class HeightGraph : public ProjectGraph
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit HeightGraph(QWidget *parent, ProjectWidget *projectWidget, ProjectData *projectData);
    virtual ~HeightGraph();

    ProfileGraphScene *getScene() const
    {
        return scene_;
    }
    ProfileGraphView *getView() const
    {
        return view_;
    }

protected:
private:
    HeightGraph(); /* not allowed */
    HeightGraph(const HeightGraph &); /* not allowed */
    HeightGraph &operator=(const HeightGraph &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    ProfileGraphScene *scene_;
    ProfileGraphView *view_;
};

#endif // HEIGHTGRAPH_HPP
