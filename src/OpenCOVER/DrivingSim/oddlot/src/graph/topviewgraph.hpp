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

#ifndef TOPVIEWGRAPH_HPP
#define TOPVIEWGRAPH_HPP

#include "projectgraph.hpp"

class GraphScene;
class GraphView;

class TopviewGraph : public ProjectGraph
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TopviewGraph(ProjectWidget *projectWidget, ProjectData *projectData);
    virtual ~TopviewGraph();

    void updateSceneSize();

    GraphScene *getScene() const
    {
        return graphScene_;
    }
    GraphView *getView() const
    {
        return graphView_;
    }

    // Observer Pattern //
    //
    virtual void updateObserver();

private:
    TopviewGraph(); /* not allowed */
    TopviewGraph(const TopviewGraph &); /* not allowed */
    TopviewGraph &operator=(const TopviewGraph &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:

    // Tools, Mouse & Key //
    //
    void toolActionSignal(ToolAction *);
    void mouseActionSignal(MouseAction *);
    void keyActionSignal(KeyAction *);

    //################//
    // SLOTS          //
    //################//

public slots:

    // Tools, Mouse & Key //
    //
    void toolAction(ToolAction *);
    void mouseAction(MouseAction *);
    void keyAction(KeyAction *);

    // Change //
    //
    void preEditorChange();
    void postEditorChange();

    //################//
    // PROPERTIES     //
    //################//

private:
    GraphScene *graphScene_; // Qt Model
    GraphView *graphView_; // Qt View
};

#endif // TOPVIEWGRAPH_HPP
