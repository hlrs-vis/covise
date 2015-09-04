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

#ifndef PROJECTGRAPH_HPP
#define PROJECTGRAPH_HPP

#include <QWidget>
#include "src/data/observer.hpp"

#include "src/util/odd.hpp"

class ProjectWidget;

class ProjectData;
class DataCommand;

class ToolAction;
class MouseAction;
class KeyAction;

// Qt //
//
#include <QList>
class QGraphicsItem;

class ProjectGraph : public QWidget, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectGraph(ProjectWidget *projectWidget, ProjectData *projectData);
    virtual ~ProjectGraph();

    ProjectWidget *getProjectWidget() const
    {
        return projectWidget_;
    }
    ProjectData *getProjectData() const
    {
        return projectData_;
    }

    void addToGarbage(QGraphicsItem *);

    // Commands //
    //
    bool executeCommand(DataCommand *command);
    void beginMacro(const QString &text);
    void endMacro();

    // Observer Pattern //
    //
    virtual void updateObserver();

    void postponeGarbageDisposal();

    void finishGarbageDisposal();

private:
    ProjectGraph(); /* not allowed */
    ProjectGraph(const ProjectGraph &); /* not allowed */
    ProjectGraph &operator=(const ProjectGraph &); /* not allowed */

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

    // Garbage //
    //
    void garbageDisposal();

    //################//
    // PROPERTIES     //
    //################//

private:
    ProjectWidget *projectWidget_; // Project

    ProjectData *projectData_; // Model

    QList<QGraphicsItem *> garbageList_;

    int numPostpones;
};

#endif // PROJECTGRAPH_HPP
