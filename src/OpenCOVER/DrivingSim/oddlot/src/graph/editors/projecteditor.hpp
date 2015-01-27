/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10.03.2010
**
**************************************************************************/

#ifndef PROJECTEDITOR_HPP
#define PROJECTEDITOR_HPP

#include <QObject>

#include "src/util/odd.hpp"

class ProjectWidget;
class ProjectData;
class ProjectGraph;
class TopviewGraph;

class ToolAction;
class MouseAction;
class KeyAction;

/** \brief MVC: Controller. Baseclass for all editors.
*
*
*/
class ProjectEditor : public QObject
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~ProjectEditor();

    // Tool //
    //
    virtual void toolAction(ToolAction *toolAction);
    void setTool(ODD::ToolId id);
    ODD::ToolId getCurrentTool()
    {
        return currentTool_;
    }
    bool isCurrentTool(ODD::ToolId toolId)
    {
        if (toolId == currentTool_)
            return true;
        else
            return false;
    }

    // Mouse & Key //
    //
    virtual void mouseAction(MouseAction *mouseAction);
    virtual void keyAction(KeyAction *keyAction);

    // Project, Data, Graph //
    //
    ProjectWidget *getProjectWidget() const
    {
        return projectWidget_;
    }
    ProjectData *getProjectData() const
    {
        return projectData_;
    }
    ProjectGraph *getProjectGraph() const;
    TopviewGraph *getTopviewGraph() const
    {
        return topviewGraph_;
    }

    // StatusBar //
    //
    void printStatusBarMsg(const QString &text, int milliseconds);

protected:
    virtual void init() = 0;
    virtual void kill() = 0;

private:
    ProjectEditor(); /* not allowed */
    ProjectEditor(const ProjectEditor &); /* not allowed */
    ProjectEditor &operator=(const ProjectEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:
    void show();
    void hide();

    //################//
    // PROPERTIES     //
    //################//

private:
    // Project, Data, Graph //
    //
    ProjectWidget *projectWidget_;
    ProjectData *projectData_;
    TopviewGraph *topviewGraph_;

    // Tool //
    //
    ODD::ToolId currentTool_;
};

#endif // PROJECTEDITOR_HPP
