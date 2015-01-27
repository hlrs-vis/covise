/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   31.03.2010
**
**************************************************************************/

#ifndef TOOLACTION_HPP
#define TOOLACTION_HPP

#include "src/util/odd.hpp"

class ProjectWidget;
class ProjectData;
class ProjectGraph;
class ProjectEditor;

class ToolAction
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ToolAction(ODD::EditorId editorId = ODD::ENO_EDITOR, ODD::ToolId toolId = ODD::TNO_TOOL);
    virtual ~ToolAction()
    { /* does nothing */
    }

    ODD::EditorId getEditorId() const
    {
        return editorId_;
    }
    ODD::ToolId getToolId() const
    {
        return toolID_;
    }

protected:
private:
    //	ToolAction(); /* not allowed */
    ToolAction(const ToolAction &); /* not allowed */
    ToolAction &operator=(const ToolAction &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
    ODD::EditorId editorId_;
    ODD::ToolId toolID_;

private:
};

#endif // TOOLACTION_HPP
