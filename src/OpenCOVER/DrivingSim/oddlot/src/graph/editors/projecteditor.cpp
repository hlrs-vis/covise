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

#include "projecteditor.hpp"

// Project //
//
#include "src/gui/projectwidget.hpp"

// Data //
//
#include "src/data/projectdata.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"

// Qt //
//
#include <QStatusBar>

// Utils //
//
#include "src/mainwindow.hpp"

ProjectEditor::ProjectEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : QObject(projectWidget)
    , projectWidget_(projectWidget)
    , projectData_(projectData)
    , topviewGraph_(topviewGraph)
    , currentTool_(ODD::TNO_TOOL)
{
}

ProjectEditor::~ProjectEditor()
{
}

ProjectGraph *
ProjectEditor::getProjectGraph() const
{
    return topviewGraph_;
}

//################//
// TOOL           //
//################//

/*! \brief Called when a tool button has been triggered.
*
* Sets the tool.
*/
void
ProjectEditor::toolAction(ToolAction *toolAction)
{
    // Change Tool if necessary //
    //
    ODD::ToolId id = toolAction->getToolId();
    if (id != ODD::TNO_TOOL)
    {
        setTool(id);
    }
}

/*! \brief Sets the active tool.
*
*/
void
ProjectEditor::setTool(ODD::ToolId id)
{
    currentTool_ = id;
}

//################//
// MOUSE & KEY    //
//################//

/*! \brief Does nothing. To be implemented by child classes.
*
*/
void
ProjectEditor::mouseAction(MouseAction *mouseAction)
{
}

/*! \brief Does nothing. To be implemented by child classes.
*
*/
void
ProjectEditor::keyAction(KeyAction *keyAction)
{
}

//################//
// STATUS BAR     //
//################//

/*! \brief .
*
*/
void
ProjectEditor::printStatusBarMsg(const QString &text, int milliseconds)
{
    ODD::mainWindow()->statusBar()->showMessage(text, milliseconds);
}

//################//
// SLOTS          //
//################//

/*! \brief Called when editor is activated.
*
* Calls the virtual init() function.
*/
void
ProjectEditor::show()
{
    // Init (Factory Method, to be implemented by child classes //
    //
    init();
}

/*! \brief Deletes the items created by this editor.
*
* Calls the virtual kill() function.
*/
void
ProjectEditor::hide()
{
    // Kill (Factory Method, to be implemented by child classes //
    //
    kill();
}
