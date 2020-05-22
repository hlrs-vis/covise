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
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/topviewgraph.hpp"
#include "src/graph/graphscene.hpp"
#include "src/graph/items/roadsystem/roadlink/roadlinkhandle.hpp"
#include "src/graph/items/roadsystem/roadlink/roadlinksinkitem.hpp"


// GUI //
//
#include "src/gui/parameters/tool.hpp"
#include "src/gui/parameters/toolvalue.hpp"
#include "src/gui/parameters/toolparametersettings.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/keyaction.hpp"

// Qt //
//
#include <QStatusBar>
#include <QKeyEvent>

// Utils //
//
#include "src/mainwindow.hpp"

template
void ProjectEditor::setToolValue<Lane>(Lane *, const QString &);
template
void ProjectEditor::setToolValue<RSystemElementRoad>(RSystemElementRoad *, const QString &);
template
void ProjectEditor::setToolValue<RSystemElementJunction>(RSystemElementJunction *, const QString &);
template
void ProjectEditor::setToolValue<RoadLinkSinkItem>(RoadLinkSinkItem *, const QString &);
template
void ProjectEditor::setToolValue<RoadLinkHandle>(RoadLinkHandle *, const QString &);
template
void ProjectEditor::createToolParameters<RSystemElementRoad>(RSystemElementRoad *object);
template
void ProjectEditor::removeToolParameters<RSystemElementRoad>(RSystemElementRoad *object);

ProjectEditor::ProjectEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph)
    : QObject(projectWidget)
    , projectWidget_(projectWidget)
    , projectData_(projectData)
    , topviewGraph_(topviewGraph)
    , currentTool_(ODD::TNO_TOOL)
	, currentParameterTool_(ODD::TNO_TOOL)
	, tool_(NULL)
{
	mainWindow_ = ODD::mainWindow();
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

		id = toolAction->getParamToolId();
		setParameterTool(id);
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

/*! \brief Sets the active tool.
*
*/
void
ProjectEditor::setParameterTool(ODD::ToolId id)
{
	currentParameterTool_ = id;
}

//###############################//
// Editor ToolAction Parameters //
//###############################//

void
ProjectEditor::createToolParameterSettingsApplyBox(Tool *tool, const ODD::EditorId &editorID)
{
	ToolManager *toolManager = mainWindow_->getToolManager();

	settingsApplyBox_ = new ToolParameterSettingsApplyBox(this, toolManager, editorID, mainWindow_->getParameterDialogBox());
	settings_ = static_cast<ToolParameterSettings *>(settingsApplyBox_);

	generateToolParameterUI(tool);
}

void
ProjectEditor::createToolParameterSettings(Tool *tool, const ODD::EditorId &editorID)
{
	ToolManager *toolManager = mainWindow_->getToolManager();

	settings_ = new ToolParameterSettings(toolManager, editorID);

	generateToolParameterUI(tool);
}

void
ProjectEditor::deleteToolParameterSettings()
{
	delete settings_;
	delete tool_;
	tool_ = NULL;
}

void
ProjectEditor::generateToolParameterUI(Tool *tool)
{
	settings_->setTool(tool);
	settings_->generateUI(mainWindow_->getParameterBox());
}

void
ProjectEditor::updateToolParameterUI(ToolParameter *param)
{
	settings_->updateUI(param);
}

void
ProjectEditor::delToolParameters()
{
	settings_->deleteUI();
	delete tool_;
	tool_ = NULL;
}

template<class T>
void
ProjectEditor::setToolValue(T *object, const QString &valueDisplayed)
{
	int currentParamId = settings_->getCurrentParameterID();
	ToolParameter *p = tool_->getLastParam(currentParamId);

	ToolValue<T> *v = dynamic_cast<ToolValue<T> *>(p);
	v->setValue(object);
	p->setValueDisplayed(valueDisplayed);

	settings_->setObjectSelected(currentParamId, p->getValueDisplayed(), p->getText());
}

template<class T>
void
ProjectEditor::createToolParameters(T *object)
{

	ToolParameter *p = tool_->getLastParam(settings_->getCurrentParameterID());

	ToolValue<T> *v = dynamic_cast<ToolValue<T> *>(p);
	v->setValue(object);
	p->setText("Remove Object");
	p->setValueDisplayed(v->getValue()->getIdName());
	int objectCount = tool_->getObjectCount(p->getToolId(), p->getParamToolId());
	if (objectCount < tool_->getListSize())
	{
		settings_->setObjectSelected(tool_->getParamId(p), p->getValueDisplayed(), p->getText());
	}

	// clone this parameter, because we need a list //
	ToolValue<T> *param = v->parameterClone();
	param->setText("Select/Remove");

	tool_->readParams(param);
	settings_->addUI(tool_->getParamId(param), param, true);
}

template<class T>
void
ProjectEditor::removeToolParameters(T *object)
{
	ToolValue<T> *v = tool_->getValue<T>(object);
	if (v)
	{
		int id = tool_->deleteValue(v);
		settings_->removeUI(id);
	}
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
	if (keyAction->getKeyActionType() == KeyAction::ATK_PRESS)
	{
		QKeyEvent *keyEvent = keyAction->getEvent();
		if (keyEvent->key() == Qt::Key_Escape)
		{
			getTopviewGraph()->getScene()->deselectAll();
		}
	}
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

void
ProjectEditor::reject()
{
	settings_->hide();
}
