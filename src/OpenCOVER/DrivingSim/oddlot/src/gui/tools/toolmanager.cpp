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

#include "toolmanager.hpp"

#include "src/mainwindow.hpp"

 // Tools //
 //
#include "editortool.hpp"
#include "toolwidget.hpp"

#include "zoomtool.hpp"
#include "selectiontool.hpp"

#include "roadlinkeditortool.hpp"
#include "typeeditortool.hpp"
#include "trackeditortool.hpp"
#include "elevationeditortool.hpp"
#include "superelevationeditortool.hpp"
#include "crossfalleditortool.hpp"
#include "shapeeditortool.hpp"
#include "laneeditortool.hpp"
#include "junctioneditortool.hpp"
#include "signaleditortool.hpp"
#include "osceditortool.hpp"

#include "maptool.hpp"

#include "src/gui/projectwidget.hpp"
#include "src/graph/editors/projecteditor.hpp"

#include "src/gui/parameters/toolparametersettings.hpp"
// Qt //
//
#include <QToolBox>

//################//
// CONSTRUCTOR    //
//################//

ToolManager::ToolManager(PrototypeManager *prototypeManager, QObject *parent)
    : QObject(parent)
    , prototypeManager_(prototypeManager)
    , lastToolAction_(NULL)
{
    mainWindow_ = dynamic_cast<MainWindow *>(parent);
    initTools();
}

// ToolManager
// ::~ToolManager()
//{
// // TODO
//}

void
ToolManager::addMenu(QMenu *menu)
{
    menus_.append(menu);
}

void
ToolManager::addToolBoxWidget(ToolWidget *widget, const QString &title)
{
    int index = toolBox_->addItem(widget, title);
    widget->setToolBoxIndex(index);
    //   connect(toolBox_, SIGNAL(currentChanged(int)), widget, SLOT(activateWidget(int)));
       // parentship of the widget is not set? TODO (toolmanager is no widget)
}

void
ToolManager::addRibbonWidget(ToolWidget *widget, const QString &title, int index)
{
    //  int index = ribbon_->addTab(widget,title);
    ribbon_->insertTab(index, widget, title);
    widget->setToolBoxIndex(index);
    connect(ribbon_, SIGNAL(currentChanged(int)), widget, SLOT(activateWidget(int)));
}
void
ToolManager::initTools()
{
    // Tool Box //
    //
    toolBox_ = new QToolBox();
    toolBox_->setMinimumWidth(190);

    // Ribbon based Toolbox
    //
    ribbon_ = new QTabWidget;
    ribbon_->setMinimumHeight(100);
    ribbon_->setMaximumHeight(100);


    // Zoom //
    //
    zoomTool_ = new ZoomTool(this);

    // Selection //
    //
 //   selectionTool_ = new SelectionTool(this);

    // RoadLinkEditor //
    //
    RoadLinkEditorTool *defaultEditor = new RoadLinkEditorTool(this);
    standardToolAction_.insert(ODD::ERL, new RoadLinkEditorToolAction(ODD::TRL_SELECT));

    // TypeEditor //
    //
    new TypeEditorTool(this);
    standardToolAction_.insert(ODD::ERT, new TypeEditorToolAction(ODD::TRT_MOVE));


    // TrackEditor //
    //
    new TrackEditorTool(this);
    standardToolAction_.insert(ODD::ETE, new TrackEditorToolAction(ODD::TTE_ROAD_MOVE_ROTATE));

    // ElevationEditor //
    //
    new ElevationEditorTool(this);
    standardToolAction_.insert(ODD::EEL, new ElevationEditorToolAction(ODD::TEL_SELECT, ODD::TNO_TOOL, 900.0, 0.0, 0.0));

    // SuperelevationEditor //
    //
    new SuperelevationEditorTool(this);
    standardToolAction_.insert(ODD::ESE, new SuperelevationEditorToolAction(ODD::TSE_SELECT, ODD::TNO_TOOL, 2000.0));

    // CrossfallEditor //
    //
    new CrossfallEditorTool(this);
    standardToolAction_.insert(ODD::ECF, new CrossfallEditorToolAction(ODD::TCF_SELECT, ODD::TNO_TOOL, 2000.0));

    // RoadShapeEditor //
    //
    new ShapeEditorTool(this);
    standardToolAction_.insert(ODD::ERS, new ShapeEditorToolAction(ODD::TRS_SELECT));

    // LaneEditor //
    //
    new LaneEditorTool(this);
    standardToolAction_.insert(ODD::ELN, new LaneEditorToolAction(ODD::TLE_SELECT, 0.0));

    // JunctionEditor //
    new JunctionEditorTool(this);
    standardToolAction_.insert(ODD::EJE, new JunctionEditorToolAction(ODD::TJE_SELECT));

    // Signal and Object Editor
    signalEditorTool_ = new SignalEditorTool(this);
    standardToolAction_.insert(ODD::ESG, new SignalEditorToolAction(ODD::TSG_SELECT));

    // Map //
    //
    new MapTool(this);

    // OpenScenario //
    //
    oscEditorTool_ = new OpenScenarioEditorTool(this);
    standardToolAction_.insert(ODD::EOS, new OpenScenarioEditorToolAction(ODD::TOS_SELECT, ""));

    // Default //
    //
    defaultEditor->activateEditor();
}

/*! Resends a toolAction with the last EditorId and ToolId.
*/
void
ToolManager::resendCurrentTool(ProjectWidget *project)
{
    ToolAction *lastToolAction = getProjectEditingState(project);
    if (lastToolAction)
    {
        ribbon_->setCurrentIndex(lastToolAction->getEditorId());

    }
}

void
ToolManager::resendStandardTool(ProjectWidget *project)
{
    ToolAction *lastToolAction = getProjectEditingState(project);
    lastToolAction = standardToolAction_.value(lastToolAction->getEditorId());
    setProjectEditingState(project, lastToolAction);

    ToolWidget *widget = dynamic_cast<ToolWidget *>(ribbon_->currentWidget());
    widget->activateWidget(lastToolAction->getEditorId());
}


void
ToolManager::enableOSCEditorToolButton(bool state)
{
    oscEditorTool_->enableGraphEdit(state);
}

void
ToolManager::activateSignalSelection(bool state)
{
    signalEditorTool_->signalSelection(state);
}

void
ToolManager::activateOSCObjectSelection(bool state)
{
    oscEditorTool_->objectSelection(state);
}

void
ToolManager::setPushButtonColor(const QString &name, QColor color)
{
    oscEditorTool_->setButtonColor(name, color);
}

void
ToolManager::addProjectEditingState(ProjectWidget *project, ToolAction *toolAction)
{
    if (!editingStates_.contains(project))
    {
        QList<ToolAction *> toolList;
        toolList.append(toolAction);
        editingStates_.insert(project, toolList);
    }
}

void
ToolManager::setProjectEditingState(ProjectWidget *project, ToolAction *toolAction)
{
    int editorId = toolAction->getEditorId();

    QMap<ProjectWidget *, QList<ToolAction *>>::iterator it = editingStates_.find(project);
    if (it != editingStates_.end())
    {
        QList<ToolAction *> *toolList = &it.value();
        if (toolList->first() == toolAction)
        {
            return;
        }

        foreach(ToolAction * tool, *toolList)
        {
            if (tool->getEditorId() == editorId)
            {
                ParameterToolAction *toolParams = dynamic_cast<ParameterToolAction *>(toolAction);
                if (toolParams)
                {
                    toolList->first()->setParamToolId(toolParams->getParamToolId());
                }
                else
                {
                    toolList->removeOne(tool);
                    if ((tool != standardToolAction_.value(editorId)) && (tool != toolAction))
                    {
                        delete tool;
                    }
                    toolList->prepend(toolAction);
                }
                return;
            }
        }
        toolList->prepend(toolAction);
    }
    else
    {
        addProjectEditingState(project, toolAction);
    }
}

// Gets the toolId of Editor editorId and inserts the related ToolAction as first element of the list
//
ToolAction *
ToolManager::getProjectEditingState(ProjectWidget *project, ODD::EditorId editorId)
{
    if (!project)
    {
        MainWindow *mainWindow = dynamic_cast<MainWindow *>(parent());
        project = mainWindow->getActiveProject();
    }

    QMap<ProjectWidget *, QList<ToolAction *>>::iterator it = editingStates_.find(project);
    if (it != editingStates_.end())
    {
        QList<ToolAction *> *toolList = &it.value();
        foreach(ToolAction * tool, *toolList)
        {
            if (tool->getEditorId() == editorId)
            {
                return tool;
            }
        }
    }

    return NULL;
}


ToolAction *
ToolManager::getProjectEditingState(ProjectWidget *project)
{
    QMap<ProjectWidget *, QList<ToolAction *>>::iterator it = editingStates_.find(project);
    if (it != editingStates_.end())
    {
        return it.value().first();
    }

    return NULL;
}


ToolAction *
ToolManager::getLastToolAction(ODD::EditorId editorID)
{
    ProjectWidget *currentProject = mainWindow_->getActiveProject();
    ToolAction *lastToolAction = getProjectEditingState(currentProject, editorID);
    if (!lastToolAction)
    {
        lastToolAction = standardToolAction_.value(editorID);
        setProjectEditingState(currentProject, lastToolAction);
    }

    return lastToolAction;
}


//################//
// SLOTS          //
//################//
void
ToolManager::loadProjectEditor(bool active)
{
    if (active) /* project has changed */
    {
        ProjectWidget *currentProject = mainWindow_->getActiveProject();

        ToolAction *lastToolAction = getProjectEditingState(currentProject);
        if (!lastToolAction)
        {
            lastToolAction = standardToolAction_.value(ODD::ERL);
            setProjectEditingState(currentProject, lastToolAction);
        }

        if (ribbon_->currentIndex() == lastToolAction->getEditorId())
        {
            ToolWidget *widget = dynamic_cast<ToolWidget *>(ribbon_->currentWidget());
            widget->activateWidget(lastToolAction->getEditorId());
        }
        else
        {
            ribbon_->setCurrentIndex(lastToolAction->getEditorId());
        }
    }
}

void
ToolManager::toolActionSlot(ToolAction *action)
{
    static QSet<ODD::ToolId> forgetEditingStateTools = QSet<ODD::ToolId>() << ODD::TNO_TOOL << ODD::TOS_CREATE_CATALOG << ODD::TOS_GRAPHELEMENT << ODD::TOS_ELEMENT;

    ODD::EditorId editorId = action->getEditorId();
    ODD::ToolId toolId = action->getToolId();

    ProjectWidget *currentProject = mainWindow_->getActiveProject();

    if (currentProject && (editorId != ODD::ENO_EDITOR))
    {
        if (!forgetEditingStateTools.contains(toolId))
        {

            setProjectEditingState(currentProject, action);

        }
    }

    emit(toolAction(action));
}
