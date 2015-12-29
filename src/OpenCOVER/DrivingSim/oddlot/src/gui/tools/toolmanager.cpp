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

// Tools //
//
#include "tool.hpp"
#include "toolwidget.hpp"

#include "zoomtool.hpp"
#include "selectiontool.hpp"

#include "roadlinkeditortool.hpp"
#include "typeeditortool.hpp"
#include "trackeditortool.hpp"
#include "elevationeditortool.hpp"
#include "superelevationeditortool.hpp"
#include "crossfalleditortool.hpp"
#include "laneeditortool.hpp"
#include "junctioneditortool.hpp"
#include "signaleditortool.hpp"
#include "osceditortool.hpp"

#include "maptool.hpp"

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
    initTools();
}

//	ToolManager
//	::~ToolManager()
//{
//	// TODO
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
    connect(toolBox_, SIGNAL(currentChanged(int)), widget, SLOT(activateWidget(int)));
    // parentship of the widget is not set? TODO (toolmanager is no widget)
}

void
ToolManager::addRibbonWidget(ToolWidget *widget, const QString &title)
{
    int index = ribbon_->addTab(widget,title);
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

    QString ribbonStyle = "\
QTabWidget::pane { /* The tab widget frame */\
    border-top: 1px solid  rgb(230, 230, 230);\
}\
QTabWidget::tab-bar {\
    left: 5px; /* move to the right by 5px */\
}\
\
/* Style the tab using the tab sub-control. Note that\
    it reads QTabBar _not_ QTabWidget */\
QTabBar::tab {\
    background: #3e3e3e;\
    color: white;\
    border: 2px #3e3e3e;\
    border-bottom-color:  rgb(230, 230, 230); /* same as the pane color */\
    border-top-left-radius: 0px;\
    border-top-right-radius: 0px;\
    min-width: 8ex;\
    padding: 2px;\
}\
\
QTabBar::tab:selected, QTabBar::tab:hover {\
   \
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\
                                stop: 0  rgb(200, 200, 200), stop: 1.0  rgb(230, 230, 230));\
    border-top-left-radius: 4px;\
    border-top-right-radius: 4px;\
    border: 2px white;\
    color: black;\
}\
\
QTabBar::tab:selected {\
     \
    color: black;\
    border-color: #525252;\
    border-bottom-color:  rgb(230, 230, 230); /* same as pane color */\
}\
\
QTabBar::tab:!selected {\
    margin-top: 2px; /* make non-selected tabs look smaller */\
}\
    \
/* make use of negative margins for overlapping tabs */\
QTabBar::tab:selected {\
    /* expand/overlap to the left and right by 4px */\
    margin-left: -4px;\
    margin-right: -4px;\
}\
\
QGroupBox {\
    background-color: rgb(230, 230, 230);\
    border: 0px solid gray;\
    border-bottom: 4ex solid rgb(220, 220, 220);\
    border-radius: 0px;\
    margin-top: 0px; /* leave space at the top for the title */\
    padding-bottom: 5ex; /* leave space at the top for the title */\
}\
\
QGroupBox::title {\
    subcontrol-origin: margin;\
    subcontrol-position: bottom center; /* position at the top center */\
    padding: -2 0px;\
    background-color: rgb(220, 220, 220);\
    color: black;\
}\
QPushButton {\
    background-color: rgb(200,200,200);\
    color: black;\
    border-style: outset;\
    border-width: 2px;\
    border-radius: 4px;\
    border-color: beige;\
    min-width: 2em;\
}\
QPushButton:open {\
    background-color:  rgb(100,100,100);\
    border-style: inset;\
    color: white;\
}\
QPushButton:pressed {\
    background-color:  rgb(100,100,100);\
    border-style: outset;\
    color: white;\
}\
QLabel {\
    background-color: rgb(230, 230, 230);\
    color: black;\
    }\
QTextEdit {\
    background-color: rgb(230, 230, 230);\
    color: black;\
    }\
QLineEdit {\
    background-color: rgb(230, 230, 230);\
    color: black;\
    }\
QSpinBox {\
    background-color: rgb(230, 230, 230);\
    color: black;\
    }\
QDoubleSpinBox {\
    background-color: rgb(230, 230, 230);\
    color: black;\
    }\
";
    ribbon_->setStyleSheet(ribbonStyle);


    // Zoom //
    //
    zoomTool_ = new ZoomTool(this);

    // Selection //
    //
    selectionTool_ = new SelectionTool(this);

    // RoadLinkEditor //
    //
    RoadLinkEditorTool *defaultEditor = new RoadLinkEditorTool(this);

    // TypeEditor //
    //
    new TypeEditorTool(this);

    // TrackEditor //
    //
    new TrackEditorTool(prototypeManager_, this);

    // ElevationEditor //
    //
    new ElevationEditorTool(this);

    // SuperelevationEditor //
    //
    new SuperelevationEditorTool(this);

    // CrossfallEditor //
    //
    new CrossfallEditorTool(this);

    // LaneEditor //
    //
    new LaneEditorTool(this);

    // JunctionEditor //
    new JunctionEditorTool(this);

    // Signal and Object Editor
    new SignalEditorTool(this);

    // Map //
    //
    new MapTool(this);

	// OpenScenario //
	//
	new OpenScenarioEditorTool(this);

    // Default //
    //
    defaultEditor->activateEditor();
}

/*! Resends a toolAction with the last EditorId and ToolId.
*/
void
ToolManager::resendCurrentTool()
{
    if (lastToolAction_)
    {
        emit(toolAction(lastToolAction_));
    }
}

//################//
// SLOTS          //
//################//

void
ToolManager::toolActionSlot(ToolAction *action)
{
    if (action->getEditorId() != ODD::ENO_EDITOR && action->getToolId() != ODD::TNO_TOOL)
    {
        lastToolAction_ = new ToolAction(action->getEditorId(), action->getToolId());
    }
    emit(toolAction(action));
}
