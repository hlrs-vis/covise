/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   6/11/2010
**
**************************************************************************/

#include "maptool.hpp"

#include "toolmanager.hpp"
#include "src/mainwindow.hpp"

// Qt //
//
#include <QWidget>
#include <QToolBar>
#include <QAction>
#include <QComboBox>
#include <QMenu>
#include <QLabel>

//################//
//                //
// MapTool       //
//                //
//################//

MapTool::MapTool(ToolManager *toolManager)
    : Tool(toolManager)
    ,
    //		keepRatio_(true),
    //		lastX_(0.0),
    //		lastY_(0.0),
    //		lastWidth_(100.0),
    //		lastHeight_(100.0),
    active_(false)
{
    // Connect //
    //
    connect(this, SIGNAL(toolAction(ToolAction *)), toolManager, SLOT(toolActionSlot(ToolAction *)));

    QLabel *opacityLabel = new QLabel(" Opacity: ");

    opacityComboBox_ = new QComboBox();
    QStringList opacities;
    opacities << tr("100%") << tr("90%") << tr("80%") << tr("70%") << tr("60%") << tr("50%") << tr("40%") << tr("30%") << tr("20%") << tr("10%");
    opacityComboBox_->addItems(opacities);
    opacityComboBox_->setCurrentIndex(0);
    opacityComboBox_->setStatusTip(tr("Set Map Opacity."));
    opacityComboBox_->setToolTip(tr("Set Map Opacity"));
    connect(opacityComboBox_, SIGNAL(currentIndexChanged(QString)), this, SLOT(setOpacity(QString)));

    loadMapAction_ = new QAction(tr("Load &Map"), this);
    loadMapAction_->setStatusTip(tr("Load a background image."));
    connect(loadMapAction_, SIGNAL(triggered()), this, SLOT(loadMap()));

    loadGoogleAction_ = new QAction(tr("Load &Google Map"), this);
    loadGoogleAction_->setStatusTip(tr("Load a Google Maps image from the selected location."));
    connect(loadGoogleAction_, SIGNAL(triggered()), this, SLOT(loadGoogleMap()));

    deleteMapAction_ = new QAction(tr("&Delete Map"), this);
    deleteMapAction_->setStatusTip(tr("Delete the selected background images."));
    connect(deleteMapAction_, SIGNAL(triggered()), this, SLOT(deleteMap()));

    lockMapAction_ = new QAction(tr("&Lock Maps"), this);
    lockMapAction_->setStatusTip(tr("Toggle locking of the maps."));
    lockMapAction_->setCheckable(true);
    lockMapAction_->setChecked(true);
    connect(lockMapAction_, SIGNAL(triggered(bool)), this, SLOT(lockMap(bool)));

    //	QLabel * xLabel = new QLabel(" x: ");
    //	xLineEdit_ = new QDoubleSpinBox();
    //	xLineEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    //	xLineEdit_->setRange(-1000000.0, 1000000.0);
    //	xLineEdit_->setValue(lastX_);
    //	xLineEdit_->setMinimumWidth(100);
    //	xLineEdit_->setMaximumWidth(100);
    //	connect(xLineEdit_, SIGNAL(editingFinished()), this, SLOT(setX()));

    //	QLabel * yLabel = new QLabel(" y: ");
    //	yLineEdit_ = new QDoubleSpinBox();
    //	yLineEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    //	yLineEdit_->setRange(-1000000.0, 1000000.0);
    //	yLineEdit_->setValue(lastY_);
    //	yLineEdit_->setMinimumWidth(100);
    //	yLineEdit_->setMaximumWidth(100);
    //	connect(yLineEdit_, SIGNAL(editingFinished()), this, SLOT(setY()));

    //	QLabel * wLabel = new QLabel(" w: ");
    //	widthLineEdit_ = new QDoubleSpinBox();
    //	widthLineEdit_->setRange(1.0, 1000000.0);
    //	widthLineEdit_->setValue(lastWidth_);
    //	widthLineEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    //	widthLineEdit_->setMinimumWidth(100);
    //	widthLineEdit_->setMaximumWidth(100);
    //	connect(widthLineEdit_, SIGNAL(editingFinished()), this, SLOT(setWidth()));

    //	QLabel * hLabel = new QLabel(" h: ");
    //	heightLineEdit_ = new QDoubleSpinBox();
    //	heightLineEdit_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    //	heightLineEdit_->setRange(1.0, 1000000.0);
    //	heightLineEdit_->setValue(lastHeight_);
    //	heightLineEdit_->setMinimumWidth(100);
    //	heightLineEdit_->setMaximumWidth(100);
    //	connect(heightLineEdit_, SIGNAL(editingFinished()), this, SLOT(setHeight()));

    // Deactivate if no project //
    //
    connect(ODD::instance()->mainWindow(), SIGNAL(hasActiveProject(bool)), this, SLOT(activateProject(bool)));

    // ToolBar //
    //
    QToolBar *mapToolBar = new QToolBar(tr("Map"));
    mapToolBar->addWidget(opacityLabel);
    mapToolBar->addWidget(opacityComboBox_);
    mapToolBar->addAction(loadMapAction_);
    mapToolBar->addAction(loadGoogleAction_);
    mapToolBar->addAction(deleteMapAction_);
    mapToolBar->addAction(lockMapAction_);
    //	mapToolBar->addWidget(xLabel);
    //	mapToolBar->addWidget(xLineEdit_);
    //	mapToolBar->addWidget(yLabel);
    //	mapToolBar->addWidget(yLineEdit_);
    //	mapToolBar->addWidget(wLabel);
    //	mapToolBar->addWidget(widthLineEdit_);
    //	mapToolBar->addWidget(hLabel);
    //	mapToolBar->addWidget(heightLineEdit_);

    // ToolManager //
    //
    ODD::instance()->mainWindow()->addToolBar(mapToolBar);

    // View Menu //
    //
    QMenu *viewMenu = ODD::instance()->mainWindow()->getViewMenu();

    QMenu *mapMenu = new QMenu("Background Images", viewMenu);
    mapMenu->addAction(loadMapAction_);
    mapMenu->addAction(loadGoogleAction_);
    mapMenu->addAction(deleteMapAction_);
    mapMenu->addAction(lockMapAction_);

    viewMenu->addSeparator();
    viewMenu->addMenu(mapMenu);
}

//################//
// SLOTS          //
//################//

/*! \brief.
*/
void
MapTool::activateProject(bool active)
{
    active_ = active;

    // Enable/Disable //
    //
    opacityComboBox_->setEnabled(active_);
    loadMapAction_->setEnabled(active_);
    deleteMapAction_->setEnabled(active_);
    lockMapAction_->setEnabled(active_);
    loadGoogleAction_->setEnabled(active_);
    //	xLineEdit_->setEnabled(active_);
    //	yLineEdit_->setEnabled(active_);
    //	widthLineEdit_->setEnabled(active_);
    //	heightLineEdit_->setEnabled(active_);

    // Rulers //
    //
    lockMap(lockMapAction_->isChecked()); // it would be better to get the project's value somehow!
}

/*! \brief.
*/
void
MapTool::setOpacity(const QString &opacity)
{
    MapToolAction *action = new MapToolAction(opacity);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
MapTool::loadMap()
{
    MapToolAction *action = new MapToolAction(MapTool::TMA_LOAD);
    emit toolAction(action);
    delete action;
}


/*!
*/
void
MapTool::loadGoogleMap()
{
    MapToolAction *action = new MapToolAction(MapTool::TMA_GOOGLE);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
MapTool::deleteMap()
{
    MapToolAction *action = new MapToolAction(MapTool::TMA_DELETE);
    emit toolAction(action);
    delete action;
}

/*!
*/
void
MapTool::lockMap(bool lock)
{
    /*	opacityComboBox_->setEnabled(!lock && active_);
	loadMapAction_->setEnabled(!lock && active_);
	deleteMapAction_->setEnabled(!lock && active_);*/
    //	widthLineEdit_->setEnabled(!lock && active_);
    //	heightLineEdit_->setEnabled(!lock && active_);
    // x and y?

    MapToolAction *action = new MapToolAction(MapTool::TMA_LOCK);
    action->setToggled(lock);
    emit toolAction(action);
    delete action;
}

///*!
//*/
//void
//	MapTool
//	::setX()
//{
//	MapToolAction * action = new MapToolAction(MapTool::TMA_X);
//	action->setX(xLineEdit_->value());
//	emit toolAction(action);
//	delete action;

//	lastX_ = xLineEdit_->value();
//}

///*!
//*/
//void
//	MapTool
//	::setY()
//{
//	MapToolAction * action = new MapToolAction(MapTool::TMA_Y);
//	action->setY(yLineEdit_->value());
//	emit toolAction(action);
//	delete action;

//	lastY_ = yLineEdit_->value();
//}

///*!
//*/
//void
//	MapTool
//	::setWidth()
//{
////	if(lastWidth_ != widthLineEdit_->value())
//	{
//		MapToolAction * action = new MapToolAction(MapTool::TMA_WIDTH);
//		action->setWidth(widthLineEdit_->value(), keepRatio_);
//		emit toolAction(action);
//		delete action;

//		lastWidth_ = widthLineEdit_->value();
//	}
//}

///*!
//*/
//void
//	MapTool
//	::setHeight()
//{
////	if(lastHeight_ != heightLineEdit_->value())
//	{
//		MapToolAction * action = new MapToolAction(MapTool::TMA_HEIGHT);
//		action->setHeight(heightLineEdit_->value(), keepRatio_);
//		emit toolAction(action);
//		delete action;

//		lastHeight_ = heightLineEdit_->value();
//	}
//}

//################//
//                //
// MapToolAction //
//                //
//################//

// Note: This is not a typical Editor/Tool combination since this is not bound to
// a specify editor! So ENO_EDITOR and TNO_TOOL is set (Otherwise an editor would
// be loaded).

MapToolAction::MapToolAction(MapTool::MapToolId mapToolId)
    : ToolAction(ODD::ENO_EDITOR, ODD::TNO_TOOL)
    , mapToolId_(mapToolId)
    , opacity_("")
    , toggled_(true) /*,
		x_(0.0),
		y_(0.0),
		width_(0.0),
		height_(0.0)*/
{
}

MapToolAction::MapToolAction(const QString &opacity)
    : ToolAction(ODD::ENO_EDITOR, ODD::TNO_TOOL)
    , mapToolId_(MapTool::TMA_OPACITY)
    , opacity_(opacity)
    , toggled_(true) /*,
		x_(0.0),
		y_(0.0),
		width_(0.0),
		height_(0.0)*/
{
}

void
MapToolAction::setToggled(bool toggled)
{
    toggled_ = toggled;
}

//void
//	MapToolAction
//	::setX(double x)
//{
//	x_ = x;
//}

//void
//	MapToolAction
//	::setY(double y)
//{
//	y_ = y;
//}

//void
//	MapToolAction
//	::setWidth(double width, bool keepRatio)
//{
//	width_ = width;
//	keepRatio_ = keepRatio;
//}

//void
//	MapToolAction
//	::setHeight(double height, bool keepRatio)
//{
//	height_ = height;
//	keepRatio_ = keepRatio;
//}
