/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/11/2010
**
**************************************************************************/

#include "signaltreewidget.hpp"

// Manager //
//
#include "src/data/signalmanager.hpp" 

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/commands/signalcommands.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"
#include "src/gui/tools/signaleditortool.hpp"
#include "src/gui/tools/toolmanager.hpp"

// MainWindow//
//
#include "src/mainwindow.hpp"

//Settings//
//
#include "src/settings/projectsettings.hpp"

// Editor //
//
#include "src/graph/editors/signaleditor.hpp"

#include <QWidget>

//################//
// CONSTRUCTOR    //
//################//

SignalTreeWidget::SignalTreeWidget(SignalManager *signalManager, MainWindow *mainWindow)
	: QTreeWidget()
	, signalManager_(signalManager)
	, mainWindow_(mainWindow)
	, projectWidget_(NULL)
	, signalEditor_(NULL)
	, currentTool_(ODD::TNO_TOOL)
{
    init();
}

SignalTreeWidget::~SignalTreeWidget()
{
    
}

//################//
// FUNCTIONS      //
//################//

void
SignalTreeWidget::init()
{
	// Connect with the ToolManager to send the selected signal or object //
    //
	toolManager_ = mainWindow_->getToolManager();
	if (toolManager_)
	{
		connect(this, SIGNAL(toolAction(ToolAction *)), toolManager_, SLOT(toolActionSlot(ToolAction *)));
	}

    setSelectionMode(QAbstractItemView::ExtendedSelection);
    setUniformRowHeights(true);
	setIndentation(6);

	// Signals Widget //
    //
	setColumnCount(3);
	setColumnWidth(0, 180);
	setColumnWidth(1, 30);

	setHeaderHidden(true);
	QList<QTreeWidgetItem *> rootList;

	// Signals //
	//
	QTreeWidgetItem *signalsWidget = new QTreeWidgetItem;
	signalsWidget->setText(0, "Signals");
	rootList.append(signalsWidget);

	// Objects //
	//
	QTreeWidgetItem *objectsWidget = new QTreeWidgetItem;
	objectsWidget->setText(0, "Objects");
	rootList.append(objectsWidget);

	QList<QString> countries = signalManager_->getCountries();
	int categorySize = signalManager_->getCategoriesSize();
	int colorIndex = 360 / (categorySize + 1);

	for (int i = 0; i < countries.size(); i++)
	{
		QTreeWidgetItem *signalsCountryWidget = new QTreeWidgetItem(signalsWidget); 
		signalsCountryWidget->setText(0,countries.at(i));

		// Add Signals //
		QMap<QString, QTreeWidgetItem *> categoryMap;
		foreach (const SignalContainer *container, signalManager_->getSignals(countries.at(i)))
		{
			const QString &signCategory = container->getSignalCategory();
			//qDebug() << signCategory;
			QTreeWidgetItem *categoryWidget;
			if (categoryMap.contains(signCategory))
			{
				categoryWidget = categoryMap.value(signCategory);
			}
			else 
			{
				categoryWidget = new QTreeWidgetItem(signalsCountryWidget);
				categoryWidget->setText(0,signCategory);
				categoryWidget->setText(1,tr(""));
				categoryMap.insert(signCategory,categoryWidget); 
				QColor color;
	            color.setHsv(signalManager_->getCategoryNumber(signCategory) * colorIndex, 255, 255, 255);
				categoryWidget->setBackgroundColor(1, color);
			}
			QTreeWidgetItem *signs = new QTreeWidgetItem(categoryWidget); 
			signs->setText(0,container->getSignalName());
			signs->setIcon(0,container->getSignalIcon());	
		}

		QTreeWidgetItem *objectsCountryWidget = new QTreeWidgetItem(objectsWidget); 
		objectsCountryWidget->setText(0,countries.at(i));

		// Add objects //
		categoryMap.clear();
		foreach (const ObjectContainer *container, signalManager_->getObjects(countries.at(i)))
		{
			const QString &objectCategory = container->getObjectCategory();
			QTreeWidgetItem *categoryWidget;
			if (categoryMap.contains(objectCategory))
			{
				categoryWidget = categoryMap.value(objectCategory);
			}
			else 
			{
				categoryWidget = new QTreeWidgetItem(objectsCountryWidget);
				categoryWidget->setText(0,objectCategory);
				categoryWidget->setText(1,tr(""));
				categoryMap.insert(objectCategory, categoryWidget); 
				QColor color;
	            color.setHsv(signalManager_->getCategoryNumber(objectCategory) * colorIndex, 255, 255, 255);
				categoryWidget->setBackgroundColor(1, color);
			}
			QTreeWidgetItem *object = new QTreeWidgetItem(categoryWidget); 
			object->setText(0,container->getObjectType());	
			object->setIcon(0,container->getObjectIcon());
		}
	}
	// add bridge
	QTreeWidgetItem *bridgeWidget = new QTreeWidgetItem();
	bridgeWidget->setText(0, "Bridge");
	bridgeWidget->setText(1,tr(""));
	QColor color;
	color.setHsv((categorySize - 1) * colorIndex, 255, 255, 255);
	bridgeWidget->setBackgroundColor(1, color);
	rootList.append(bridgeWidget);

	// add tunnel
	QTreeWidgetItem *tunnelWidget = new QTreeWidgetItem();
	tunnelWidget->setText(0, "Tunnel");
	tunnelWidget->setText(1,tr(""));
	color.setHsv(categorySize * colorIndex, 255, 255, 255);
	tunnelWidget->setBackgroundColor(1, color);
	rootList.append(tunnelWidget);

	insertTopLevelItems(0,rootList);
}


void 
SignalTreeWidget::setSignalEditor(SignalEditor *signalEditor)
{
	signalEditor_ = signalEditor;
	
	if (!signalEditor_)
	{
		clearSelection();
	}
}

  
//################//
// EVENTS         //
//################//
void
SignalTreeWidget::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
	if (signalEditor_ && (selectedItems().size() > 0))
	{
		toolManager_->activateSignalSelection(false);

		QTreeWidgetItem *item = selectedItems().at(0);
		const QString text = item->text(0);
		QString country;
		QString category;
		QTreeWidgetItem *parentItem = item->parent();
		if (parentItem)
		{
			parentItem = parentItem->parent();
			if (parentItem)
			{
				country = parentItem->text(0);
			}
		}
		SignalContainer *signalContainer = signalManager_->getSignalContainer(country, text);
		if (signalContainer)				// selected item is a signal
		{
			signalManager_->setSelectedSignalContainer(signalContainer);
			currentTool_ = ODD::TSG_SIGNAL;

			if (signalEditor_ && projectWidget_ && parentItem && !country.isEmpty())
			{
//				const QString &country = signalManager_->getCountry(signalContainer);
				QString type = signalContainer->getSignalType();
				const QString &typeSubclass = signalContainer->getSignalTypeSubclass();
				QString subtype = signalContainer->getSignalSubType();
				double value = signalContainer->getSignalValue();


				foreach (DataElement *element, projectWidget_->getProjectData()->getSelectedElements())
				{
					Signal *signal = dynamic_cast<Signal *>(element);
					if (signal)
					{
						SetSignalPropertiesCommand *command = new SetSignalPropertiesCommand(signal, signal->getId(), signal->getName(), 
							signal->getT(), signal->getDynamic(), signal->getOrientation(), signal->getZOffset(), country, type, typeSubclass, 
							subtype, value, signal->getHeading(), signal->getPitch(), signal->getRoll(), signal->getUnit(), signal->getText(), 
							signal->getWidth(), signal->getHeight(), signal->getPole(), signal->getSize(), signal->getValidFromLane(), 
							signal->getValidToLane(), signal->getCrossingProbability(), signal->getResetTime(), NULL);
						projectWidget_->getProjectSettings()->executeCommand(command);
					}
				}
			}
		}
		else
		{

			ObjectContainer *objectContainer = signalManager_->getObjectContainer(text);
			if (objectContainer)				// selected item is an object
			{
				signalManager_->setSelectedObjectContainer(objectContainer);
				currentTool_ = ODD::TSG_OBJECT;
				if (signalEditor_ && projectWidget_)
				{
					const QString &country = signalManager_->getCountry(objectContainer);
					const QString &type = objectContainer->getObjectType();
					double length = objectContainer->getObjectLength();
					double width = objectContainer->getObjectWidth();
					double radius = objectContainer->getObjectRadius();
					double height = objectContainer->getObjectHeight();
					double heading = objectContainer->getObjectHeading();
					double repeatDistance = objectContainer->getObjectRepeatDistance();
					const QString &file = objectContainer->getObjectFile();


					foreach (DataElement *element, projectWidget_->getProjectData()->getSelectedElements())
					{
						Object *object = dynamic_cast<Object *>(element);

						if (object)
						{
							Object::ObjectProperties objectProps{ object->getT(), object->getOrientation(), object->getzOffset(), type, object->getValidLength(), length, width, radius, 
								height, heading, object->getPitch(), object->getRoll(), object->getPole() };
							Object::ObjectRepeatRecord repeatProps = object->getRepeatProperties();
							object->setRepeatDistance(repeatDistance);
							SetObjectPropertiesCommand *command = new SetObjectPropertiesCommand(object, object->getId(), object->getName(), objectProps, repeatProps, object->getTextureFileName());
							projectWidget_->getProjectSettings()->executeCommand(command);
						}
					}
				}
			}

			else if (text == "Bridge")
			{
				currentTool_ = ODD::TSG_BRIDGE;
			}
			else if (text == "Tunnel")
			{
				currentTool_ = ODD::TSG_TUNNEL;

			}
			else
			{
				currentTool_ = ODD::TSG_SELECT;
			}
		}

		if (signalEditor_)
		{
			// Set a tool //
			//
			SignalEditorToolAction *action = new SignalEditorToolAction(currentTool_);
			emit toolAction(action);
			delete action;
		}


		QTreeWidget::selectionChanged(selected, deselected);
	}
	else
	{
		clearSelection();
		clearFocus();
		update();
	}

}

