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

// MainWindow//
//
#include "src/mainwindow.hpp"

//Settings//
//
#include "src/settings/projectsettings.hpp"

#include <QWidget>

//################//
// CONSTRUCTOR    //
//################//

SignalTreeWidget::SignalTreeWidget(SignalManager *signalManager, MainWindow *mainWindow)
	: QTreeWidget()
	, signalManager_(signalManager)
	, mainWindow_(mainWindow)
	, projectWidget_(NULL)
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
    setSelectionMode(QAbstractItemView::ExtendedSelection);
    setUniformRowHeights(true);

	// Signals Widget //
    //
	setColumnCount(3);
	setColumnWidth(0, 180);
	setColumnWidth(1, 30);
	QStringList headers;
	headers << "" << "" <<"" ;
	setHeaderLabels(QStringList(headers));
	QList<QTreeWidgetItem *> rootList;

	QList<QString> countries = signalManager_->getCountries();
	int categorySize = signalManager_->getCategoriesSize();
	for (int i = 0; i < countries.size(); i++)
	{
		QTreeWidgetItem *countryWidget = new QTreeWidgetItem; 
		countryWidget->setText(0,countries.at(i));
		rootList.append(countryWidget);
		QMap<QString, QTreeWidgetItem *> categoryMap;
		foreach (const SignalContainer *container, signalManager_->getSignals(countries.at(i)))
		{
			const QString &signCategory = container->getsignalCategory();
			//qDebug() << signCategory;
			QTreeWidgetItem *categoryWidget;
			if (categoryMap.contains(signCategory))
			{
				categoryWidget = categoryMap.value(signCategory);
			}
			else 
			{
				categoryWidget = new QTreeWidgetItem(countryWidget);
				categoryWidget->setText(0,signCategory);
				categoryWidget->setText(1,tr(""));
				categoryMap.insert(signCategory,categoryWidget);
				countryWidget->addChild(categoryWidget);  
				int j = 360 / categorySize;
				QColor color;
	            color.setHsv(signalManager_->getCategoryNumber(signCategory) * j, 255, 255, 255);
				categoryWidget->setBackgroundColor(1, color);
			}
			QTreeWidgetItem *signs = new QTreeWidgetItem(categoryWidget); 
			signs->setText(0,container->getSignalName());
			signs->setIcon(0,container->getSignalIcon());	
			categoryWidget->addChild(signs);	
		}
	}
	insertTopLevelItems(0,rootList);
}

  
//################//
// EVENTS         //
//################//

void
SignalTreeWidget::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
	QTreeWidgetItem *item = selectedItems().at(0);
	SignalContainer *signalContainer = signalManager_->getSignalContainer(item->text(0));
	if(signalContainer && projectWidget_)
	{
		const QString &country = signalManager_->getCountry(signalContainer);
		int type = signalContainer->getSignalType();
		const QString &typeSubclass = signalContainer->getSignalTypeSubclass();
		int subtype = signalContainer->getSignalSubType();
		
		foreach (DataElement *element, projectWidget_->getProjectData()->getSelectedElements())
		{
			Signal *signal = dynamic_cast<Signal *>(element);
			if (signal)
			{
				SetSignalPropertiesCommand *command = new SetSignalPropertiesCommand(signal, signal->getId(), signal->getName(), signal->getT(), signal->getDynamic(), signal->getOrientation(), signal->getZOffset(), country, type, typeSubclass, subtype, signal->getValue(), signal->getHeading(), signal->getPitch(), signal->getRoll(), signal->getPole(), signal->getSize(), signal->getValidFromLane(), signal->getValidToLane(), signal->getCrossingProbability(), signal->getResetTime(), NULL);
				projectWidget_->getProjectSettings()->executeCommand(command);
			}
		}
	}
    QTreeWidget::selectionChanged(selected, deselected);
}
