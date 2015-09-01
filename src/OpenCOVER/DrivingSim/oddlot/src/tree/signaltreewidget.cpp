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

// gui //
//
#include "src/gui/projectwidget.hpp"

// MainWindow//
//
#include "src/mainwindow.hpp"


#include <QWidget>

//################//
// CONSTRUCTOR    //
//################//

SignalTreeWidget::SignalTreeWidget(ProjectWidget *projectWidget, ProjectData *projectData)
	: QTreeWidget(projectWidget)
	, projectWidget_(projectWidget)
    , projectData_(projectData)
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
	/*QTreeWidget *signalTree = new QTreeWidget();
	signalTree->setIconSize(QSize(40,40));
	signalsDock_->setWidget(signalTree);
	signalTree->setHeaderLabel(""); */
	
	QList<QTreeWidgetItem *> rootList;

	SignalManager *signalManager = projectWidget_->getMainWindow()->getSignalManager();

	QList<QString> countries = signalManager->getCountries();

	for (int i = 0; i < countries.size(); i++)
	{
		QTreeWidgetItem *countryWidget = new QTreeWidgetItem;   
		countryWidget->setText(0,countries.at(i));
		rootList.append(countryWidget);
		QMap<QString, QTreeWidgetItem *> categoryMap;
		foreach (const SignalContainer *container, signalManager->getSignals(countries.at(i)))
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
				categoryWidget = new QTreeWidgetItem(countryWidget,QStringList(signCategory));
				categoryMap.insert(signCategory,categoryWidget);
				countryWidget->addChild(categoryWidget);   
			}
			QTreeWidgetItem *signs = new QTreeWidgetItem(categoryWidget,QStringList(container->getSignalName())); 
			signs->setIcon(0,container->getSignalIcon());
			signs->setSizeHint(0,QSize(45,45));
			categoryWidget->addChild(signs);	
		}
	}
	//signalTree->insertTopLevelItems(0,rootList);
}

  
//################//
// EVENTS         //
//################//

void
SignalTreeWidget::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
    //	qDebug() << "selected: " << selected.indexes().count();
    //	qDebug() << "deselected: " << deselected.indexes().count();

    // Set the data of the item, so the item notices it's selection
    //
   /* foreach (QModelIndex index, deselected.indexes())
    {
        model()->setData(index, false, Qt::UserRole + ProjectTree::PTR_Selection);
    }

    foreach (QModelIndex index, selected.indexes())
    {
        model()->setData(index, true, Qt::UserRole + ProjectTree::PTR_Selection);
    }*/

    QTreeWidget::selectionChanged(selected, deselected);
}
