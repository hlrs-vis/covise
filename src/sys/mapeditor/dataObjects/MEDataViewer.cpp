/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QScrollArea>
#include <QSplitter>
#include <QLabel>
#include <QVBoxLayout>
#include <QStackedWidget>

#include "MEDataViewer.h"
#include "MEDataArray.h"
#include "MEDataTree.h"
#include "MEDataObject.h"
#include "handler/MEMainHandler.h"

/*!
    \class MEDataViewer
    \brief Main container widget for the examination of data objects and structures

    The widget is mainly based on three subwindows. <br>
    MEDataTree    contains a tree widget of existing data objects <br>
    MEDataObject  shows general informations like type, host, attributes and data structure <br>
    MEDataArray   shows the content (array) of a data object
*/

MEDataViewer::MEDataViewer(QWidget *parent)
    : QWidget(parent)
    , m_maxArray(3)
{

    // set the main layout
    QVBoxLayout *main = new QVBoxLayout(this);

    // generate main horizontal & vertical splitter window
    m_splitter = new QSplitter(this);
    QList<int> list;
    list << 300 << 600;
    m_splitter->setSizes(list);
    m_left = new QSplitter(Qt::Vertical);
    m_right = new QSplitter(Qt::Vertical);

    m_scrolling = new QScrollArea(this);
    m_scrolling->setWidgetResizable(true);
    m_scrolling->setWidget(m_right);

    m_splitter->addWidget(m_left);
    m_splitter->addWidget(m_scrolling);
    m_splitter->setStretchFactor(0, 0);
    m_splitter->setStretchFactor(1, 1);

    // populate m_left splitter with the data object browser and the object info view with scroll bars
    // data browser
    m_left->addWidget(MEDataTree::instance());

    // data info
    m_widget = new QWidget();
    m_widget->hide();
    QVBoxLayout *vb = new QVBoxLayout(m_widget);

    m_infoView = new QStackedWidget();
    QScrollArea *area = new QScrollArea();
    area->setWidgetResizable(true);
    area->setWidget(m_infoView);

    QLabel *caption = new QLabel("Data Object Info", m_widget);

    vb->addWidget(caption);
    vb->addWidget(area);

    m_left->addWidget(m_widget);

    // don't show m_right splitter containing data arrays at beginning
    m_scrolling->hide();

    main->addWidget(m_splitter);
}

MEDataViewer *MEDataViewer::instance()
{
    static MEDataViewer *singleton = 0;
    if (singleton == 0)
        singleton = new MEDataViewer();

    return singleton;
}

//!
//! reset pointer after "New" (clear canvas), hide the data array widget
//!
void MEDataViewer::reset()
{
    m_scrolling->hide();
}

#ifdef YAC

//!
//! create a new data information object
//!
void MEDataViewer::createObject(MEDataTreeItem *item, covise::coRecvBuffer &recb)
{
    MEDataObject *dobj = new MEDataObject(recb, item, m_infoView);
    m_infoView->addWidget(dobj);
    m_infoView->setCurrentWidget(dobj);
    item->setDataObject(dobj);
}

//!
//! create a new data array info
//!
void MEDataViewer::createArray(MEDataTreeItem *item, covise::coRecvBuffer &recb)
{
    MEDataArray *array = new MEDataArray(recb, item);
    showArray(array);
    item->setDataArray(array);
}

#else

//!
//! create a new data information object
//!
MEDataObject *MEDataViewer::createObject(MEDataTreeItem *item)
{
    MEDataObject *dobj = new MEDataObject(item, m_infoView);
    m_infoView->addWidget(dobj);
    m_infoView->setCurrentWidget(dobj);
    return dobj;
}

//!
//! create a new data array info
//!
MEDataArray *MEDataViewer::createArray(MEDataTreeItem *item)
{
    MEDataArray *array = new MEDataArray(item);
    showArray(array);
    return array;
}
#endif

//!
//! show the selected data object content (which already exists), user has clicked a data object in the tree
//!
void MEDataViewer::showObject(MEDataObject *object)
{

    if (m_widget->isHidden())
        m_widget->show();
    m_infoView->setCurrentWidget(object);
}

//!
//! show the data array in one of the three possible positions on the right splitter
//!
void MEDataViewer::showArray(MEDataArray *array)
{

    if (m_scrolling->isHidden())
        m_scrolling->show();
    array->show();

    // is array already shown ?
    // move it to first position

    int count = m_right->count();
    for (int i = 0; i < count; i++)
    {
        if (m_right->widget(i) == array)
        {
            m_right->insertWidget(0, array);
            return;
        }
    }

    // all possible cells crowded ?
    // remove oldest (last) array

    if (count == m_maxArray)
    {
        MEDataArray *oldest = qobject_cast<MEDataArray *>(m_right->widget(m_maxArray - 1));
        oldest->setParent(0);
        oldest->hide();
    }

    // insert current array at the beginning

    m_right->insertWidget(0, array);
}

//!
//! no. of visible data array in data viewer
//!
void MEDataViewer::spinCB(int num)
{
    m_maxArray = num;
}
