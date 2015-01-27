/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QMenu>

#include "MECategory.h"
#include "widgets/MEGraphicsView.h"

/*!
    \class MECategory
    \brief Class manage the module categories
*/

MECategory::MECategory(const QString &category)
    : m_categoryName(category)
    , m_categoryItem(NULL)
{
    m_modulePopup = new QMenu(0);
    m_categoryAction = new QAction(category, 0);
    m_categoryAction->setMenu(m_modulePopup);
    QVariant var;
    var.setValue(this);
    m_categoryAction->setData(var);
    QObject::connect(m_categoryAction, SIGNAL(hovered()), MEGraphicsView::instance(), SLOT(hoveredCategoryCB()));
    qRegisterMetaType<MECategory *>("MECategory");
}

MECategory::MECategory(const QString &category, const QString &modulename)
    : m_categoryName(category)
    , m_categoryItem(NULL)
{
    moduleList << modulename;

    m_modulePopup = new QMenu(0);
    m_categoryAction = new QAction(category, 0);
    m_categoryAction->setMenu(m_modulePopup);
    QVariant var;
    var.setValue(this);
    m_categoryAction->setData(var);
    QObject::connect(m_categoryAction, SIGNAL(hovered()), MEGraphicsView::instance(), SLOT(hoveredCategoryCB()));
}

MECategory::MECategory() {}
MECategory::~MECategory() {}

void MECategory::addModuleName(const QString &name)
{
    moduleList << name;

    // create an action used in node popup when replacing module
    QAction *moduleAction = new QAction(name, 0);
    QObject::connect(moduleAction, SIGNAL(triggered()), MEGraphicsView::instance(), SLOT(replaceModulesCB()));
    m_modulePopup->addAction(moduleAction);
}
