/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_CATEGORY_H
#define ME_CATEGORY_H

#include <QStringList>

class QAction;
class QString;
class QTreeWidgetItem;

//================================================
class MECategory
//================================================
{

public:
    MECategory();
    MECategory(const QString &category);
    MECategory(const QString &category, const QString &modulename);
    ~MECategory();

    QStringList moduleList;

    void addModuleName(const QString &category);
    void setCategoryItem(QTreeWidgetItem *item)
    {
        m_categoryItem = item;
    };

    QTreeWidgetItem *getCategoryItem()
    {
        return m_categoryItem;
    };
    QString getName()
    {
        return m_categoryName;
    };
    QAction *getAction()
    {
        return m_categoryAction;
    };
    QMenu *getMenu()
    {
        return m_modulePopup;
    };

private:
    QString m_categoryName; // name of category
    QTreeWidgetItem *m_categoryItem; // pointer to item in QTreeWidget
    QAction *m_categoryAction;
    QMenu *m_modulePopup;
};

Q_DECLARE_METATYPE(MECategory *);
#endif
