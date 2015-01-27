/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MODULETREE_H
#define ME_MODULETREE_H

#include <QTreeWidget>
#include <QMap>

class QMenu;
class QAction;
class QTreeWidgetItem;
class QString;
class MEMainHandler;

class MEHost;

//=========================================================================
class MEModuleTree : public QTreeWidget
//=========================================================================
{
    Q_OBJECT

public:
    MEModuleTree(QWidget *parent = 0);
    ~MEModuleTree();

    void addHostList(MEHost *host);
    void changeBrowserItems();
    void showMatchingItems(const QString &searchString);

    QTreeWidgetItem *findCategory(QTreeWidgetItem *host, const QString &name);
    QTreeWidgetItem *findModule(QTreeWidgetItem *category, const QString &name);

signals:

    void showUsedNodes(const QString &category, const QString &modulename);
    void showUsedCategory(const QString &category);

private:
    static MEMainHandler *m_mainHandler;

    QMenu *m_hostMenu, *m_categoryMenu, *m_moduleMenu;
    QAction *m_delhost_a, *m_addUI_a, *m_deleteUI_a, *m_help_a, *m_separator_a, *m_exec_debug_a, *m_exec_memcheck_a;
    QTreeWidgetItem *m_clickedItem, *m_clickedCategory;
    QString m_currentModuleName;
    QStringList m_usedModules;
    static QMap<QString, QString> s_moduleHelp;

    int getDepth(const QTreeWidgetItem *item) const;
    void highlightHistoryModules();
    bool getHostUserCategoryName(const QTreeWidgetItem *item,
                                 QString *host, QString *user, QString *category, QString *name) const;
    void hideUnusedItems(QTreeWidgetItem *category);
    void readModuleTooltips();

public slots:

    void moduleUseNotification(const QString &modname);
    void restoreList();
    void executeVisibleModule();
    void developerMode(bool);

private slots:

    void doubleClicked(QTreeWidgetItem *, int);
    void clicked(QTreeWidgetItem *, int);
    void collapsed(QTreeWidgetItem *);
    void expanded(QTreeWidgetItem *);

    void removeHostCB();
    void infoCB();
    void execDebugCB();
    void execMemcheckCB();

protected:
    void startDrag(Qt::DropActions supportedActions);
    void contextMenuEvent(QContextMenuEvent *e);
    void dragEnterEvent(QDragEnterEvent *e);
    void dragMoveEvent(QDragMoveEvent *e);
    void dragLeaveEvent(QDragLeaveEvent *e);

    bool dropMimeData(QTreeWidgetItem *parent, int index, const QMimeData *data, Qt::DropAction action);
    QMimeData *mimeData(const QList<QTreeWidgetItem *>) const;
};

class OperationWaiter : public QObject
{
    Q_OBJECT

public:
    OperationWaiter();
    bool wait(int limit = -1);
public slots:
    void finished();

private:
    bool m_done;
};

#endif
