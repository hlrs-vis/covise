/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QHash>
#include <QMultiMap>

#include <config/coConfigEntry.h>
#include <config/coConfigGroup.h>
#include <config/coConfig.h>
#include <config/coConfigSchemaInfosList.h>

class QAction;
class QHBoxLayout;
class QComboBox;
class QTreeView;
class QStandardItemModel;
class QModelIndex;
class QToolButton;
class QListWidget;
class QStackedWidget;
class QFile;
namespace covise
{
class coConfigEntry;
class coConfigGroup;
class coConfig;
class coConfigSchemaInfosList;
}

class coEditorMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    coEditorMainWindow(const QString &fileName = QString::null);
    ~coEditorMainWindow();
    QHash<QString, covise::coConfigEntry *> loadFile(const QString &fileName);

public slots:
    void setValue(const QString &variable, const QString &value,
                  const QString &section, const QString &targetHost = QString::null);
    void deleteValue(const QString &variable, const QString &section,
                     const QString &targetHost = QString::null);
    // show a message for timeout time in the status bar
    void showStatusBar(const QString &message, int timeout = 0);
    void putInLog(qint64 bytes); // add a message to the ErrorLog
protected:
    //    void closeEvent( QCloseEvent *event );

private slots:
    void newFile();
    void openConfig();
    void addConfig();
    void removeConfig();
    void openSchema();
    void addHost();
    void initEmpty();
    bool save();
    bool saveTo();
    void about();
    void errorLog();
    void changeHost(const QString &activeHost);
    void changeGroup(const QModelIndex &index);
    void informcoEditorGroupWidgets(const QString &hostName);

private:
    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void readSettings();
    void writeSettings();

    bool saveFile(const QString &fileName = QString::null);
    void setCurrentFile(const QString &fileName);
    void loadConfig(const QString &filename);
    void addFile(const QString &fileName);
    void removeFile(const QString &file);
    QString strippedName(const QString &fullFileName) const;
    void createConstruct();
    QHash<QString, covise::coConfigEntry *> getEntriesForGroup(const QString groupNamePath);
    void workGroup(const QString &name, covise::coConfigEntry *entry = 0, covise::coConfigSchemaInfosList *infos = 0);
    void createTreeModel(QStringList elementGroups);
    void clearData();

    QString currentFile;

    QMenu *fileMenu;
    QMenu *editMenu;
    QMenu *helpMenu;
    QToolBar *fileToolBar;
    QToolBar *editToolBar;
    QToolButton *schemaButton;
    QAction *newAct;
    QAction *openConfigAct;
    QAction *addConfigAct;
    QAction *removeConfigAct;
    QAction *openSchemaAct;
    QAction *saveAct;
    QAction *saveToAct;
    QAction *exitAct;
    QAction *aboutAct;
    QAction *errorLogAct;
    QAction *aboutQtAct;
    QAction *changeConfigScopeAct;
    QAction *addHostAct;

    QTreeView *treeView;
    QStackedWidget *stackedCenterWidget;
    QWidget *startScreen;
    QComboBox *hostsComboBox;
    QComboBox *archComboBox;
    QListWidget *ErrorLogWidget;
    QHBoxLayout *layout;
    covise::coConfigGroup *oneConfigGroup;
    covise::coConfig *konfig;
    QMultiMap<QString, covise::coConfigEntry *> mainWinHostConfigs;
    QStringList files;

    QIODevice *orgCerrDevice;
    QIODevice *orgCoutDevice;
    QFile *myErrorFile;
};
#endif
