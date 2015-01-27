/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <config/coConfigRoot.h>
#include <config/coConfigGroup.h>

#include "coEditorMainWindow.h"
#include "coEditorGroupWidget.h"
#include "coEditorEntryWidget.h"
#include <QtGui>
#include <QLayout>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QString>
#include <QList>
#include <QComboBox>
#include <QToolButton>
#include <QStackedWidget>
#include <QListWidget>
#include <QTreeView>
#include <QMenu>
#include <QMenuBar>
#include <QStatusBar>
#include <QToolBar>
#include <QMessageBox>
#include <QHeaderView>
#include <QLabel>
#include <QFileDialog>
#include <QInputDialog>

#include <config/coConfigLog.h> // Log umleiten
#include <config/coConfig.h>
#include <config/coConfigEntry.h>
#include <config/coConfigEntryToEditor.h>

//the observer
// #include "config/coConfigEditorEntry.h"
//

#include <config/kernel/coConfigSchema.h>
#include <config/coConfigSchemaInfos.h>

#include <iostream>
#include <fstream>

using namespace covise;

coEditorMainWindow::coEditorMainWindow(const QString &fileName)
{
    setObjectName(QString::fromUtf8("coEditorMainWindow"));
    setMinimumSize(640, 480);
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    createConstruct();
    createActions();
    createMenus();
    createToolBars();
    createStatusBar();

    readSettings();

    if (!fileName.isEmpty())
    {
        loadConfig(fileName);
    }
}

coEditorMainWindow::~coEditorMainWindow()
{
}

//NOTE useless atm show ErrorLogWidget
void coEditorMainWindow::errorLog()
{
    //ErrorLogWidget->insertItem(ErrorLogWidget->count(), line);  //QLISTWidget
    stackedCenterWidget->setCurrentWidget(ErrorLogWidget);
}

//NOTE useless atm
// add an error to the Errorlog
void coEditorMainWindow::putInLog(qint64 bytes)
{
    if (ErrorLogWidget != 0)
    {
        //QByteArray data = file->readLine();
        //       myCerrBuffer.
        //QByteArray data = coConfigLog::cerr.device()/*myCerrBuffer*/->readLine(bytes);
        /*data();*/
        QString line;
        // line = QString::fromUtf8(data.constData());
        line = myErrorFile->readLine(bytes);
        //QLISTWidget
        ErrorLogWidget->insertItem(ErrorLogWidget->count(), line);
    }
}

void coEditorMainWindow::createConstruct()
{

    // build up widgets
    QWidget *middle = new QWidget(this);
    middle->setObjectName(QString::fromUtf8("middleWidget"));
    stackedCenterWidget = new QStackedWidget;
    stackedCenterWidget->setObjectName(QString::fromUtf8("stackedCenterWidget"));

    hostsComboBox = new QComboBox(this);
    hostsComboBox->setMinimumSize(140, 10);
    hostsComboBox->setDuplicatesEnabled(0);
    hostsComboBox->setStatusTip(tr("The active host can be selected here."));
    hostsComboBox->setToolTip(tr("The active host can be selected here."));
    // deactivated
    archComboBox = new QComboBox(this);
    archComboBox->setMinimumSize(90, 10);
    archComboBox->setDuplicatesEnabled(0);
    archComboBox->setEditable(0);
    archComboBox->addItem("windows");
    archComboBox->addItem("unix");
    archComboBox->addItem("mac");
    archComboBox->addItem("x11");
    archComboBox->setDisabled(1);

    /*QToolButton**/ schemaButton = new QToolButton();
    //    schemaButton->setText ("Show All");
    schemaButton->setIcon(QIcon(":/images/ktip.png"));
    schemaButton->setCheckable(1);
    schemaButton->setAutoRepeat(1);
    schemaButton->setStatusTip(tr("Show all possible items or only declared items."));
    schemaButton->setToolTip(tr("Toggle view between all possible items or only declared items."));

    treeView = new QTreeView(this);
    treeView->setMaximumWidth(220);
    treeView->setMinimumWidth(180);
    treeView->setSortingEnabled(1);
    treeView->sortByColumn(0, Qt::AscendingOrder);
    treeView->header()->hide();
    treeView->setAutoScroll(1);
    treeView->setModel(new QStandardItemModel);
    treeView->setProperty("infoWidget", true);

    startScreen = new QWidget(this);
    startScreen->setObjectName("startScreen");
    QHBoxLayout *startLayout = new QHBoxLayout();
    startScreen->setLayout(startLayout);
    QLabel *startLineEdit = new QLabel(startScreen);
    startLineEdit->setText(tr("Covise Config Editor \n"
                              "Load an existing Config or start with a blank paper.\n\n"
                              "On the left side, all defined groups are presented in a tree.\n"
                              "Click one item to see its content.\n"
                              "Above a host can be selected.\n\n"
                              "Toggle view between only declared items and all possible items by clicking \n"
                              "the blue info icon in the toolbar on top.\n\n"
                              "This Editor uses colors for easy understanding. Here`s a little description: \n"
                              " a red field means text in this field is not valid to the rules provided by the schema.\n"
                              " red fields wont be saved.\n"
                              " a lightblue field means that this field is optional.\n"
                              " a greyd field tells you, this item cannot be accessed.\n\n\n"
                              " the statusBar at the bottom of this window will show information for the rules\n"
                              " of the item the mouse is over.\n"));

    startLayout->setMargin(10);
    startLayout->addWidget(startLineEdit);
    stackedCenterWidget->addWidget(startScreen);
    stackedCenterWidget->setCurrentWidget(startScreen);

    layout = new QHBoxLayout;
    layout->setObjectName(QString::fromUtf8("coEditorMainWindowLayout"));
    layout->addWidget(treeView);
    layout->addWidget(stackedCenterWidget);

    middle->setLayout(layout);
    setCentralWidget(middle);

    // create ErrorLogWidget:
    ErrorLogWidget = new QListWidget();
    ErrorLogWidget->setObjectName(QString::fromUtf8("ErrorLogWidget"));
    QSize size(622, 509);
    size = size.expandedTo(ErrorLogWidget->minimumSizeHint());
    ErrorLogWidget->resize(size);
    stackedCenterWidget->addWidget(ErrorLogWidget);

    //Systemwide layout:
    //setStyleSheet(coEditorEntryWidget[infoWidget="true"] { background-color: lightblue });

    konfig = coConfig::getInstance();
    konfig->setDebugLevel(coConfig::DebugAll);
    oneConfigGroup = new coConfigGroup(QString::fromUtf8("gruppe"));
    konfig->addConfig(oneConfigGroup);
}

// create a blueprint from schema to create a new configFile
void coEditorMainWindow::initEmpty()
{
    coConfigSchema *schema = coConfigSchema::getInstance();
    if (!schema)
    {
        coConfigSchema::loadSchema(); //fileName can be given here
        schema = coConfigSchema::getInstance();
        statusBar()->showMessage(tr("Loaded schema"), 3000);
    }
    // now we have a schema,
    createTreeModel(schema->getGroupsFromSchema());
    schemaButton->click();
}

void coEditorMainWindow::newFile()
{
    clearData();
    hostsComboBox->clear();
    addHost();
    if (hostsComboBox->count() == 0)
        return;
    //get a Schema
    initEmpty();
    //TODO change unkown
    oneConfigGroup->addConfig("unkown", "unkown", 1);
    setCurrentFile("unkown");
}

void coEditorMainWindow::openConfig()
{
    //clean up if files are loaded
    if (!files.isEmpty())
    {
        clearData();
        hostsComboBox->clear();
        while (!files.isEmpty())
        {
            removeFile(files.takeFirst());
        }
    }
    //load a new file
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Covise Config file"), QString(getenv("COVISEDIR")).append("/src/kernel/config"), tr("XML Files (*.xml);; Any (*.*)"));
    loadConfig(fileName);
}

// loads a config file.
void coEditorMainWindow::loadConfig(const QString &filename)
{
    if (!filename.isEmpty())
    {
        QString startingHost = "";
        clearData();
        coConfigSchema::getInstance()->loadSchema(); // initialize schema TODO filename?
        // get root entries
        QHash<QString, coConfigEntry *> hosts = loadFile(filename);
        QList<QString> hostList = hosts.keys();
        if (!hostList.isEmpty())
        {
            for (QList<QString>::const_iterator iter = hostList.begin(); iter != hostList.end(); ++iter)
            {
                mainWinHostConfigs.insert((*iter), hosts.value(*iter));
                hostsComboBox->addItem((*iter));
            }
            startingHost = hostList.first();
            //TODO set starting Host, get it from commandline arg
            changeHost(startingHost);
            hostsComboBox->setCurrentIndex(hostsComboBox->findText(startingHost));

            setCurrentFile(filename);
            files.append(filename);
            createTreeModel(coConfigSchema::getInstance()->getGroupsFromSchema());
        }
        else
        {
            QMessageBox::warning(this, tr("Covise Config Editor"),
                                 tr("Config File loading failed.\n"
                                    "Was that really a Covise Config File?\n"));
        }

        //          statusBar() ->showMessage(tr("File loaded"), 2000);
    }
}

void coEditorMainWindow::addConfig()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Covise Config file"), QString(getenv("COVISEDIR")).append("/src/kernel/config"),
                                                    tr("XML Files (*.xml);; Any (*.*)"));
    if (!fileName.isEmpty())
    {
        addFile(fileName);
        QString startingHost = mainWinHostConfigs.keys().first();
        changeHost(startingHost);
        hostsComboBox->setCurrentIndex(hostsComboBox->findText(startingHost));

        setCurrentFile(fileName);
        files.append(fileName);
        statusBar()->showMessage(tr("File added"), 2000);
    }
}

void coEditorMainWindow::removeConfig()
{
    bool ok;
    QString fileName = QInputDialog::getItem(this, tr("Remove a Config file"),
                                             tr("Choose which Config files to remove:"), files, 0, false, &ok);
    if (ok && !fileName.isEmpty())
        removeFile(fileName);
}

// load a new schema file, throw away the old one
void coEditorMainWindow::openSchema()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Schema file"), "", tr("Schema files (*.xsd);; Any (*.*)"));
    if (!fileName.isEmpty())
    {
        coConfigSchema::loadSchema(fileName);
        coConfigSchema *schema = coConfigSchema::getInstance();
        if (schema)
        {
            createTreeModel(schema->getGroupsFromSchema());
            statusBar()->showMessage(tr("Loaded schema"), 3000);
            schemaButton->click();
            //TODO change unkown
            oneConfigGroup->addConfig("unkown", "unkown", 1);
            setCurrentFile("unkown");
        }
        else
            QMessageBox::warning(this, tr("Covise Config Editor"), tr("Loading Schema file failed.\n"));
    }
}

bool coEditorMainWindow::save()
{
    // check for filename. "unkown" is set when an empty file is initiated
    if (currentFile.isEmpty() || currentFile.compare("unkown") == 0)
    {
        return saveTo();
    }
    else
    {
        return saveFile(currentFile);
    }
}

bool coEditorMainWindow::saveTo()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Covise Config file"), "", tr("XML files (*.xml )"));
    ;
    if (fileName.isEmpty()) // NOTE check for file existens
        return false;
    else // check wether the filename has an xml ending.
    {
        if (fileName.section(".", -1).compare("xml", Qt::CaseInsensitive) == 0)
        {
            return saveFile(fileName);
        }
        else
            return saveFile(fileName.append(".xml"));
    }
}

void coEditorMainWindow::addHost()
{
    bool ok;
    QString host = QInputDialog::getText(this, tr("Add a new Host"),
                                         tr("Type name of host to be added: "), QLineEdit::Normal,
                                         0, &ok);
    if (ok && !host.isEmpty())
        hostsComboBox->addItem(host);
}

void coEditorMainWindow::createActions()
{
    newAct = new QAction(QIcon(":/images/new.png"), tr("&New"), this);
    newAct->setShortcut(tr("Ctrl+N"));
    newAct->setStatusTip(tr("Create a new file"));
    connect(newAct, SIGNAL(triggered()), this, SLOT(newFile()));

    openConfigAct = new QAction(QIcon(":/images/open.png"), tr("&Open Config file..."), this);
    openConfigAct->setShortcut(tr("Ctrl+O"));
    openConfigAct->setStatusTip(tr("Open an existing Config file"));
    connect(openConfigAct, SIGNAL(triggered()), this, SLOT(openConfig()));

    addConfigAct = new QAction(QIcon(":/images/add.png"), tr("&Add Config file..."), this);
    addConfigAct->setShortcut(tr("Ctrl+A"));
    addConfigAct->setStatusTip(tr("Add an existing Config file"));
    connect(addConfigAct, SIGNAL(triggered()), this, SLOT(addConfig()));

    removeConfigAct = new QAction(QIcon(":/images/remove.png"), tr("Remove Config file..."), this);
    // removeConfigAct->setShortcut(tr("Ctrl+A"));
    removeConfigAct->setStatusTip(tr("Remove a loaded Config file"));
    connect(removeConfigAct, SIGNAL(triggered()), this, SLOT(removeConfig()));

    openSchemaAct = new QAction(QIcon(":/images/openSchema.png"), tr("&Use another Schema file ..."), this);
    openSchemaAct->setShortcut(tr("Ctrl+U"));
    openSchemaAct->setStatusTip(tr("Use another Schema file"));
    connect(openSchemaAct, SIGNAL(triggered()), this, SLOT(openSchema()));

    saveAct = new QAction(QIcon(":/images/save.png"), tr("&Save Config file"), this);
    saveAct->setShortcut(tr("Ctrl+S"));
    saveAct->setStatusTip(tr("Save the Config to disk"));
    connect(saveAct, SIGNAL(triggered()), this, SLOT(save()));

    saveToAct = new QAction(QIcon(":/images/saveas.png"), tr("S&ave Config to file ..."), this);
    saveToAct->setStatusTip(tr("Save the Config under a new name"));
    connect(saveToAct, SIGNAL(triggered()), this, SLOT(saveTo()));

    exitAct = new QAction(tr("E&xit"), this);
    exitAct->setShortcut(tr("Ctrl+Q"));
    exitAct->setStatusTip(tr("Exit the application"));
    connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

    aboutAct = new QAction(tr("&About"), this);
    aboutAct->setStatusTip(tr("Show the application's About box"));
    connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

    errorLogAct = new QAction(tr("&Error Log"), this);
    errorLogAct->setStatusTip(tr("Show Errors"));
    connect(errorLogAct, SIGNAL(triggered()), this, SLOT(errorLog()));

    aboutQtAct = new QAction(tr("About &Qt"), this);
    aboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
    connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
    /*
      changeConfigScopeAct = new QAction(QIcon(":/images/copy.png"), tr("Change Config scope"), this);
      changeConfigScopeAct->setStatusTip(tr("Change the scope on wich you are working"));
      connect(changeConfigScopeAct , SIGNAL(triggered()), this, SLOT(initEmpty()));
   */

    connect(hostsComboBox, SIGNAL(activated(const QString &)),
            this, SLOT(changeHost(const QString &)));

    connect(archComboBox, SIGNAL(activated(const QString &)),
            this, SLOT(changeArch(const QString &)));

    addHostAct = new QAction(QIcon(":/images/add.png"), tr("Add a new Host"), this);
    addHostAct->setStatusTip(tr("Add a new Host"));
    connect(addHostAct, SIGNAL(triggered()), this, SLOT(addHost()));

    connect(treeView, SIGNAL(clicked(QModelIndex)),
            this, SLOT(changeGroup(QModelIndex)));
}

void coEditorMainWindow::createMenus()
{
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(openConfigAct);
    fileMenu->addAction(newAct);
    fileMenu->addAction(openSchemaAct);
    fileMenu->addAction(addConfigAct);
    fileMenu->addAction(removeConfigAct);
    fileMenu->addAction(saveAct);
    fileMenu->addAction(saveToAct);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAct);

    editMenu = menuBar()->addMenu(tr("&Edit"));
    //editMenu->addAction(changeConfigScopeAct);
    editMenu->addAction(addHostAct);
    menuBar()->addSeparator();

    helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction(errorLogAct);
    helpMenu->addAction(aboutAct);
    helpMenu->addAction(aboutQtAct);
}

void coEditorMainWindow::createToolBars()
{
    fileToolBar = addToolBar(tr("File"));
    fileToolBar->addAction(openConfigAct);
    fileToolBar->addAction(newAct);
    fileToolBar->addAction(openSchemaAct);
    fileToolBar->addAction(addConfigAct);
    fileToolBar->addAction(removeConfigAct);
    fileToolBar->addAction(saveAct);
    fileToolBar->addAction(saveToAct);
    //fileToolBar->addAction(changeConfigScopeAct);
    fileToolBar->addSeparator();
    fileToolBar->addWidget(new QLabel(" host: "));
    fileToolBar->addWidget(hostsComboBox);
    fileToolBar->addAction(addHostAct);
    //    fileToolBar->addSeparator();
    //    fileToolBar->addWidget (archComboBox);
    fileToolBar->addSeparator();
    fileToolBar->addWidget(schemaButton);
}

void coEditorMainWindow::createStatusBar()
{
    statusBar()->showMessage(tr("Ready"));
}

// shows a message in the status bar for a amount of time
void coEditorMainWindow::showStatusBar(const QString &message, int timeout)
{
    statusBar()->showMessage(message, timeout);
}

// add another configFile. newer values overwrite older ones (just like coConfig does)
void coEditorMainWindow::addFile(const QString &fileName)
{
    if (!fileName.isEmpty())
    {
        //clearData();
        QHash<QString, coConfigEntry *> hosts = loadFile(fileName);
        QList<QString> hostList = hosts.keys();
        if (!hostList.isEmpty())
        {
            for (QList<QString>::const_iterator iter = hostList.begin(); iter != hostList.end(); ++iter)
            {
                //check if host is already in declared,
                int index = hostsComboBox->findText((*iter));
                if (index >= 0)
                {
                    informcoEditorGroupWidgets((*iter));
                }
                else
                    hostsComboBox->addItem((*iter));

                mainWinHostConfigs.insert((*iter), hosts.value(*iter));
            }
        }
    }
}

// tells all coEditorGroupWidgets for this host, that they are out of Date. (happens after config has beed added)
void coEditorMainWindow::informcoEditorGroupWidgets(const QString &hostName)
{
    QString lookup = hostName; // gruppenobjectName = Host:GroupName
    QRegExp rx(hostName + ".*");
    // find all coEditorGroupWidgets for this host
    QList<coEditorGroupWidget *> searchedGroups = stackedCenterWidget->findChildren<coEditorGroupWidget *>(rx);
    if (!searchedGroups.isEmpty())
    {
        //tell them that they are out of Date, though the can be updated, when necessary
        for (QList<coEditorGroupWidget *>::const_iterator iter = searchedGroups.begin(); iter != searchedGroups.end(); ++iter)
        {
            (*iter)->setOutOfDate();
        }
    }
}

// fetches rootentries. uses coConfigGroup and coConfigRoot (is friend)
QHash<QString, coConfigEntry *> coEditorMainWindow::loadFile(const QString &fileName)
{
    QHash<QString, coConfigEntry *> hostConfigs;
    hostConfigs.clear();
    if (QFileInfo(fileName).isFile())
    {
        // add the config to Editor's coConfigGroup. second arg is actually name, but here the same
        oneConfigGroup->addConfig(fileName, fileName);
        // get the coConfigRoot for the just added file
        coConfigRoot *root = oneConfigGroup->addConfig(fileName, fileName);
        // get all HostNames for this coConfigRoot,i.e duplicate Roots hostConfigs
        QStringList hosts = root->getHosts();
        for (QList<QString>::const_iterator hostname = hosts.begin(); hostname != hosts.end(); ++hostname)
        {
            // insert all HostName, coConfigEntry* into hostConfigs Hash
            hostConfigs.insert(*hostname, root->getConfigForHost(*hostname));
        }
        // special handling of global section
        if (root->getGlobalConfig() && root->getGlobalConfig()->hasChildren())
        {
            hostConfigs.insert("GLOBAL", root->getGlobalConfig());
        }
    }
    else
    {
        QMessageBox::warning(this, tr("Covise Config Editor"),
                             tr("Config File has not been found.\n"));
        return hostConfigs; //empty
    }
    //remove NullEntry from Hash
    QHash<QString, coConfigEntry *>::iterator iter = hostConfigs.begin();
    while (iter != hostConfigs.end())
    {
        if (!iter.value())
        {
            iter = hostConfigs.erase(iter);
        }
        else
        {
            ++iter;
        }
    }
    return hostConfigs;
}

// removes a configFile that is currently loaded
void coEditorMainWindow::removeFile(const QString &file)
{
    //delete all coEditorGroupWidgets, clear mainWinHostConfigs
    clearData();
    hostsComboBox->clear();
    //remove file
    oneConfigGroup->removeConfig(file);
    int i = files.indexOf(file);
    files.removeAt(i);
    if (files.isEmpty())
        return;
    //  mainWinHostConfigs.take (lastAddedHosts)
    for (QStringList::const_iterator iter = files.begin(); iter != files.end(); iter++)
    {
        QString file = (*iter);
        addFile(*iter);
    }
    setCurrentFile(files.last());
    statusBar()->showMessage(tr("File removed"), 2000);
}

bool coEditorMainWindow::saveFile(const QString &fileName)
{
    QFile file(fileName); // create file
    if (!file.open(QFile::WriteOnly | QFile::Text))
    {
        QMessageBox::warning(this, tr("Application"),
                             tr("Cannot write file %1:\n%2.")
                                 .arg(fileName)
                                 .arg(file.errorString()));
        return false;
    }

    bool saved = 0;
    if (currentFile.compare(fileName) != 0) // if filename has changed
    {
        //setCurrentFile(fileName);
        saved = oneConfigGroup->save(fileName);
    }
    else // filename is still the same
    {
        saved = oneConfigGroup->save();
    }

    QString msg = QString("Saving file successful? : ") + (saved ? "Yes" : "No");
    statusBar()->showMessage(msg, 2000);
    return true;
}

// set window title to fileName
void coEditorMainWindow::setCurrentFile(const QString &fileName)
{
    currentFile = fileName;
    setWindowModified(false);

    QString shownName;
    if (currentFile.isEmpty())
        shownName = "untitled.txt";
    else
        shownName = strippedName(currentFile);

    setWindowTitle(tr("%1[*] - %2").arg(shownName).arg(tr("Covise Config Editor")));
}

QString coEditorMainWindow::strippedName(const QString &fullFileName) const
{
    return QFileInfo(fullFileName).fileName();
}

///changes centerWindow to a new Group. (activated by treeView, item clicked)
void coEditorMainWindow::changeGroup(const QModelIndex &index)
{
    QStandardItem *item = static_cast<QStandardItemModel *>(treeView->model())->itemFromIndex(index);
    QString lookup = item->data().toString(); //name of active group
    workGroup(lookup, 0, coConfigSchema::getInstance()->getSchemaInfosForGroup(item->data().toString()));
}

void coEditorMainWindow::changeHost(const QString &activeHost)
{
    if (activeHost.isEmpty())
    {
        // cerr << "empty host"
        return;
    }

    stackedCenterWidget->setCurrentWidget(startScreen);
    treeView->clearSelection();
}

/// looks for group with "name" for active Host. shows this group in centerWindow.
/// creates the group if necessary. Refreshes group if it is out of Date.
void coEditorMainWindow::workGroup(const QString &name, coConfigEntry *entry, coConfigSchemaInfosList *infos)
{
    // check if the host is given, otherwise ask for it.
    QString lookup = name; // gruppenobjectName = Host:GroupName
    lookup.prepend(":");
    if (hostsComboBox->currentText().isEmpty())
    {
        addHost();
        return;
    }
    else
    {
        lookup.prepend(hostsComboBox->currentText());
    }
    // check if this group already exists.
    coEditorGroupWidget *searchedGroup = stackedCenterWidget->findChild<coEditorGroupWidget *>(lookup);
    if (searchedGroup == 0)
    {
        coEditorGroupWidget *aGroup = new coEditorGroupWidget(this, lookup, entry, infos);
        connect(aGroup, SIGNAL(saveValue(const QString &, const QString &, const QString &, const QString &)),
                this, SLOT(setValue(const QString &, const QString &, const QString &, const QString &)));
        connect(aGroup, SIGNAL(deleteValue(const QString &, const QString &, const QString &)),
                this, SLOT(deleteValue(const QString &, const QString &, const QString &)));
        connect(aGroup, SIGNAL(showStatusBar(const QString &, int)),
                this, SLOT(showStatusBar(const QString &, int)));
        connect(schemaButton, SIGNAL(toggled(bool)), aGroup, SLOT(showInfoWidget(bool)));
        aGroup->addEntries(getEntriesForGroup(name), 0);
        // to show the InfoWidgets
        if (schemaButton->isChecked())
            aGroup->showInfoWidget(1);
        stackedCenterWidget->addWidget(aGroup);
        stackedCenterWidget->setCurrentWidget(aGroup);
    }
    else
    { // check if that group is up to date.
        if (searchedGroup->outOfDate())
        {
            // Add all entries to the existing Group. (old entries will be overwriten).
            searchedGroup->addEntries(getEntriesForGroup(name), 1);
        }
        // show the group in the center area.
        stackedCenterWidget->setCurrentWidget(searchedGroup);
    }
}

// iterate over all coConfigEntries of the current host and return a Hash that contains all entries that belong to given group.
QHash<QString, coConfigEntry *> coEditorMainWindow::getEntriesForGroup(const QString groupNamePath)
{

    QHash<QString, coConfigEntry *> groupList;
    groupList.clear();
    if (!groupNamePath.isEmpty())
    {
        //get a List of all rootEntries for this host. From newest to oldest
        QList<coConfigEntry *> rootEntries = mainWinHostConfigs.values(hostsComboBox->currentText());
        if (!rootEntries.isEmpty())
        {
            for (QList<coConfigEntry *>::const_iterator roots = rootEntries.begin(); roots != rootEntries.end(); ++roots)
            {
                // get all of their children , then add first occurances
                QList<coConfigEntry *> subEntries = coConfigEntryToEditor::getSubEntries(*roots);
                for (QList<coConfigEntry *>::const_iterator iter = subEntries.begin(); iter != subEntries.end(); ++iter)
                {
                    if ((*iter) != 0 /*&& (*iter)->getSchemaInfos() != 0*/) // check if coConfigEntry not is Null and has a coConfigSchemaInfos
                    {
                        // add schemaInfos to Entry NOTE new here was before in entry
                        // check if entry already has a schemaInfos, otherwise fetch it
                        coConfigSchemaInfos *entriesInfo = (*iter)->getSchemaInfos();
                        if (!entriesInfo) // try getting schemaInfos for this entry
                        {
                            entriesInfo = coConfigSchema::getInstance()->getSchemaInfosForElement((*iter)->getPath());
                            (*iter)->setSchemaInfos(entriesInfo);
                        }
                        // check if this coConfigEntry belongs to the searched group
                        if (entriesInfo && QString::compare(entriesInfo->getElementGroup(), groupNamePath) == 0)
                        {
                            // use name of elemet as name, e.g Shortcut:bin
                            QString name = (*iter)->getName();
                            // if element with this name has not been added, add it. Otherwise ignore
                            if (!groupList.contains(name))
                            {
                                groupList.insert(name, (*iter));
                            }
                        }
                    }
                }
            }
        }
    }

    return groupList;
}

void coEditorMainWindow::createTreeModel(QStringList elementGroups)
{
    if (elementGroups.isEmpty())
    {
        statusBar()->showMessage(tr("coEditorMainWindow::createTreeModel warn: No elementGroups"), 10000);
        return;
    }
    else
    {
        QStandardItemModel *model = static_cast<QStandardItemModel *>(treeView->model());
        model->clear();
        QStandardItem *parentItem = model->invisibleRootItem();
        // create group ALL at the beginning of the tree
        //       QStandardItem *generalItem = new QStandardItem ("ALL");
        //       generalItem->setData ("ALL");
        //       generalItem->setEditable (0);
        //       parentItem->appendRow (generalItem);

        for (QStringList::const_iterator group = elementGroups.begin(); group != elementGroups.end(); ++group)
        {
            //group should be sth like LOCAL.Cover.Plugin,,  make a tree
            QStringList list = (*group).split("."); //so or -> ?
            for (int i = 0; i < list.size(); ++i)
            {
                // check from left if parts of this new group are already created
                QString test = group->section(".", i, i);
                QList<QStandardItem *> searchedItem = model->findItems(test, Qt::MatchRecursive);
                if (searchedItem.isEmpty())
                {
                    //create new element in Tree, set its data to elementGroup,
                    QStandardItem *item = new QStandardItem(test);
                    item->setData(group->section(".", 0, i));
                    item->setEditable(0);
                    parentItem->appendRow(item);
                    parentItem = item;
                }
                // if a part already exists, set parentitem to be it
                else
                    parentItem = searchedItem.takeFirst();
            }
            parentItem = model->invisibleRootItem();
        }
    }
}

// delete all coEditorGroupWidgets
void coEditorMainWindow::clearData()
{
    mainWinHostConfigs.clear();
    QList<coEditorGroupWidget *> coEditorGroupWidgets = stackedCenterWidget->findChildren<coEditorGroupWidget *>();
    for (int i = 0; i < coEditorGroupWidgets.size(); ++i)
    {
        delete coEditorGroupWidgets.at(i);
    }
}

void coEditorMainWindow::setValue(const QString &variable, const QString &value, const QString &section, const QString &targetHost)
{
    QString target = targetHost;
    if (target.isEmpty())
    {
        target = hostsComboBox->currentText();
    }
    // section is GLOBAL.FSFSDF.aaa.bbb or LOCAL.FSFSDF.aaa.bbb with e.g. targetHost visible. Therefore GLOBAL or LOCAL has to be cut.
    QString saveTosection = section.section("AL.", 1);
    if (section.isEmpty())
    {
        statusBar()->showMessage(tr("Warning: Section ist Empty"), 8000);
        return;
    }

    if (target.toUpper() == "GLOBAL")
    {
        //section is like GLOBAL.Plugin, now remove GLOBAL
        //section.section (".", 1);
        //       cerr << "coEditorMainWindow::set gobal var "  << variable.toLatin1().data() << " value " << value.toLatin1().data() << " SaveTosection " << saveTosection.toLatin1().data() << " Insection " << section.toLatin1().data() << endl;
        oneConfigGroup->setValue(variable, value, section /*, currentFile, target */);
    }
    else
    {
        // NOTE currentFile must have a value
        //       cerr << "coEditorMainWindow::set local var "  << variable.toLatin1().data() << " value " << value.toLatin1().data() << " section " << section.toLatin1().data()  << " Insection " << section.toLatin1().data() << " file " << currentFile.toLatin1().data() << " " << endl;
        oneConfigGroup->setValue(variable, value, section, currentFile, target);
    }

    statusBar()->showMessage(tr("set a value"), 2000);
}

void coEditorMainWindow::deleteValue(const QString &variable, const QString &section,
                                     const QString &targetHost)
{
    QString target = targetHost;
    //    QString saveTosection = section;
    if (target.isEmpty())
    {
        target = hostsComboBox->currentText();
    }
    if (target.toUpper() == "GLOBAL")
    {
        oneConfigGroup->deleteValue(variable, section /*.section ("AL.", 1)*/ /*, currentFile, target*/);
    }
    else
    {
        oneConfigGroup->deleteValue(variable, section /*.section ("AL.", 1)*/, currentFile, target);
    }

    statusBar()->showMessage(tr("deleted a value"), 2000);
}

void coEditorMainWindow::about()
{
    QMessageBox::about(this, tr("About Config Editor"),
                       tr("This <b>Config Editor</b> is build to easily and  "
                          "safely configure Covise. Input is verified "
                          "by a Schema. "
                          ""));
}

// read custom window location and size
void coEditorMainWindow::readSettings()
{
    QSettings settings("COVISE", "COVISE Config Editor");
    QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
    QSize size = settings.value("size", QSize(400, 400)).toSize();
    resize(size);
    move(pos);
}

// save custom window location and size
void coEditorMainWindow::writeSettings()
{
    QSettings settings("COVISE", "COVISE Config Editor");
    settings.setValue("pos", pos());
    settings.setValue("size", size());
}
