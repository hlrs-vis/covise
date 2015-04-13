/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MAINHANDLER_H
#define ME_MAINHANDLER_H

#include <QLineEdit>
#include <QMap>
#include <QSslError>

class QListWidget;
class QListWidgetItem;
class QNetworkReply;

class MEUserInterface;
class MEMainHandler;
class MEMessageHandler;
class MELinkListHandler;
class MEHelpViewer;
class MESessionSettings;
class MECSCW;
class MECSCWParam;
class MENode;
class MEPort;
class MEHost;
class MEModuleTree;
class METimer;
class MEMessageHandler;
class MEDeleteHostDialog;
class MEMirrorHostDialog;

namespace covise
{
class coSendBuffer;
class coRecvBuffer;
class Message;
}

#include <config/coConfig.h>

class MEMainHandler : public QObject
{
    friend class MEMessageHandler;

    Q_OBJECT

public:
    MEMainHandler(int, char *[]);
    ~MEMainHandler();

    static MEMainHandler *instance();

    enum copyModes
    {
        NORMAL = 0,
        COPY,
        MOVE,
        SYNC = 4,
        DEBUG,
        MEMCHECK,
        MOVE_DEBUG,
        MOVE_MEMCHECK
    };
    enum addHostModes
    {
        UNKNOWN,
        ADDHOST,
        ADDPARTNER
    };

    QString localHost, localUser, onlineDir, localIP;
    static QString framework;
    static QFont s_normalFont, s_boldFont, s_itBoldFont, s_italicFont;
    static QSize screenSize;
    static QPalette defaultPalette;

    QPixmap pm_table, pm_bullet, pm_bulletmore, pm_bookopen, pm_bookclosed;
    QPixmap pm_folderopen, pm_folderclosed, pm_forward, pm_master, pm_adduser;
    QPixmap pm_pinup, pm_pindown, pm_host, pm_file, pm_copy, pm_help, pm_logo;
    QPixmap pm_stop, pm_exec, pm_exec2, pm_colorpicker, pm_addhost;
    QPixmap pm_collapse, pm_expand, pm_lighton, pm_lightoff;
    QStringList moduleHistory, networkHistory;

    covise::coConfigBool cfg_storeWindowConfig, cfg_ErrorHandling, cfg_DeveloperMode;
    covise::coConfigBool cfg_HideUnusedModules, cfg_AutoConnect, cfg_TopLevelBrowser, cfg_ImbeddedRenderer;
    covise::coConfigInt cfg_AutoSaveTime, cfg_ModuleHistoryLength, cfg_GridSize;
    covise::coConfigString cfg_HostColors, cfg_QtStyle, cfg_HighColor;
    QString cfg_SavePath;
    int cfg_NetworkHistoryLength;

    static QColor s_paramColor, s_reqMultiColor, s_reqDataColor, s_multiColor, s_dataColor, s_chanColor;
    static QColor s_highlightColor, s_requestedColor, s_defColor, s_dependentColor, s_optionalColor;

    QVector<METimer *> timerList;
    QAction *m_deleteAutosaved_a;
    QList<QAction *> recentMapList, autosaveMapList;
    QMap<QAction *, QString> autosaveNameMap;

    bool isInMapLoading()
    {
        return m_inMapLoading;
    };
    bool isExecOnChange()
    {
        return m_executeOnChange;
    };
    bool isThereAnyHelp()
    {
        return m_helpExist;
    };
    bool isThereAWebHelp()
    {
        return m_helpFromWeb;
    };
    bool isWaitingForClose()
    {
        return m_waitForClose;
    };
    bool getConnectedPartner()
    {
        return m_connectedPartner;
    };
    bool canQuitSession();

    int getAddHostMode()
    {
        return m_hostMode;
    };
    int isMaster();
    int getGridSize();
    int getPortSize()
    {
        return m_portSize;
    };
    int getSliderWidth()
    {
        return m_sliderWidth;
    };

    void reset();
    void init();
    void quit();
    void checkHelp();
    void updateTimer();
    void setMaster(bool);
    void changeExecButton(bool);
    void enableExecution(bool);
    void setInMapLoading(bool);
    void removeNodesOfHost(MEHost *);
    void addNewHost(MEHost *);
    void closeApplication(QCloseEvent *);
    void insertModuleInHistory(const QString &module);
    void insertNetworkInHistory(const QString &module);
    void saveNetwork(const QString &mapname);
    void openDroppedMap(const QString &mapname);
    void requestNode(const QString &module, const QString &host, int x, int y, MENode *, copyModes);
    void mapWasChanged(const QString &reason);
    void showModuleHelp(const QString &category, const QString &module);
    void updateLoadedMapname(const QString &mapname);
    void initNode(const QStringList &list);
    void showHostState(const QStringList &list);
    void removeNode(MENode *);
    void moveNode(MENode *, int x, int y);
    void addLink(MENode *n1, MEPort *p1, MENode *n2, MEPort *p2);
    void removeLink(MENode *n1, MEPort *p1, MENode *n2, MEPort *p2);
    void removeHost(MEHost *host);
    void setCloseImm()
    {
        force = true;
    };
    void updateHistoryFiles(const QString &);
    void storeMapName(const QString &mapname);
    void showCSCWParameter(const QStringList &, addHostModes);
    void showCSCWDefaults(const QStringList &);
    void initHost(const QStringList &);
    void finishNode(const QStringList &);
    void setDescriptionOfNode(const QStringList &);
    void showClipboardNodes(const QStringList &);
    void startNode(const QString &module, const QString &instance, const QString &host);

    covise::coConfigGroup *getConfig() const;
    QString getMapPath();
    QString getMapName();
    QString getLibraryName()
    {
        return m_libFileName;
    };
    QString generateTitle(const QString &text);
    QColor getHostColor(int);

    MECSCW *getAddHostBox()
    {
        return m_addHostBox;
    };

    void initNode(int nodeid, MEHost *host, covise::coRecvBuffer &tb);

signals:

    void usingNode(const QString &);
    void developerMode(bool);

public slots:

    void initModule();
    void clearNet();
    void autoSaveNet();
    void about();
    void aboutQt();
    void reportbug();
    void tutorial();
    void usersguide();
    void moduleguide();
    void progguide();
    void help();
    void mapeditor();
    void chatCB();
    void openNet();
    void saveNet();
    void addHost();
    void saveAsNet();
    void addPartner();
    void execNet();
    void settingXML();
    void deleteSelectedNodes();
    void changeCB(bool);
    void masterCB();
    void undoAction();
    void openNetworkFile(bool);
    void openNetworkFile(QString);
    void deleteAutosaved(bool);
    void developerModeHasChanged();
    void setMapModified(bool);
    void execTriggered();

private:
    static MEMainHandler *singleton;
    static MEUserInterface *mapEditor;
    static MEMessageHandler *messageHandler;

    copyModes m_copyMode;
    addHostModes m_hostMode;

    bool m_helpFromWeb, m_helpExist;
    bool m_masterUI, force, m_loadedMapWasModified, m_autoSave;
    bool m_waitForClose;
    bool m_executeOnChange, m_inMapLoading;
    int m_mirrorMode, m_localHostID, m_connectedPartner;
    int m_portSize, m_sliderWidth;

    QString m_mapName, m_libFileName;
    QStringList m_hostColor;
    QLineEdit *m_selectHostLine;
    QTimer *m_autoSaveTimer;

    QVector<MEHost *> m_syncList;

    MENode *m_currentNode, *m_newNode;
    MESessionSettings *m_settings;
    MECSCW *m_addHostBox;
    MECSCWParam *m_CSCWParam;
    MEDeleteHostDialog *m_deleteHostBox;
    MEMirrorHostDialog *m_mirrorBox;

    bool m_requestingMaster;

    void setHostMode(addHostModes mode)
    {
        m_hostMode = mode;
    };
    void switchModuleTree(MEModuleTree *);
    void storeSessionParam();
    void readConfigFile();
    void openBrowser(const QString &title, int openmode);
    void makeDeleteHostBox();
    void embeddedRenderer();
    void saveMap();
    void setLocalHost(int id, const QString &name, const QString &user);

private slots:

    void printCB();
    void delHost();
    void onlineCB(const QString &html);
    void getHostCB(QListWidgetItem *);
    void handleSslErrors(QNetworkReply *reply, const QList<QSslError> &errors);

protected:
    covise::coConfigGroup *mapConfig;
};
#endif
