/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_MAINHANDLER_H
#define ME_MAINHANDLER_H

#include <QLineEdit>
#include <QMap>
#include <QSslError>
#include <QSettings>

#include <vector>
#include <string>
#include <utility>
#include <mutex>
#include <atomic>
#include <messages/NEW_UI.h>

#include "../covise/MEMessageHandler.h"
#include "widgets/MEUserInterface.h"
#include "widgets/MEHelpViewer.h"
class QListWidget;
class QListWidgetItem;
class QMessageBox;
class QNetworkReply;

class MENode;
class MEPort;
class MESessionSettings;
class MERemotePartner;
class MECSCWParam;
class MEDeleteHostDialog;
class MEMirrorHostDialog;
namespace covise
{
enum class LaunchStyle : int;
class coSendBuffer;
class coRecvBuffer;
class Message;
class NonBlockingDialogue;
} // namespace covise

#include <config/coConfig.h>
#include <config/config.h>
class MEMainHandler : public QObject
{
    friend class MEMessageHandler;

    Q_OBJECT
private:
    covise::config::File m_mapEditorConfig;
    QSettings m_guiSettings;

public:
    MEMainHandler(int, char *[], std::function<void(void)> quitFunc);

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

    std::unique_ptr<covise::ConfigBool> cfg_DeveloperMode,
                                        cfg_storeWindowConfig,
                                        cfg_ErrorHandling,
                                        cfg_HideUnusedModules,
                                        cfg_AutoConnect,
                                        cfg_TopLevelBrowser,
                                        cfg_ImbeddedRenderer,
                                        cfg_TabletUITabs;

    std::unique_ptr<covise::ConfigInt> cfg_AutoSaveTime, cfg_ModuleHistoryLength, cfg_NetworkHistoryLength, cfg_GridSize;
    std::unique_ptr<covise::ConfigString> cfg_HighColor;
    std::unique_ptr<covise::ConfigArray<std::string>> cfg_HostColors;

    QString cfg_QtStyle()
    {
        return QString(std::string(m_cfg_QtStyle).c_str());
    }

    void cfg_QtStyle(const QString &s)
    {
        m_cfg_QtStyle = s.toStdString();
    }

private:
    covise::coConfigString m_cfg_QtStyle; //this ist not updated yet because the value is not mapeditor only

public:
    bool isDeveloperMode() const;
    void setDeveloperMode(bool devMode);
    QString cfg_SavePath;

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
        return m_helpViewer != nullptr;
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
        return 0;
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
    void checkHelp();
    void updateTimer();
    void setMaster(bool);
    void changeExecButton(bool);
    void enableExecution(bool);
    void setInMapLoading(bool);
    void removeNodesOfHost(MEHost *);
    void addNewHost(MEHost *);
    void closeApplication(QCloseEvent *);
    void insertNetworkInHistory(const QString &module);
    void saveNetwork(const QString &mapname);
    void openDroppedMap(const QString &mapname);
    void requestNode(const QString &module, const QString &hostIP, int x, int y, MENode *, copyModes);
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
        m_force = true;
    };
    void updateHistoryFiles(const QString &);
    void storeMapName(const QString &mapname);
    void initHost(int clientId, const covise::UserInfo &partnerInfo, const std::vector<std::string> &modules, const std::vector<std::string> &categories);
    void finishNode(const QStringList &);
    void setDescriptionOfNode(const QStringList &);
    void showClipboardNodes(const QStringList &);
    void startNode(const QString &module, const QString &instance, const QString &host);

    covise::config::File &getConfig(); // config file that migt be edited  
    QSettings &getUserBehaviour(); //registry entry that stores usage parameteters and is not meant to be modified manually
    QString getMapPath();
    QString getMapName();
    QString getLibraryName()
    {
        return m_libFileName;
    };
    QString generateTitle(const QString &text);
    QColor getHostColor(int);

    void initNode(int nodeid, MEHost *host, covise::coRecvBuffer &tb);
    void updateRemotePartners(const covise::ClientList &partners);
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
    void quit();

private:
    static MEMainHandler *singleton;
    std::unique_ptr<MEHelpViewer> m_helpViewer;
    MEUserInterface m_mapEditor;
    MEMessageHandler m_messageHandler;



    copyModes m_copyMode = NORMAL;
    std::function<void(void)> m_quitFunc;
    bool m_helpFromWeb = false;
    bool m_masterUI = false, m_force = false, m_loadedMapWasModified = false, m_autoSave = false;
    bool m_waitForClose = false;
    bool m_executeOnChange = false, m_inMapLoading = false;
    int m_mirrorMode, m_localHostID = -1, m_connectedPartner = 0;
    int m_portSize = 14, m_sliderWidth;
    std::mutex m_remotePartnerMutex;
    covise::ClientList m_remotePartners;

    QString m_mapName, m_libFileName;
    QLineEdit *m_selectHostLine = nullptr;
    QTimer *m_autoSaveTimer = nullptr;

    QVector<MEHost *> m_syncList;

    MENode *m_currentNode = nullptr, *m_newNode = nullptr;
    MESessionSettings *m_settings = nullptr;
    MERemotePartner *m_addPartnerDialog = nullptr;
    MECSCWParam *m_CSCWParam = nullptr;
    MEDeleteHostDialog *m_deleteHostBox = nullptr;
    MEMirrorHostDialog *m_mirrorBox = nullptr;

    bool m_requestingMaster = false;

    void switchModuleTree(MEModuleTree *);
    void storeCategoryExpandedState();
    void readConfigFile();
    void openBrowser(const QString &title, int openmode);
    void makeDeleteHostBox();
    void embeddedRenderer();
    void saveMap();
    void setLocalHost(int id, const QString &name, const QString &user);
    void requestPartnerAction(covise::LaunchStyle launchStyle, const std::vector<int> &clients);
private slots:

    void printCB();
    void onlineCB(const QString &html);
    void getHostCB(QListWidgetItem *);


};
#endif


