/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_USERINTERFACE_H
#define ME_USERINTERFACE_H

#include "MEModuleTree.h"

#include <QMainWindow>
#include <QTabWidget>
#include <QLineEdit>
#include <QToolBar>
#include <QTabBar>
#include <QAction>

#include <config/config.h>

class QPalette;
class QLabel;
class QStackedWidget;
class QSplitter;
class QTextEdit;
class QProgressBar;
class QLineEdit;
class QFrame;
class QPushButton;
class QLineEdit;
class QComboBox;
class QSpinBox;
class QDockWidget;

class VinceRendererWidget;
class SGLogWindow;

namespace covise
{
class coConfigGroup;
class coSendBuffer;
struct CRB_EXEC;
} // namespace covise

class MEUserInterface;
class MEMainHandler;
class MEModulePanel;
class MEDataTree;
class MEModuleParameter;
class MEGraphicsView;
class MEGridProxy;
class MEColorMap;
class MEDataViewer;
class MEControlPanel;
class MEHost;
class METimer;
class METoolBar;
class MENode;
class MEFileBrowser;
class METabBar;
class METabWidget;
class MEMessageHandler;
class MEDaemon;
class MERegistry;
class MEPreference;

#ifdef TABLET_PLUGIN
class TUIMainFrame;
#endif

typedef VinceRendererWidget *(*VinceRendererWidgetCreate)(QWidget *w, int argc, const char **argv);

class MEUserInterface : public QMainWindow
{
    friend class MEGraphicsView; // for the actions
    friend class MEMainHandler;
    Q_OBJECT

public:
    MEUserInterface(MEMainHandler *handler);

    static MEUserInterface *instance();

    bool hasMiniGUI()
    {
        return m_miniGUI;
    };
    bool rendererToBeStarted()
    {
        return m_willStartRenderer;
    };
    void init();
    void hideUnusedModulesHasChanged();
    void deleteSelectedNodes();
    void openChatWindow();
    void storeSessionParam(bool);
    void switchMasterState(bool);
    void enableExecution(bool);
    void enableUndo(bool);
    void changeExecButton(bool);
    void changeEditItems(bool);
    void setCollabItems(int, bool);
    void resetStatusBar();
    void addHostToModuleTree(MEHost *);
    void switchModuleTree(MEModuleTree *, bool);
    void resetModuleFilter();
    void printMessage(const QString &);
    void showMapName(const QString &);
    void updateMainBrowser(const QStringList &);
    void openBrowser(const QString &title, const QString &mapName, int mode);
    void lookupResult(const QString &text, const QString &filename, QString &type);
    void writeInfoMessage(const QString &text, const QColor &color = Qt::black);
    void writeChatContent(const QStringList &);
    void removeTabletUI();
    void startRenderer(const covise::CRB_EXEC &message);
    void stopRenderer(const QString &name, const QString &number, const QString &host);
    void setMiniGUI(bool);
    void activateTabletUI();
    void reset();
    void quit();
    void showFavoriteLabel()
    {
        m_favoriteLabel_a->setVisible(true);
    };
    void hideFavoriteLabel()
    {
        m_favoriteLabel_a->setVisible(false);
    };
    void showMessageArea(bool);

    MEModuleTree *getModuleTree()
    {
        return m_moduleTree.get();
    };
    MEModuleTree *getFilterTree()
    {
        return m_filterTree.get();
    };
    MEColorMap *getColorMap()
    {
        return m_colorMap;
    };
    QLineEdit *getChatLine()
    {
        return m_chatLine;
    };
    METoolBar *getToolBar()
    {
        return m_toolBar;
    };

    void clearOpenAutosaveMenu();
    void insertIntoOpenRecentMenu(QAction *ac);

public slots:

    void showDataViewer(bool);
    void showControlPanel(bool);
    void developerMode(bool);

private:
    static MEUserInterface *m_singleton;
    MEMainHandler *m_mainHandler = nullptr;
    MEGraphicsView *m_graphicsView = nullptr;

    int m_errorNumber = 0, m_errorLevel = 0;
    int m_foregroundCount = 0;
    bool m_tabletUIisDead = false, m_willStartRenderer = false, m_miniGUI = false;

    QString m_renderName, m_renderInstance, m_renderHost;
    QWidget *m_renderer = nullptr;
    QProgressBar *m_progressBar = nullptr;
    QPushButton *m_messageWindowPB = nullptr, *m_restoreListPB = nullptr;
    QLabel *m_messageWindowText = nullptr, *m_miniMapLabel = nullptr;
    QTextEdit *m_infoWindow = nullptr;
    QMenu *m_openRecentMenu = nullptr, *m_openAutosaveMenu = nullptr;

    QAction *m_filenew_a = nullptr, *m_fileopen_a = nullptr, *m_filesave_a = nullptr, *m_filesaveas = nullptr, *m_settings_a = nullptr;
    QAction *m_colormap_a = nullptr, *m_snapshot_a = nullptr, *m_exit_a = nullptr, *m_exec_a = nullptr, *m_master_a = nullptr;
    QAction *m_addpartner_a = nullptr;
    QAction *m_setmirror_a = nullptr, *m_startmirror_a = nullptr, *_delmirror = nullptr;
    QAction *m_about_a = nullptr, *m_about_qt_a = nullptr, *m_tutorial_a = nullptr, *m_usersguide_a = nullptr, *m_moduleguide_a = nullptr, *m_progguide_a = nullptr, *m_reportbug_a = nullptr;
    QAction *m_gridproxy_a = nullptr, *m_whatsthis_a = nullptr, *m_undo_a = nullptr;
    QAction *m_selectAll_a = nullptr, *m_deleteAll_a = nullptr, *m_keyDelete_a = nullptr;
    QAction *m_showDataViewer_a = nullptr;
    QAction *m_showTabletUI_a = nullptr, *m_showToolbar_a = nullptr, *m_showMessageArea_a = nullptr, *m_showControlPanel_a = nullptr;
    QAction *m_execOnChange_a = nullptr, *m_help_a = nullptr;
    QAction *m_showCME_a = nullptr, *m_showReg_a = nullptr, *m_viewAll_a = nullptr, *m_view100_a = nullptr, *m_view50_a = nullptr;
    QAction *m_actionCopy = nullptr, *m_actionCut = nullptr, *m_actionPaste = nullptr;
    QAction *m_layoutMap_a = nullptr, *m_favoriteLabel_a = nullptr, *m_comboLabel_a = nullptr;
    QAction *m_comboSeparator = nullptr, *m_favSeparator = nullptr;

    METoolBar *m_miniToolbar = nullptr, *m_toolBar = nullptr;
    QWidget *m_main = nullptr, *m_mainRight = nullptr, *m_favorite = nullptr, *m_chat = nullptr, *m_info = nullptr;
    QStackedWidget *m_widgetStack = nullptr;
    QSplitter *m_mainArea = nullptr;
    QMainWindow *m_miniUserInterface = nullptr;
    QLineEdit *m_chatLine = nullptr;
    QComboBox *m_scaleViewBox = nullptr;
    QSpinBox *m_visibleArray = nullptr;
    QLabel *m_chatLineLabel = nullptr;
    QLineEdit *m_filterLine = nullptr;
    QDockWidget *m_bottomDockWindow = nullptr, *m_bottomChatWindow = nullptr;
    QList<QAction *> m_toolBarActionList, m_pipeActionList, m_fileActionList, m_sessionActionList, m_editActionList;

    std::unique_ptr<MEModuleTree> m_moduleTree, m_filterTree;
    MEDataViewer *m_dataPanel = nullptr;
    MEGridProxy *m_gridProxyBox = nullptr;
    MEColorMap *m_colorMap = nullptr;
    MEFileBrowser *m_mainBrowser = nullptr;
    METabBar *m_tabBar = nullptr;
    METabWidget *m_tabWidgets = nullptr;

#ifdef TABLET_PLUGIN
    TUIMainFrame *m_tablet = nullptr;
#endif
    covise::coConfigGroup *getConfig() const;
    
    covise::config::File &m_mapConfig;
    std::unique_ptr<covise::ConfigBool> m_showDataViewerCfg, m_showTabletUICfg, m_showToolbarCfg, m_showMessageAreaCfg, m_showControlPanelCfg;
    

    std::atomic_bool m_testScreenOffset{false};
    QPoint m_screenOffset;

    void createActions();
    void createMenubar();
    void createToolbar();
    void makeMainWidgets();
    void makeCenter();
    void makeStatusBar();
    void makeLeftContent(QWidget *);
    void makeRightContent(QWidget *);
    void makeParameterWindow();
    void makeMessageArea();

public slots:

    void updateScaleViewCB(qreal);

private slots:

    void messagePBClicked();
    void restartRenderer();
    void removeRenderer();
    void gridProxy();
    void showTabletUI(bool);
    void showMatchingNodes();
    void showColorMap(bool);
    void scaleViewCB(const QString &);
    void tabChanged(int index);
    void filterCB();
    void bringApplicationToForeground();

protected:
    void closeEvent(QCloseEvent *);
    QMenu *createPopupMenu();
    VinceRendererWidgetCreate getRendererFunction;
};

// modified  classes for adaption of standard layout
//======================================================
class METoolBar : public QToolBar
{
    Q_OBJECT

public:
    METoolBar(QWidget *parent = 0);

protected:
    void dropEvent(QDropEvent *e);
    void dragEnterEvent(QDragEnterEvent *e);
    void dragLeaveEvent(QDragLeaveEvent *e);
    void dragMoveEvent(QDragMoveEvent *e);
};

class METabWidget : public QTabWidget
{
    Q_OBJECT

public:
    METabWidget(QWidget *parent = 0);
    void setTabBar(METabBar *tb);
};

class METabBar : public QTabBar
{
    Q_OBJECT

public:
    METabBar(QWidget *parent = 0);

protected:
    QSize tabSizeHint(int) const;
};

class MEFilterLineEdit : public QLineEdit
{
public:
    MEFilterLineEdit(QWidget *parent = 0);

private:
    virtual void keyPressEvent(QKeyEvent *ev);
    virtual bool event(QEvent *ev);
};
#endif
