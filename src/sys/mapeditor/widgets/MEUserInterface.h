/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_USERINTERFACE_H
#define ME_USERINTERFACE_H

#include <QMainWindow>
#include <QTabWidget>
#include <QLineEdit>
#include <QToolBar>
#include <QTabBar>
#include <QAction>

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
class CRB_EXEC;
} // namespace covise

class MEUserInterface;
class MEMainHandler;
class MEModulePanel;
class MEDataTree;
class MEModuleParameter;
class MEModuleTree;
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

class TUIMainWindow;

typedef VinceRendererWidget *(*VinceRendererWidgetCreate)(QWidget *w, int argc, const char **argv);

class MEUserInterface : public QMainWindow
{
    friend class MEGraphicsView; // for the actions
    friend class MEMainHandler;
    Q_OBJECT

public:
    MEUserInterface();
    ~MEUserInterface();

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
        return m_moduleTree;
    };
    MEModuleTree *getFilterTree()
    {
        return m_filterTree;
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
    static MEMainHandler *m_mainHandler;
    static MEGraphicsView *m_graphicsView;

    int m_progress, m_errorNumber, m_errorLevel;
    int m_foregroundCount;
    bool m_tabletUIisDead, m_willStartRenderer, m_miniGUI;

    QString m_renderName, m_renderInstance, m_renderHost;
    QWidget *m_renderer;
    QProgressBar *m_progressBar;
    QPushButton *m_messageWindowPB, *m_restoreListPB;
    QLabel *m_messageWindowText, *m_miniMapLabel;
    QTextEdit *m_infoWindow;
    QMenu *m_openRecentMenu, *m_openAutosaveMenu;

    QAction *m_filenew_a, *m_fileopen_a, *m_filesave_a, *m_filesaveas, *m_settings_a;
    QAction *m_colormap_a, *m_snapshot_a, *m_exit_a, *m_exec_a, *m_master_a;
    QAction *m_addhost_a, *m_addpartner_a, *m_delhost_a;
    QAction *m_setmirror_a, *m_startmirror_a, *_delmirror;
    QAction *m_about_a, *m_about_qt_a, *m_tutorial_a, *m_usersguide_a, *m_moduleguide_a, *m_progguide_a, *m_reportbug_a;
    QAction *m_gridproxy_a, *m_whatsthis_a, *m_undo_a;
    QAction *m_selectAll_a, *m_deleteAll_a, *m_keyDelete_a;
    QAction *m_showDataViewer_a;
    QAction *m_showTabletUI_a, *m_showToolbar_a, *m_showMessageArea_a, *m_showControlPanel_a;
    QAction *m_execOnChange_a, *m_help_a;
    QAction *m_showCME_a, *m_showReg_a, *m_viewAll_a, *m_view100_a, *m_view50_a;
    QAction *m_actionCopy, *m_actionCut, *m_actionPaste;
    QAction *m_layoutMap_a, *m_favoriteLabel_a, *m_comboLabel_a;
    QAction *m_comboSeparator, *m_favSeparator;

    METoolBar *m_miniToolbar, *m_toolBar;
    QWidget *m_main, *m_mainRight, *m_favorite, *m_chat, *m_info;
    QStackedWidget *m_widgetStack;
    QSplitter *m_mainArea;
    QMainWindow *m_miniUserInterface;
    QLineEdit *m_chatLine;
    QComboBox *m_scaleViewBox;
    QSpinBox *m_visibleArray;
    QLabel *m_chatLineLabel;
    QLineEdit *m_filterLine;
    QDockWidget *m_bottomDockWindow, *m_bottomChatWindow;
    QList<QAction *> m_toolBarActionList, m_pipeActionList, m_fileActionList, m_sessionActionList, m_editActionList;

    MEModuleTree *m_moduleTree, *m_filterTree;
    MEDataViewer *m_dataPanel;
    MEGridProxy *m_gridProxyBox;
    MEColorMap *m_colorMap;
    MEFileBrowser *m_mainBrowser;
    METabBar *m_tabBar;
    METabWidget *m_tabWidgets;

    TUIMainWindow *m_tablet;
    covise::coConfigGroup *getConfig() const;

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

    int getProgress()
    {
        return m_progress;
    };
    void setProgress(int value)
    {
        m_progress = value;
    };

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
    void showRegistry(bool);
    void scaleViewCB(const QString &);
    void tabChanged(int index);
    void filterCB();
    void bringApplicationToForeground();

protected:
    void closeEvent(QCloseEvent *);
    QMenu *createPopupMenu();
    covise::coConfigGroup *mapConfig;
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
