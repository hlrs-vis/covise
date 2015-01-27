/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef INVMAIN_H
#define INVMAIN_H

#include <util/coTypes.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/times.h>
#endif

#include <QMoveEvent>
#include <QCloseEvent>
#include <QResizeEvent>
#include <QMenu>
#include <QTreeView>
#include <QMainWindow>
#include <QListWidget>
#include <QListWidgetItem>
#include <QStringListModel>
#include <objTreeWidget.h>
#include <objTreeItem.h>

#include <config/coConfig.h>
#include "InvViewer.h"

class QWidget;
class QString;
class QTabWidget;
class QFont;
class QListWidget;
class QListView;
class QMenu;
class QListWidgetItem;

class InvCommunicator;
class InvObjectManager;
class InvMsgManager;
class InvMain;
class InvSequencer;

class SoQtColorEditor;
namespace covise
{
class ApplicationProcess;
}

extern InvMain *renderer;

class InvMain : public QMainWindow
{
    Q_OBJECT

public:
    InvMain(int, char **);
    ~InvMain();

    enum appl_port_type
    {
        DESCRIPTION = 0, //0
        INPUT_PORT, //1
        OUTPUT_PORT, //2
        PARIN, //3
        PAROUT //4
    };

    enum
    {
        SYNC_LOOSE,
        SYNC_SYNC,
        SYNC_TIGHT
    };

    QTabWidget *listTabs;
    InvSequencer *sequencer;
    InvObjectManager *om;
    InvViewer *viewer;
    objTreeWidget *treeWidget;
    QTreeView *objTreeView;
    /* QListView            *colorListBox;
   QStringList          *colorStringList;
   QStringListModel     *colorListModel;*/
    QListWidget *colorListWidget;
#ifndef YAC
    InvMsgManager *mm;
    InvCommunicator *cm;
#endif

    covise::coConfigGroup *renderConfig;
    covise::ApplicationProcess *appmod;
    int port;
    int proc_id;
    int socket_id;
    QString host;
    QString proc_name;
    QString instance;

    bool isMaster()
    {
        return master;
    };
    void setMaster(int mode);
    int getSyncMode()
    {
        return synced;
    };
    void setSyncMode(int mode);
    void switchSyncMode();
    void switchAxisMode(int);
    int getRendererPropMode()
    {
        return rendererProp;
    };
    void setRendererPropMode(int mode);

    QString &getUsername()
    {
        return m_username;
    };

    void insertColorListItem(const char *name, const char *colormap);

#ifndef _WIN32
    struct tms tinfo;
    clock_t tp_lasttime, cam_lasttime;
    clock_t curr_time;
    float tp_rate;
    double diff;
#endif

    enum appl_port_type port_type[200];
    int port_required[200];
    int port_immediate[200];
    int modId;

    QString port_description[200];
    QString port_datatype[200];
    QString port_dependency[200];
    QString port_default[200];
    QString port_name[200];
    QString module_description;
    QString m_name;
    QString h_name;
    QString render_name;
    QString m_username;
    QString hostname;

    QFont boldfont;

private:
    SoQtColorEditor *backgroundColorEditor;

    QMenu *sync, *file, *viewing, *manip, *edit, *renderer_props;
    int master, synced, rendererProp, rendererPropBlending, rendererProperties;
    int iCaptureWindow, m_iShowWindowDecorations, m_iShowFullSizeWindow;

    QAction *NoCouplingAction, *MasterSlaveAction, *TightCouplingAction;
    QAction *BBAction[4];
    QAction *EBAction, *SCAction, *CPAction, *SWAction, *SFAction;
    QAction *CaptureAction, *SnapshotAction;
    QAction *MaterialEditorAction, *ColorEditorAction, *ObjectTransformAction, *PartsAction, *SnapHandleToAxisAction, *FreeHandleMotionAction, *NumericClipPlaneAction;

    int fid[8];
    QAction *mid[8];

#ifndef YAC
    void set_module_description(QString);
    void add_port(enum appl_port_type type, QString);
    void add_port(enum appl_port_type type, QString, QString, QString);
    void set_port_description(QString, QString);
    void set_port_default(QString, QString);
    void set_port_datatype(QString, QString);
    void set_port_required(QString, int);
    void set_port_immediate(QString, int);
    QString get_description_message();
    void printDesc(const char *);
#endif

    void makeLayout();
    void createMenubar();

    //void  editBackgroundColor();
    //static void backgroundColorCB(void *, const SbColor *);

protected:
    void closeEvent(QCloseEvent *);
    void resizeEvent(QResizeEvent *);
    void moveEvent(QMoveEvent *);

private slots:
    void sendSyncMode(bool);
    void rendererPropMode(bool);
    void setCapture(bool);
    void doSnap(bool);
    void colorSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
    void objectListCB(const QModelIndex &index);

    void doEditBackground(bool);
    void showCoordinateAxis(bool);
    void clippingPlane(bool);
    void showWindowDecoration(bool);
    void showFullSizeWindow(bool);

    void doMaterialEditor(bool);
    void doColorEditor(bool);
    void doObjectTransform(bool);
    void doParts(bool);
    void doSnapHandleToAxis(bool);
    void doFreeHandleMotion(bool);
    void doNumericClipPlane(bool);

    void ManipT(bool);
    void ManipH(bool);
    void ManipJ(bool);
    void ManipC(bool);
    void ManipTF(bool);
    void ManipTB(bool);
    void ManipNone(bool);
};

#endif
