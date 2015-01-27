/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FILEBROWSER_DLG_H_
#define FILEBROWSER_DLG_H_
#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QComboBox>
#include <QDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QListView>
#include <QListWidget>
#include <QStringList>
#include <QPushButton>
#include <QSplitter>
#include <QVBoxLayout>
#include <QCheckBox>
#include <coTUIFileBrowser/IData.h>
#include "../TUIFileBrowserButton.h"
#include "RemoteClientDialog.h"

class TUIFileBrowserButton;
/**
 * The class provides a remote file browser capable of accessing files on
 * - the local machine
 * - a remote machine through a CRB
 * - an AccessGrid data store on a venue server through webservice calls
 * The class is usable with both the VRB and the Controler
 * @author Michael Braitmaier
 * @date 2007-01-12
 */
class FileBrowser : public QDialog
{
    Q_OBJECT
public:
    /**
       * The standard constructor for class initialization
       * @param parent - A optional given reference to the filebrowser's
       *                 parent. This is used for embedding the filebrowser
       *                 into other Qt widgets.
       * @param instance - TUIFileBrowserButton Instance when used in context of
       *				     tabletUI and OpenCOVER
       */
    FileBrowser(TUIFileBrowserButton *instance, QWidget *parent = 0, int id = 0);

    /**
       * The standard constructor for class initialization
       * @param parent - A optional given reference to the filebrowser's
       *                 parent. This is used for embedding the filebrowser
       *                 into other Qt widgets.
       */
    FileBrowser(QWidget *parent = 0);

    /**
       * The standard destructor for cleaning up used memory of
       * the class.
       */
    ~FileBrowser(void);

    //enumerations
    /**
       * The DialogModes are used upon initial creation of the FileBrowser
       * to indicate which mode the FileBrowser should run in.
       */
    enum DialogModes
    {
        OPEN = 1,
        SAVE = 2
    };

    //Property Access methods
    /**
       * Access method for read-only property Filename
       * @Return Returns the filenames of the user-selected files in the dialog.
       */
    QStringList *Filename();

    /**
       * Access methods for class attribute data object.
       */
    opencover::IData *DataObject();
    void DataObject(opencover::IData *value);

    /**
       * Acces method for class attribute LocationPath
       */
    QString LocationPath();
    void LocationPath(QString value);

    /**
       * Access methods for class attribute FilterList
       */
    QStringList *FilterList();
    void FilterList(QStringList *value);

    /**
       * Access mehtods for getting and setting the mode the dialog runs in.
       */
    void getMode(DialogModes *mode);
    void setMode(DialogModes mode);

    bool exists(QString file);

    Ui_RemoteClients *mRCDialog;

signals:
    void requestLists(QString filter, QString location);
    void filterChange(QString filter);
    void dirChange(QString dir);
    void fileSelected(QString file, QString dir, bool loadAll);
    void requestClients();
    void locationChanged(QString location);
    void requestLocalHome();
    void reqDriveList();
    void reqDirUp(QString path);
    void pathSelected(QString file, QString path);
    void fileNameEntered();

public slots:
    void handleCurDirUpdate(QString curDir);
    void handleDirUpdate(QStringList list);
    void handleFileUpdate(QStringList list);
    void handleClientUpdate(QStringList list);
    void remoteClientsAccept();
    void remoteClientsReject();
    void handleItemClicked(QListWidgetItem *item);
    void handleDriveUpdate(QStringList list);
    void handleUpdateMode(int mode);
    void handleUpdateFilterList(char *filterList);
    void handleLocationUpdate(QString strEntry);
    void handleupdateRemoteButtonState(int);
    void handleUpdateLoadCheckBox(bool);

private:
    /**
       * The method is used for complete creation and arrangement of all GUI widgets
       * required for the desires FileBrowser dialog. The actual layout is based upon
       * the selected DialogMode.
       * @see DialogModes
       */
    void CreateDialogLayout();

    //Class attributes
    /**
       * Stores the list of selected files which the user has selected
       * in the dialog. Allowed range of filenames is 1..n
       */
    QStringList *mFilename;

    /**
       * The mDataObject attribute contains a references to the FileBrowsers
       * underlying data management object, which performs file list retrieval
       * utilizing either a VRB, the Controler or AccessGrid.
       * @see IData
       */
    opencover::IData *mDataObject;

    /**
       * Path to the location the user has selected files in or is intended to
       * use as a startup location path.
       */
    QString mLocationPath;

    /**
       * List of filters limiting the amount of file-types shown in the FileBrowser
       */
    QStringList mFilterList;

    TUIFileBrowserButton *mTUIElement;

    int mId;

    DialogModes mMode;

    //QT-Dialog widgets
    QListWidget *lstDir;
    QListWidget *lstFile;
    QPushButton *btnCancel;
    QLineEdit *txtFilename;
    QComboBox *cmbFilter;
    QPushButton *btnAccept;
    QVBoxLayout *vboxLayout;
    QPushButton *btnHome;
    QPushButton *btnMachine;
    QPushButton *btnAccessGrid;
    QPushButton *btnRemote;
    QHBoxLayout *hboxLayout;
    QComboBox *cmbDirectory;
    QPushButton *btnDirUp;
    QComboBox *cmbHistory;
    QString mLocation;
    QCheckBox *chkLoadAllClients;

private slots:
    void onCancelPressed();
    void onAcceptPressed();
    void onHomePressed();
    void onMachinePressed();
    void onAccessGridPressed();
    void onRemotePressed();
    void onDirUpPressed();
    void onDirActivated(int index);
    void onHistoryActivated(int index);
    void onFilenameChanged(QString text);
    void onFilterActivated(int index);
    void onLstDirDoubleClick(QListWidgetItem *item);
    void onLstFileDoubleClick(QListWidgetItem *item);
    void onLstFileClick(QListWidgetItem *item);
};
#endif
