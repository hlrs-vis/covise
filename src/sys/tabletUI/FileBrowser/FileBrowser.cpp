/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FileBrowser.h"
#include "32x32/home32.xpm"
#include "32x32/computer32.xpm"
#include "32x32/network32.xpm"
#include "32x32/ag32.xpm"
#include "32x32/undo32.xpm"
#include "16x16/apply.xpm"
#include "16x16/stop.xpm"

#include <QDir>
#include <QMessageBox>

FileBrowser::FileBrowser(TUIFileBrowserButton *instance, QWidget *parent, int id)
{
    Q_UNUSED(parent);

    this->mTUIElement = instance;
    this->CreateDialogLayout();
    connect(lstDir, SIGNAL(itemDoubleClicked(QListWidgetItem *)), this, SLOT(onLstDirDoubleClick(QListWidgetItem *)));
    connect(lstFile, SIGNAL(itemDoubleClicked(QListWidgetItem *)), this, SLOT(onLstFileDoubleClick(QListWidgetItem *)));
    connect(lstFile, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(onLstFileClick(QListWidgetItem *)));
    connect(btnCancel, SIGNAL(pressed()), this, SLOT(onCancelPressed()));
    connect(txtFilename, SIGNAL(textEdited(QString)), this, SLOT(onFilenameChanged(QString)));
    connect(txtFilename, SIGNAL(returnPressed()), this, SLOT(onAcceptPressed()));
    connect(this, SIGNAL(fileNameEntered()), this, SLOT(onAcceptPressed()));
    connect(cmbFilter, SIGNAL(activated(int)), this, SLOT(onFilterActivated(int)));
    connect(btnHome, SIGNAL(pressed()), this, SLOT(onHomePressed()));
    connect(btnMachine, SIGNAL(pressed()), this, SLOT(onMachinePressed()));
    connect(btnRemote, SIGNAL(pressed()), this, SLOT(onRemotePressed()));
    connect(btnAccessGrid, SIGNAL(pressed()), this, SLOT(onAccessGridPressed()));
    connect(cmbDirectory, SIGNAL(activated(int)), this, SLOT(onDirActivated(int)));
    connect(btnDirUp, SIGNAL(pressed()), this, SLOT(onDirUpPressed()));
    connect(cmbHistory, SIGNAL(activated(int)), this, SLOT(onHistoryActivated(int)));
    connect(btnAccept, SIGNAL(pressed()), this, SLOT(onAcceptPressed()));
    connect(this, SIGNAL(requestLists(QString, QString)), this->mTUIElement, SLOT(handleRequestLists(QString, QString)));
    connect(this, SIGNAL(filterChange(QString)), this->mTUIElement, SLOT(handleFilterUpdate(QString)));
    connect(this, SIGNAL(dirChange(QString)), this->mTUIElement, SLOT(handleDirChange(QString)));
    connect(this, SIGNAL(requestClients()), this->mTUIElement, SLOT(handleClientRequest()));
    connect(this, SIGNAL(locationChanged(QString)), this->mTUIElement, SLOT(handleLocationChange(QString)));
    connect(this, SIGNAL(requestLocalHome()), this->mTUIElement, SLOT(handleLocalHome()));
    connect(this, SIGNAL(reqDriveList()), this->mTUIElement, SLOT(handleReqDriveList()));
    connect(this, SIGNAL(fileSelected(QString, QString, bool)), this->mTUIElement, SLOT(sendSelectedFile(QString, QString, bool)));

    this->mLocationPath = "";
    this->mId = id;
    this->mFilename = new QStringList();
    mRCDialog = new Ui_RemoteClients();
    mRCDialog->setupUi(mRCDialog);
    this->mMode = FileBrowser::OPEN;

    QObject::connect(this->mRCDialog->buttonBox, SIGNAL(accepted()), this, SLOT(remoteClientsAccept()));
    QObject::connect(this->mRCDialog->buttonBox, SIGNAL(rejected()), this, SLOT(remoteClientsReject()));
    QObject::connect(this->mRCDialog->listWidget, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(handleItemClicked(QListWidgetItem *)));
    emit dirChange(this->mLocationPath);
    // std::string tmp = this->mLocationPath.toStdString();
    //std::cerr << "FileBrowser: Path: " << tmp.c_str() << std::endl;
}

FileBrowser::FileBrowser(QWidget *parent)
{
    Q_UNUSED(parent);

    this->CreateDialogLayout();
    connect(lstDir, SIGNAL(itemDoubleClicked()), this, SLOT(onLstDirDoubleClick()));
    connect(lstFile, SIGNAL(itemDoubleClicked()), this, SLOT(onLstFileDoubleClick()));
    connect(btnCancel, SIGNAL(pressed()), this, SLOT(onCancelPressed()));
    connect(txtFilename, SIGNAL(textEdited(const QString &)), this, SLOT(onFilenameChanged()));
    connect(txtFilename, SIGNAL(returnPressed()), this, SLOT(onAcceptPressed()));
    connect(cmbFilter, SIGNAL(activated(int)), this, SLOT(onFilterActivated(int)));
    connect(btnHome, SIGNAL(pressed()), this, SLOT(onHomePressed()));
    connect(btnMachine, SIGNAL(pressed()), this, SLOT(onMachinePressed()));
    connect(btnRemote, SIGNAL(pressed()), this, SLOT(onRemotePressed()));
    connect(btnAccessGrid, SIGNAL(pressed()), this, SLOT(onAccessGridPressed()));
    connect(cmbDirectory, SIGNAL(activated(int)), this, SLOT(onDirActivated(int)));
    connect(btnDirUp, SIGNAL(pressed()), this, SLOT(onDirUpPressed()));
    connect(cmbHistory, SIGNAL(activated(int)), this, SLOT(onHistoryActivated(int)));
    connect(btnAccept, SIGNAL(pressed()), this, SLOT(onAcceptPressed()));
    connect(this, SIGNAL(requestLists(QString, QString)), this->mTUIElement, SLOT(handleRequestLists(QString, QString)));
    connect(this, SIGNAL(filterChange(QString)), this->mTUIElement, SLOT(handleFilterUpdate(QString)));
    connect(this, SIGNAL(dirChange(QString)), this->mTUIElement, SLOT(handleDirChange(QString)));
    connect(this, SIGNAL(requestClients()), this->mTUIElement, SLOT(handleClientRequest()));
    connect(this, SIGNAL(locationChanged(QString)), this->mTUIElement, SLOT(handleLocationChange(QString)));
    connect(this, SIGNAL(requestLocalHome()), this->mTUIElement, SLOT(handleLocalHome()));
    connect(this, SIGNAL(reqDriveList()), this->mTUIElement, SLOT(handleReqDriveList()));
    connect(this, SIGNAL(fileSelected(QString, QString, bool)), this->mTUIElement, SLOT(sendSelectedFile(QString, QString, bool)));

    this->mLocationPath = "";
    this->mFilename = new QStringList();
    mRCDialog = new Ui_RemoteClients();
    mRCDialog->setupUi(mRCDialog);
    this->mMode = FileBrowser::OPEN;
}

FileBrowser::~FileBrowser(void)
{
    delete mFilename;
}

QStringList *FileBrowser::Filename()
{
    return NULL;
}

opencover::IData *FileBrowser::DataObject()
{
    return NULL;
}

void FileBrowser::DataObject(opencover::IData *value)
{
    Q_UNUSED(value);
}

QString FileBrowser::LocationPath()
{
    return NULL;
}

void FileBrowser::LocationPath(QString value)
{
    Q_UNUSED(value);
}

QStringList *FileBrowser::FilterList()
{
    return NULL;
}

void FileBrowser::FilterList(QStringList *value)
{
    Q_UNUSED(value);
}

void FileBrowser::CreateDialogLayout()
{
    QGridLayout *gridLayout;
    QGridLayout *gridLayout1;
    QGridLayout *gridLayout2;
    QSplitter *splitter;

    this->setObjectName(QString::fromUtf8("Dialog"));
    this->setWindowModality(Qt::ApplicationModal);
    QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(0), static_cast<QSizePolicy::Policy>(0));
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
    //sizePolicy.setHeightForWidth(this->sizePolicy().hasHeightForWidth());
    this->setSizePolicy(sizePolicy);
    gridLayout = new QGridLayout(this);
    gridLayout->setSizeConstraint(QLayout::SetNoConstraint);
    gridLayout->setSpacing(6);
    gridLayout->setMargin(9);
    gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
    gridLayout1 = new QGridLayout();
    gridLayout1->setSizeConstraint(QLayout::SetNoConstraint);
    gridLayout1->setSpacing(6);
    gridLayout1->setMargin(0);
    gridLayout1->setObjectName(QString::fromUtf8("gridLayout1"));
    gridLayout2 = new QGridLayout();
    gridLayout2->setSizeConstraint(QLayout::SetNoConstraint);
    gridLayout2->setSpacing(6);
    gridLayout2->setMargin(0);
    gridLayout2->setObjectName(QString::fromUtf8("gridLayout2"));
    splitter = new QSplitter(this);
    splitter->setObjectName(QString::fromUtf8("splitter"));
    splitter->setOrientation(Qt::Horizontal);
    lstDir = new QListWidget(splitter);
    lstDir->setObjectName(QString::fromUtf8("lstDir"));
    splitter->addWidget(lstDir);

    lstFile = new QListWidget(splitter);
    lstFile->setObjectName(QString::fromUtf8("lstFile"));
    splitter->addWidget(lstFile);

    gridLayout2->addWidget(splitter, 0, 0, 1, 2);

    btnCancel = new QPushButton(this);
    btnCancel->setObjectName(QString::fromUtf8("btnCancel"));
    //btnCancel->setIcon(QIcon(QString::fromUtf8("../../../../common/src/sys/QtMapeditor/XPM/16x16/stop.xpm")));
    btnCancel->setIcon(QPixmap(stop));

    gridLayout2->addWidget(btnCancel, 2, 1, 1, 1);

    chkLoadAllClients = new QCheckBox(this);
    chkLoadAllClients->setObjectName(QString::fromUtf8("chkLoadAllClients"));

    txtFilename = new QLineEdit(this);
    txtFilename->setObjectName(QString::fromUtf8("txtFilename"));

    gridLayout2->addWidget(txtFilename, 1, 0, 1, 1);

    cmbFilter = new QComboBox(this);
    cmbFilter->setObjectName(QString::fromUtf8("cmbFilter"));

    gridLayout2->addWidget(chkLoadAllClients, 3, 0, 1, 2);
    gridLayout2->addWidget(cmbFilter, 2, 0, 1, 1);

    btnAccept = new QPushButton(this);
    btnAccept->setObjectName(QString::fromUtf8("btnAccept"));
    //btnAccept->setIcon(QIcon(QString::fromUtf8("../../../../common/src/sys/QtMapeditor/XPM/16x16/apply.xpm")));
    btnAccept->setIcon(QPixmap(apply));

    gridLayout2->addWidget(btnAccept, 1, 1, 1, 1);

    gridLayout1->addLayout(gridLayout2, 1, 1, 1, 1);

    vboxLayout = new QVBoxLayout();

    vboxLayout->setSizeConstraint(QLayout::SetNoConstraint);
    vboxLayout->setSpacing(6);
    vboxLayout->setMargin(0);
    vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
    btnHome = new QPushButton(this);
    btnHome->setObjectName(QString::fromUtf8("btnHome"));
    //btnHome->setIcon(QIcon(QString::fromUtf8("../../../../common/src/sys/QtMapeditor/XPM/32x32/home32.xpm")));
    btnHome->setIcon(QPixmap(home32));
    btnHome->setIconSize(QSize(32, 32));

    vboxLayout->addWidget(btnHome);

    btnMachine = new QPushButton(this);
    btnMachine->setObjectName(QString::fromUtf8("btnMachine"));
    //btnMachine->setIcon(QIcon(QString::fromUtf8("../../../../common/src/sys/QtMapeditor/XPM/32x32/computer32.xpm")));
    btnMachine->setIcon(QPixmap(computer32));
    btnMachine->setIconSize(QSize(32, 32));

    vboxLayout->addWidget(btnMachine);

    btnAccessGrid = new QPushButton(this);
    btnAccessGrid->setObjectName(QString::fromUtf8("btnAccessGrid"));
    //btnAccessGrid->setIcon(QIcon(QString::fromUtf8("../../../../common/src/sys/QtMapeditor/XPM/32x32/ag32.xpm")));
    btnAccessGrid->setIcon(QPixmap(ag32));
    btnAccessGrid->setIconSize(QSize(32, 32));

    vboxLayout->addWidget(btnAccessGrid);

    btnRemote = new QPushButton(this);
    btnRemote->setObjectName(QString::fromUtf8("btnRemote"));
    //btnRemote->setIcon(QIcon(QString::fromUtf8("../../../../common/src/sys/QtMapeditor/XPM/32x32/network32.xpm")));
    btnRemote->setIcon(QPixmap(network32));
    btnRemote->setIconSize(QSize(32, 32));

    vboxLayout->addWidget(btnRemote);

    gridLayout1->addLayout(vboxLayout, 1, 0, 1, 1);

    hboxLayout = new QHBoxLayout();
    hboxLayout->setSizeConstraint(QLayout::SetNoConstraint);
    hboxLayout->setSpacing(6);
    hboxLayout->setMargin(0);
    hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
    cmbDirectory = new QComboBox(this);
    cmbDirectory->setObjectName(QString::fromUtf8("cmbDirectory"));
    cmbDirectory->setMinimumSize(QSize(0, 30));

    hboxLayout->addWidget(cmbDirectory);

    btnDirUp = new QPushButton(this);
    btnDirUp->setObjectName(QString::fromUtf8("btnDirUp"));
    btnDirUp->setMinimumSize(QSize(35, 35));
    btnDirUp->setMaximumSize(QSize(40, 40));
    //btnDirUp->setIcon(QIcon(QString::fromUtf8("../../../../common/src/sys/QtMapeditor/XPM/32x32/undo32.xpm")));
    btnDirUp->setIcon(QPixmap(undo32));

    hboxLayout->addWidget(btnDirUp);

    cmbHistory = new QComboBox(this);
    cmbHistory->setObjectName(QString::fromUtf8("cmbHistory"));
    cmbHistory->setMinimumSize(QSize(0, 30));

    hboxLayout->addWidget(cmbHistory);

    gridLayout1->addLayout(hboxLayout, 0, 0, 1, 2);

    gridLayout->addLayout(gridLayout1, 0, 0, 1, 1);

    this->setWindowTitle(QApplication::translate("Dialog", "Dialog", 0));
    btnCancel->setText(QApplication::translate("Dialog", "Cancel", 0));
    btnAccept->setText(QApplication::translate("Dialog", "Accept", 0));
    chkLoadAllClients->setText(QString::fromUtf8("Load for all clients"));
    chkLoadAllClients->setDisabled(true);
    btnHome->setText(QString());
    btnMachine->setText(QString());
    btnAccessGrid->setText(QString());
    btnRemote->setText(QString());
    btnDirUp->setText(QString());

    QSize size(563, 294);
    size = size.expandedTo(this->minimumSizeHint());
    this->resize(size);
}

void FileBrowser::onCancelPressed()
{
    this->hide();
}

void FileBrowser::onAcceptPressed()
{
    QStringList *list = new QStringList(this->txtFilename->text());
    int ret = 0;

    if (exists(list->at(0)) && (this->mMode == FileBrowser::SAVE))
    {
        ret = QMessageBox::question(this, tr("Remote File Browser"),
                                    tr("The file already exists!\n Do you want to overwrite it?"),
                                    QMessageBox::Save | QMessageBox::Cancel,
                                    QMessageBox::Save);
    }

    this->mFilename = list;
    //emit fileload event here
    QString file = mFilename->at(0);

    if (!(ret == QMessageBox::Cancel) && (list->at(0) != ""))
    {

        emit fileSelected(file, this->mLocationPath, this->chkLoadAllClients->isChecked());
        this->hide();
    }
    else
    {
        /**
       * Take the local stored path location and return it as selected path
       *
       */
        emit pathSelected(file, mLocationPath);
        this->hide();
    }
}

void FileBrowser::onHomePressed()
{
    //show directories and files from home dir
    emit requestLocalHome();
}

void FileBrowser::onMachinePressed()
{
    //load directory and files list of machines root dir
    //emit requestLocalRootDir()
    emit reqDriveList();
}

void FileBrowser::onAccessGridPressed()
{
    //select files from AG dta store
    this->mLocation = "AccessGrid";
    emit locationChanged(this->mLocation);
    emit dirChange(QString(""));
}

void FileBrowser::onRemotePressed()
{
    // select remote machine to browse for files
    this->mRCDialog->raise();
    this->mRCDialog->show();
    emit requestClients();
}

void FileBrowser::onDirUpPressed()
{
    // go up one level of directories and request new lists
    // from Data class
    QDir locDir(this->mLocationPath);
    if (this->mLocation == "AccessGrid")
    {
        QStringList parts = this->mLocationPath.split("/");
        parts.removeLast();
        this->mLocationPath = parts.join("/");
    }
    else if (locDir.cdUp())
    {
        this->mLocationPath = locDir.path();
    }

    //std::string temp = this->mLocationPath.toStdString();
    //std::cerr << "FileBrowser::onDirUpPressed " <<  temp.c_str() << std::endl;
    emit dirChange(this->mLocationPath);
}

void FileBrowser::onDirActivated(int index)
{
    // request new file and directory list for recently new set
    // directory
    QString item = this->cmbDirectory->itemText(index);
    if (index == 0)
    {
        emit reqDriveList();
    }
    else
    {
        QString tempNewDir;

        this->cmbHistory->addItem(this->mLocationPath);

        QStringList pathParts = this->mLocationPath.split("/");
        QStringList newDirParts;

        int counter = 0;
        do
        {
            //std::cerr << "Path Part #" << counter << " = " << pathParts[counter].toStdString().c_str() << std::endl;
            newDirParts.append(pathParts[counter]);
        } while ((item != pathParts[counter]) && (++counter < pathParts.size()));

        this->mLocationPath = newDirParts.join("/");
        //std::cerr << "New path: " << this->mLocationPath.toStdString().c_str() << std::endl;

        //std::string temp = this->mLocationPath.toStdString();
        //std::cerr << "FileBrowser::onDirActivated " <<  temp.c_str() << std::endl;
        emit dirChange(this->mLocationPath);
    }
}

void FileBrowser::onHistoryActivated(int index)
{
    QString item = this->cmbHistory->itemText(index);
    this->cmbHistory->addItem(this->mLocationPath);
    this->mLocationPath = item;

    //std::string temp = this->mLocationPath.toStdString();
    //std::cerr << "FileBrowser::onHistoryActivated " <<  temp.c_str() << std::endl;
    emit dirChange(this->mLocationPath);
}

void FileBrowser::onFilenameChanged(QString text)
{
    this->txtFilename->setText(text);
}

void FileBrowser::onFilterActivated(int index)
{
    //QString filter = this->cmbFilter->itemText(index);
    QString filter = ((this->cmbFilter->itemData(index)).toString());

    //request new fileList from Data class for current Directory
    //based on selected filter

    emit filterChange(filter);
    emit requestLists(filter, this->mLocationPath);
}

void FileBrowser::onLstDirDoubleClick(QListWidgetItem *item)
{
    QString newDir = item->text();
    // Request Directory data from Data class for given Directory
    this->cmbHistory->addItem(this->mLocationPath);
    if (this->mLocationPath != "")
    {
        this->mLocationPath.append("/");
    }
    this->mLocationPath.append(newDir);

    //std::string temp = this->mLocationPath.toStdString();
    //std::cerr << "FileBrowser::onLstDirDoubleClick " <<  temp.c_str() << std::endl;
    emit dirChange(this->mLocationPath);
}

void FileBrowser::onLstFileDoubleClick(QListWidgetItem *item)
{
    this->txtFilename->setText(item->text());
    this->onAcceptPressed();
}

void FileBrowser::onLstFileClick(QListWidgetItem *item)
{
    this->txtFilename->setText(item->text());
}

void FileBrowser::handleDirUpdate(QStringList list)
{
    this->lstDir->clear();
    this->lstDir->reset();
    this->lstDir->addItems(list);
    QString temp;
    for (int i = 0; i < list.size(); i++)
    {
        temp += list.at(i);
        temp += "\n";
    }
    //std::cerr << temp.toStdString().c_str() << std::endl;
}

void FileBrowser::handleFileUpdate(QStringList list)
{
    this->lstFile->clear();
    this->lstFile->reset();
    this->lstFile->addItems(list);
}

void FileBrowser::handleCurDirUpdate(QString curDir)
{
    //std::string tmp = curDir.toStdString();
    //std::cerr <<"Reassigning current directory ------> " << tmp.c_str() << std::endl;
    this->mLocationPath = curDir;
    this->cmbDirectory->clear();

    QDir locDir(curDir);
    locDir.setPath(locDir.toNativeSeparators(curDir));
    //Now seperate path into its components
    QStringList dirParts = locDir.absolutePath().split("/");

    //std::cerr << "Hostname: " << mLocation.toStdString().c_str() << std::endl;

    this->cmbDirectory->addItem(this->mLocation);

    int i = 0;
    for (i = 0; i < dirParts.size();)
    {
        QString entry = dirParts.at(i++);
        if (entry != "")
        {
            cmbDirectory->addItem(entry);
        }
    }

    //Set to last and current directory entry
    emit cmbDirectory->setCurrentIndex(cmbDirectory->count() - 1);
}

void FileBrowser::setMode(DialogModes mode)
{
    this->mMode = mode;
}

void FileBrowser::getMode(DialogModes *mode)
{
    *mode = this->mMode;
}

bool FileBrowser::exists(QString file)
{
    QString fileEntry;

    for (int i = 0; i < lstFile->count(); i++)
    {
        fileEntry = lstFile->item(i)->text();
        if (fileEntry == file)
        {
            return true;
        }
    }
    return false;
}

void FileBrowser::handleClientUpdate(QStringList list)
{
    this->mRCDialog->setClients(list);
}

void FileBrowser::remoteClientsAccept()
{
    this->mLocation = this->mRCDialog->mSelectedClient;
    this->mRCDialog->hide();
    QStringList temp = mLocation.split(" ");
    this->mLocation = temp.at(0);

    emit locationChanged(this->mLocation);
    emit dirChange(".");
}

void FileBrowser::remoteClientsReject()
{
    //Do nothing at all
    this->mRCDialog->hide();
}

void FileBrowser::handleItemClicked(QListWidgetItem *item)
{
    this->mRCDialog->mSelectedClient = item->text();
};

void FileBrowser::handleDriveUpdate(QStringList list)
{
    this->lstDir->clear();
    this->lstFile->clear();
    this->lstDir->reset();
    this->lstFile->reset();
    this->lstDir->addItems(list);
    this->mLocationPath = "";
    QString temp;
    for (int i = 0; i < list.size(); i++)
    {
        temp += list.at(i);
        temp += "\n";
    }
}

void FileBrowser::handleUpdateMode(int mode)
{
    if (mode == 1)
    {
        this->mMode = FileBrowser::OPEN;
    }
    else
    {
        this->mMode = FileBrowser::SAVE;
    }
}

void FileBrowser::handleUpdateFilterList(char *filterList)
{
    QString qfilterList(filterList);
    QStringList filters = qfilterList.split(";");

    qfilterList = qfilterList.left(qfilterList.size() - 4);

    this->cmbFilter->clear();

    for (int i = 0; i < filters.size(); i++)
    {
        this->cmbFilter->addItem(filters.at(i), QStringList(filters.at(i)));
    }
    this->cmbFilter->addItem(QString("All supported formats"), qfilterList);
    this->cmbFilter->setCurrentIndex(cmbFilter->count() - 1);
    this->mFilterList.append(this->cmbFilter->itemText(0));

    QString filter = (cmbFilter->itemData(cmbFilter->currentIndex())).toString();
    emit filterChange(filter);
}

void FileBrowser::handleLocationUpdate(QString strEntry)
{
    //std::string sloc = strEntry.toStdString();
    //std::cerr << "Location reset to : " << sloc.c_str() << std::endl;

    this->mLocation = strEntry;
    handleCurDirUpdate(mLocationPath);
}

void FileBrowser::handleupdateRemoteButtonState(int state)
{
    if (state)
    {
        this->btnRemote->setEnabled(true);
    }
    else
    {
        this->btnRemote->setEnabled(false);
    }
}

void FileBrowser::handleUpdateLoadCheckBox(bool isMaster)
{
    this->chkLoadAllClients->setDisabled(!isMaster);
}
