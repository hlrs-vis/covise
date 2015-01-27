/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "frmMainWindow.h"
#include "frmAbout.h"

frmMainWindow::frmMainWindow(QApplication *app)
{
    mApplication = app;
    mIsOn = false;
    mDaemon = new SSLDaemon(this);
}

frmMainWindow::~frmMainWindow()
{
    if (mDaemon)
    {
        delete mDaemon;
    }
}

void frmMainWindow::setupUi()
{

    QMainWindow *frmMainWindow = this;

    if (frmMainWindow->objectName().isEmpty())
        frmMainWindow->setObjectName(QString::fromUtf8("frmMainWindow"));
    QSize size(398, 651);
    size = size.expandedTo(frmMainWindow->minimumSizeHint());
    frmMainWindow->resize(size);
    actionAbout = new QAction(frmMainWindow);
    actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
    actionExit = new QAction(frmMainWindow);
    actionExit->setObjectName(QString::fromUtf8("actionExit"));
    actionSave = new QAction(frmMainWindow);
    actionSave->setObjectName(QString::fromUtf8("actionSave"));
    actionLoad = new QAction(frmMainWindow);
    actionLoad->setObjectName(QString::fromUtf8("actionLoad"));
    actionRespawn = new QAction(frmMainWindow);
    actionRespawn->setObjectName(QString::fromUtf8("actionRespawn"));
    centralwidget = new QWidget(frmMainWindow);
    centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
    tabWidCovise = new QTabWidget(centralwidget);
    tabWidCovise->setObjectName(QString::fromUtf8("tabWidCovise"));
    tabWidCovise->setGeometry(QRect(10, 10, 381, 601));
    tabMonitor = new QWidget();
    tabMonitor->setObjectName(QString::fromUtf8("tabMonitor"));
    grpLog = new QGroupBox(tabMonitor);
    grpLog->setObjectName(QString::fromUtf8("grpLog"));
    grpLog->setGeometry(QRect(0, 0, 371, 571));
    lstLog = new QTextEdit(grpLog);
    lstLog->setObjectName(QString::fromUtf8("lstLog"));
    lstLog->setGeometry(QRect(10, 20, 351, 531));
    lstLog->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    tabWidCovise->addTab(tabMonitor, QString());
    tabConfig = new QWidget();
    tabConfig->setObjectName(QString::fromUtf8("tabConfig"));
    grpPartner = new QGroupBox(tabConfig);
    grpPartner->setObjectName(QString::fromUtf8("grpPartner"));
    grpPartner->setGeometry(QRect(10, 260, 351, 261));
    columnPartners = new QListWidget(grpPartner);
    columnPartners->setObjectName(QString::fromUtf8("columnPartners"));
    columnPartners->setGeometry(QRect(10, 20, 331, 231));
    grpHost = new QGroupBox(tabConfig);
    grpHost->setObjectName(QString::fromUtf8("grpHost"));
    grpHost->setGeometry(QRect(11, 11, 351, 241));
    lstHost = new QListWidget(grpHost);
    lstHost->setObjectName(QString::fromUtf8("lstHost"));
    lstHost->setGeometry(QRect(10, 20, 331, 211));
    grpCovise = new QGroupBox(tabConfig);
    grpCovise->setObjectName(QString::fromUtf8("grpCovise"));
    grpCovise->setGeometry(QRect(10, 520, 351, 51));
    lblPort = new QLabel(grpCovise);
    lblPort->setObjectName(QString::fromUtf8("lblPort"));
    lblPort->setGeometry(QRect(10, 20, 51, 16));
    lblPort->setAlignment(Qt::AlignCenter);
    txtPort = new QLineEdit(grpCovise);
    txtPort->setObjectName(QString::fromUtf8("txtPort"));
    txtPort->setGeometry(QRect(70, 20, 71, 20));
    btnOnOff = new QPushButton(grpCovise);
    btnOnOff->setObjectName(QString::fromUtf8("btnOnOff"));
    btnOnOff->setText(QString::fromUtf8("Daemon On"));
    btnOnOff->setGeometry(QRect(151, 20, 80, 20));
    tabWidCovise->addTab(tabConfig, QString());
    frmMainWindow->setCentralWidget(centralwidget);
    menubar = new QMenuBar(frmMainWindow);
    menubar->setObjectName(QString::fromUtf8("menubar"));
    menubar->setGeometry(QRect(0, 0, 398, 21));
    menuFile = new QMenu(menubar);
    menuFile->setObjectName(QString::fromUtf8("menuFile"));
    menuHelp = new QMenu(menubar);
    menuHelp->setObjectName(QString::fromUtf8("menuHelp"));
    frmMainWindow->setMenuBar(menubar);
    statusbar = new QStatusBar(frmMainWindow);
    statusbar->setObjectName(QString::fromUtf8("statusbar"));
    frmMainWindow->setStatusBar(statusbar);

    menubar->addAction(menuFile->menuAction());
    menubar->addAction(menuHelp->menuAction());
    menuFile->addAction(actionSave);
    menuFile->addAction(actionLoad);
    menuFile->addAction(actionRespawn);
    menuFile->addSeparator();
    menuFile->addAction(actionExit);
    menuHelp->addAction(actionAbout);

    retranslateUi();

    tabWidCovise->setCurrentIndex(0);

    QMetaObject::connectSlotsByName(frmMainWindow);

    connectSlots();

    mDaemon->setWindow(this);

} // setupUi

void frmMainWindow::retranslateUi()
{
    QMainWindow *frmMainWindow = this;
    frmMainWindow->setWindowTitle(QApplication::translate("frmMainWindow", "Covise Daemon", 0));
    actionAbout->setText(QApplication::translate("frmMainWindow", "About", 0));
    actionExit->setText(QApplication::translate("frmMainWindow", "Exit", 0));
    actionSave->setText(QApplication::translate("frmMainWindow", "Save", 0));
    actionLoad->setText(QApplication::translate("frmMainWindow", "Load", 0));
    actionRespawn->setText(QApplication::translate("frmMainWindow", "Respawn ", 0));
    grpLog->setTitle(QApplication::translate("frmMainWindow", "Log Messages", 0));
    tabWidCovise->setTabText(tabWidCovise->indexOf(tabMonitor), QApplication::translate("frmMainWindow", "Monitor", 0));
    grpPartner->setTitle(QApplication::translate("frmMainWindow", "Partner Security Settings", 0));
    columnPartners->setToolTip(QApplication::translate("frmMainWindow", "Shows a list of permanently allowed remote partners identified based on their SubjectUID and name.", 0));
    grpHost->setTitle(QApplication::translate("frmMainWindow", "Host Security Settings", 0));
    lstHost->setToolTip(QApplication::translate("frmMainWindow", "List of permanently allowed remote hosts that can launch applications", 0));
    grpCovise->setTitle(QApplication::translate("frmMainWindow", "Covise Daemon Settings", 0));
    lblPort->setText(QApplication::translate("frmMainWindow", "Port", 0));
    txtPort->setToolTip(QApplication::translate("frmMainWindow", "Enter the port number at which Covise Daemon should run. A change requires triggering of \"Respawn\"", 0));
    txtPort->setText(QApplication::translate("frmMainWindow", "31100", 0));
    tabWidCovise->setTabText(tabWidCovise->indexOf(tabConfig), QApplication::translate("frmMainWindow", "Config", 0));
    menuFile->setTitle(QApplication::translate("frmMainWindow", "File", 0));
    menuHelp->setTitle(QApplication::translate("frmMainWindow", "Help", 0));
} // retranslateUi

void frmMainWindow::connectSlots()
{
    connect(this->actionExit, SIGNAL(activated()), mApplication, SLOT(quit()));
    connect(this->actionRespawn, SIGNAL(activated()), this, SLOT(handleRespawn()));
    connect(this->actionAbout, SIGNAL(activated()), this, SLOT(handleAboutShow()));
    connect(this->btnOnOff, SIGNAL(clicked(bool)), this, SLOT(handleOnOff(bool)));
}

void frmMainWindow::handleAboutShow()
{
    frmAbout *about = new frmAbout(this->mApplication);
    about->setupUi(about);
    about->show();
}

void frmMainWindow::handleOnOff(bool)
{
    if (mIsOn)
    {
        btnOnOff->setText(QString::fromUtf8("Daemon On"));
        mIsOn = false;
        mDaemon->stop();
        mDaemon->closeServer();
    }
    else
    {
        btnOnOff->setText(QString::fromUtf8("Daemon Off"));
        mIsOn = true;
        mDaemon->openServer();
        mDaemon->run();
    }
}

void frmMainWindow::setPort(int port)
{
    QString qPort;
    qPort.setNum(port);
    std::string sPort = qPort.toStdString();
    txtPort->setText(QApplication::translate("frmMainWindow", sPort.c_str(), 0));
}

QTextEdit *frmMainWindow::getLog()
{
    return this->lstLog;
}

QListWidget *frmMainWindow::getHostList()
{
    return this->lstHost;
}

QListWidget *frmMainWindow::getUserList()
{
    return this->columnPartners;
}

void frmMainWindow::handleRespawn()
{
    mDaemon->stop();
    mDaemon->closeServer();
    std::cerr << "frmMainWindow::handleRespawn(): Daemon stopped!" << std::endl;
    mDaemon->openServer();
    mDaemon->run();
    std::cerr << "frmMainWindow::handleRespawn(): Daemon started!" << std::endl;
}
