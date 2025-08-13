/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "clientWidget.h"
#include "mainWindow.h"
#include "ui_mainWindow.h"

#include <util/coSpawnProgram.h>
#include <qtutil/NonBlockingDialogue.h>

#include <QCloseEvent>
#include <QComboBox>
#include <QMenu>
#include <QMessageBox>
#include <QScrollBar>
#include <QShortcut>
#include <QSystemTrayIcon>
#include <QTextBrowser>
#include <QTextStream>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>
#include <QToolButton>

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <csignal>
#include <cassert>

#ifdef _WIN32

#else
#include <unistd.h>
#endif

#include <config/coConfigGroup.h>
#include <config/coConfig.h>
#include <net/covise_host.h>
#include <demo.h>

using namespace vrb;

const std::string CoviseDaemonSection = "CoviseDaemon";

MainWindow::MainWindow(const vrb::VrbCredentials &credentials, QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
	, m_configFile(m_access.file("coviseDaemon"))
	, cfgTimeout(m_configFile->value<int64_t>(CoviseDaemonSection, "Timeout", 10))
	, cfgAutostart(m_configFile->value(CoviseDaemonSection, "Autostart", false))
	, cfgAutoConnect(m_configFile->value(CoviseDaemonSection, "AutoConnect", false))
	, cfgBackground(m_configFile->value(CoviseDaemonSection, "Background", false))
	, cfgMinimized(m_configFile->value(CoviseDaemonSection, "Minimized", false))
	, cfgArguments(m_configFile->value<std::string>(CoviseDaemonSection, "Arguments", ""))
	, cfgOutputMode(m_configFile->value<std::string>(CoviseDaemonSection, "OutputMode", "Terminal"))
	, cfgOutputModeFile(m_configFile->value<std::string>(CoviseDaemonSection, "OutputModeFile", ""))
{
	qRegisterMetaType<covise::Program>();
	qRegisterMetaType<std::vector<std::string>>();
	m_configFile->setSaveOnExit(true);
	initUi(credentials);
	initConfigSettings();
	setRemoteLauncherCallbacks();
	connect(this, &MainWindow::updateStatusBarSignal, this, &MainWindow::updateStatusBar);
	initClientList();
	setHotkeys();
	handleAutoconnect();
	setStartupWindowStyle();
}

MainWindow::~MainWindow()
{
	m_configFile->save();
	m_progressBarTimer.stop();
	m_isConnecting = false;
	delete ui;
}

// private slots
//-----------------------------------------------------------------------
void MainWindow::on_actionSideMenuAction_triggered()
{
	Guard g(m_mutex);
	int curr = ui->MenuStackedWidget->currentIndex();
	curr == 0 ? curr = 1 : curr = 0;
	setStackedWidget(ui->MenuStackedWidget, curr);
}

void MainWindow::on_timeoutSlider_sliderMoved(int val)
{
	*cfgTimeout = val;
	Guard g(m_mutex);
	ui->timeoutLabel->setText(QString("timeout: ") + QString::number(val) + QString("s"));
}

void MainWindow::on_autostartCheckBox_clicked()
{
	*cfgAutostart = ui->autostartCheckBox->isChecked();
}
void MainWindow::on_autoconnectCheckBox_clicked()
{
	*cfgAutoConnect = ui->autoconnectCheckBox->isChecked();
}
void MainWindow::on_backgroundCheckBox_clicked()
{
	*cfgBackground = ui->backgroundCheckBox->isChecked();
}
void MainWindow::on_minimizedCheckBox_clicked()
{
	*cfgMinimized = ui->minimizedCheckBox->isChecked();
}
void MainWindow::on_cmdArgsInput_textChanged()
{
	*cfgArguments = ui->cmdArgsInput->text().toStdString();
}

void MainWindow::onConnectBtnClicked()
{
	assert(!m_isConnecting);
	m_isConnecting = true;
	setStateConnecting();
	m_remoteLauncher.connect(vrb::VrbCredentials{ui->hostIpf->text().toStdString(),
												 static_cast<unsigned int>(ui->tcpInput->value()),
												 static_cast<unsigned int>(ui->udpInput->value())});
}

void MainWindow::onCancelBtnClicked()
{
	assert(m_isConnecting);
	m_isConnecting = false;
	setStateDisconnected();
}

void MainWindow::onDisconnectBtnClicked()
{
	assert(!m_isConnecting);
	setStateDisconnected();
}

void MainWindow::updateStatusBar()
{
	if (m_isConnecting)
	{
		auto newVal = ui->progressBar->value() + 1;
		if (ui->progressBar->maximum() > 0 && newVal >= ui->progressBar->maximum()) // timeout
		{
			setStateDisconnected();
			m_progressBarTimer.stop();
		}
		else
		{
			ui->progressBar->setValue(newVal);
		}
	}
	else
		m_progressBarTimer.stop();
}

void MainWindow::setStateDisconnected()
{
	m_remoteLauncher.disconnect();
	ui->progressBar->setVisible(false);
	m_isConnecting = false;
	ui->connectBtn->setText("Connect");
	ui->connectBtn->disconnect();
	ui->conncetedLabel->setStyleSheet(QString("image: url(:/images/redCircle.png);"));
	connect(ui->connectBtn, &QPushButton::clicked, this, &MainWindow::onConnectBtnClicked);
	setStackedWidget(ui->InfoStackedWidget, 1);
}

void MainWindow::setStateConnecting()
{
	showConnectionProgressBar(ui->timeoutSlider->value());
	ui->connectBtn->setText("Cancel");
	ui->connectBtn->disconnect();
	connect(ui->connectBtn, &QPushButton::clicked, this, &MainWindow::onCancelBtnClicked);
}

void MainWindow::setStateConnected()
{
	std::cerr << "Connected!" << std::endl;
	ui->progressBar->setVisible(false);
	m_isConnecting = false;
	ui->connectBtn->setText("disconnect");
	ui->connectBtn->disconnect();
	ui->conncetedLabel->setStyleSheet(QString("image: url(:/images/greenCircle.png);"));
	connect(ui->connectBtn, &QPushButton::clicked, this, &MainWindow::onDisconnectBtnClicked);

	setStackedWidget(ui->InfoStackedWidget, 0);
}

void MainWindow::updateClient(int clientID, QString clientInfo)
{
	m_clientList->addClient(clientID, clientInfo);
}

void MainWindow::removeClient(int clientID)
{
	m_clientList->removeClient(clientID);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
	if (ui->backgroundCheckBox->isChecked())
	{
		hideThis();
		event->ignore();
	}
	else
		event->accept();
}

// private functions
//----------------------------------------------------------------------------------

void MainWindow::initUi(const vrb::VrbCredentials &credentials)
{
	ui->setupUi(this);
	ui->tcpInput->setValue(credentials.tcpPort());
	ui->udpInput->setValue(credentials.udpPort());
	ui->hostIpf->setText(QString(credentials.ipAddress().c_str()));
	ui->progressBar->setVisible(false);
	connect(ui->exitBtn, &QPushButton::clicked, QApplication::quit);
	connect(&m_progressBarTimer, &QTimer::timeout, this, &MainWindow::updateStatusBar);
	setStackedWidget(ui->InfoStackedWidget, 1);
	setupDemoClientAction();
	initOutputModes();
}

void MainWindow::initOutputModes()
{
	QString dir = "/var/tmp/";
	dir += getenv("USER");
	dir += "/";
	ui->outputFile->setText(dir);
	ui->outputFile->hide();
	ui->outputFileLabel->hide();
	connect(ui->OutputModesCB, &QComboBox::currentTextChanged, this, [this](const QString &currentText)
			{
				if (currentText == "File")
				{
					ui->outputFile->show();
					ui->outputFileLabel->show();
				}
				else
				{
					ui->outputFile->hide();
					ui->outputFileLabel->hide();
				}
				*cfgOutputMode = currentText.toStdString();
				reconnectOutPut(); });
	connect(ui->outputFile, &QLineEdit::textEdited, this, [this](const QString &s)
			{ *cfgOutputModeFile = s.toStdString(); });
}
#include <functional>
void MainWindow::initConfigSettings()
{
	ui->timeoutSlider->setValue(cfgTimeout->value());
	on_timeoutSlider_sliderMoved(cfgTimeout->value());
	ui->autostartCheckBox->setChecked(cfgAutostart->value());
	ui->autoconnectCheckBox->setChecked(cfgAutoConnect->value());
	ui->backgroundCheckBox->setChecked(cfgBackground->value());
	ui->minimizedCheckBox->setChecked(cfgMinimized->value());
	ui->cmdArgsInput->setText(std::string(cfgArguments->value()).c_str());
	ui->OutputModesCB->setCurrentText(std::string(cfgOutputMode->value()).c_str());
	ui->outputFile->setText(std::string(cfgOutputModeFile->value()).c_str());
}

void MainWindow::setRemoteLauncherCallbacks()
{
	connect(&m_remoteLauncher, &CoviseDaemon::connectedSignal, this, &MainWindow::setStateConnected);
	connect(&m_remoteLauncher, &CoviseDaemon::disconnectedSignal, this, [this]()
			{
				m_clientList->clear();
				setStateDisconnected();
				if (ui->autoconnectCheckBox->isChecked())
					onConnectBtnClicked(); });
	connect(&m_remoteLauncher, &CoviseDaemon::updateClient, this, &MainWindow::updateClient);
	connect(&m_remoteLauncher, &CoviseDaemon::removeClient, this, &MainWindow::removeClient);
	reconnectOutPut();
	connect(&m_remoteLauncher, &CoviseDaemon::askForPermission, this, [this](covise::Program p, int clientId, const QString &description)
			{ askForPermission(p, clientId, description); });
	connect(&m_remoteLauncher, &CoviseDaemon::askForPermissionAbort, this, &MainWindow::removePermissionRequest);
}

void MainWindow::reconnectOutPut()
{

	static QMetaObject::Connection outPutConn;
	static QMetaObject::Connection terminateConn;
	if (outPutConn)
		disconnect(outPutConn);
	if (ui->OutputModesCB->currentText() == "Gui")
	{
		outPutConn = connect(&m_remoteLauncher, &CoviseDaemon::childProgramOutput, this, [this](const QString &childId, const QString &txt)
							 {
					auto childOutput = std::find(m_childOutputs.begin(), m_childOutputs.end(), childId);
					if (childOutput == m_childOutputs.end())
					{
						childOutput = m_childOutputs.emplace(m_childOutputs.end(), childId, ui->childTabs);
					}
					childOutput->addText(txt); });
		terminateConn = connect(&m_remoteLauncher, &CoviseDaemon::childTerminated, this, [this](const QString &childId)
								{ m_childOutputs.erase(std::remove(m_childOutputs.begin(), m_childOutputs.end(), childId), m_childOutputs.end()); });
	}
	else if (ui->OutputModesCB->currentText() == "File")
	{
		outPutConn = connect(&m_remoteLauncher, &CoviseDaemon::childProgramOutput, this, [this](const QString &childId, const QString &txt)
							 {
					auto childOutput = std::find_if(m_childOutputFiles.begin(), m_childOutputFiles.end(), [childId](const std::pair<QString, std::unique_ptr<std::fstream>> &file)
													{ return file.first == childId; });
					if (childOutput == m_childOutputFiles.end())
					{
						std::string dir = ui->outputFile->text().toStdString() + childId.toStdString() + ".log";
						childOutput = m_childOutputFiles.emplace(m_childOutputFiles.end(), std::pair<QString, std::unique_ptr<std::fstream>>{childId, std::unique_ptr<std::fstream>(new std::fstream(dir, std::ios_base::out))});
					}
					(*childOutput->second) << txt.toStdString(); });
		terminateConn = connect(&m_remoteLauncher, &CoviseDaemon::childTerminated, this, [this](const QString &childId)
								{ m_childOutputFiles.erase(std::remove_if(m_childOutputFiles.begin(), m_childOutputFiles.end(), [childId](const std::pair<QString, std::unique_ptr<std::fstream>> &file)
																		  { return file.first == childId; }),
														   m_childOutputFiles.end()); });
	}
	else if (ui->OutputModesCB->currentText() == "Cmd")
	{
		outPutConn = connect(&m_remoteLauncher, &CoviseDaemon::childProgramOutput, this, [this](const QString &childId, const QString &txt)
							 { std::cerr << txt.toStdString(); });
	}
}

void MainWindow::initClientList()
{
	m_clientList = new ClientWidgetList(ui->clientsScrollArea, ui->clientsScrollArea);
	connect(m_clientList, &ClientWidgetList::requestProgramLaunch, this, [this](covise::Program programID, int clientID)
			{
				std::cerr << "launching " << covise::programNames[programID] << " on client " << clientID << std::endl;

				m_remoteLauncher.sendLaunchRequest(programID, clientID, parseCmdArgsInput()); });
}

void MainWindow::setHotkeys()
{
	auto escape = new QShortcut(QKeySequence(Qt::Key_Escape), this);
	auto enter = new QShortcut(QKeySequence(Qt::Key_Return), this);
	auto sideMenu = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Tab), this);
	connect(enter, &QShortcut::activated, this, [this]()
			{
				if (!m_isConnecting)
				{
					onConnectBtnClicked();
				} });
	connect(escape, &QShortcut::activated, this, [this]()
			{
				if (m_isConnecting)
				{
					onCancelBtnClicked();
				}
				else
				{
					onDisconnectBtnClicked();
				} });
	connect(sideMenu, &QShortcut::activated, this, [this]()
			{ on_actionSideMenuAction_triggered(); });
}

void MainWindow::handleAutoconnect()
{
	if (ui->autoconnectCheckBox->isChecked())
	{
		onConnectBtnClicked();
	}
	else
	{
		setStateDisconnected();
	}
}

void MainWindow::setStartupWindowStyle()
{
	if (!QSystemTrayIcon::isSystemTrayAvailable())
	{
		ui->backgroundCheckBox->setChecked(false);
		ui->backgroundCheckBox->hide();
	}
	if (ui->backgroundCheckBox->isChecked())
	{
		if (!m_tray)
			createTrayIcon();
		m_tray->show();
	}
	if (ui->minimizedCheckBox->isChecked())
	{
		if (!ui->backgroundCheckBox->isChecked())
		{
			showMinimized();
		}
		else
		{
			hideThis();
		}
	}
	else
	{
		show();
	}
}

void MainWindow::showThis()
{
	if (!ui->backgroundCheckBox->isChecked() && m_tray)
	{
		m_tray->hide();
	}
	show();
}

void MainWindow::hideThis()
{
	bool created = false;
	if (!m_tray)
	{
		createTrayIcon();
		created = true;
	}
	m_tray->show();
	if (created)
		m_tray->showMessage("COVISE Daemon", " running in background");
	hide();
}

void MainWindow::createTrayIcon()
{
	m_tray = new QSystemTrayIcon(QIcon(":/images/coviseDaemon.png"), this);
	// tray->showMessage("coviseDaemon", "blablabla");
	m_tray->setToolTip("COVISE Daemon");
	auto trayMenu = new QMenu(this);
	trayMenu->setContextMenuPolicy(Qt::CustomContextMenu);
	auto exit = trayMenu->addAction("Exit");
	connect(exit, &QAction::triggered, &QApplication::quit);
	trayMenu->addAction("Open", this, &MainWindow::showThis);
	m_tray->setContextMenu(trayMenu);
	connect(m_tray, &QSystemTrayIcon::activated, this, [this](QSystemTrayIcon::ActivationReason reason)
			{
				switch (reason)
				{
				case QSystemTrayIcon::DoubleClick:
				case QSystemTrayIcon::Trigger:
                                     if (isVisible())
					hideThis();
                                     else
					showThis();
					break;
				default:
					break;
				} });
}

void MainWindow::showConnectionProgressBar(int seconds)
{
	int resolution = 2;
	ui->progressBar->reset();
	ui->progressBar->setRange(0, resolution * seconds);
	ui->progressBar->setVisible(true);
	if (!m_progressBarTimer.isActive())
	{
		m_progressBarTimer.start(1000 / resolution);
	}
}

void MainWindow::askForPermission(covise::Program p, int clientID, const QString &description)
{
	if (ui->autostartCheckBox->isChecked())
		m_remoteLauncher.answerPermissionRequest(p, clientID, true);
	else
	{
		auto pr = m_permissionRequests.emplace(m_permissionRequests.end(), new PermissionRequest{p, clientID, description, this});
		connect(pr->get(), &PermissionRequest::permit, this, [p, clientID, this](int answer)
				{ m_remoteLauncher.answerPermissionRequest(p, clientID, answer); });
		connect(pr->get(), &PermissionRequest::permit, this, [p, clientID, this](int answer)
				{ removePermissionRequest(p, clientID); });
		pr->get()->show();
	}
}

void MainWindow::removePermissionRequest(covise::Program p, int clientID)
{
	m_permissionRequests.erase(std::remove_if(m_permissionRequests.begin(), m_permissionRequests.end(), [p, clientID](const std::unique_ptr<PermissionRequest> &request)
											  { return request->program() == p && request->requestorId() == clientID; }),
							   m_permissionRequests.end());
}

std::vector<std::string> MainWindow::parseCmdArgsInput()
{
	return covise::parseCmdArgString(ui->cmdArgsInput->text().toStdString());
}

void MainWindow::setupDemoClientAction()
{
    // Connect the demo client action
    connect(ui->actionDemoClient, &QAction::triggered, this, &MainWindow::openDemoClient);
    
    // Set up custom context menu for right-click functionality
    // We need to get the actual toolbar button for the action
    QTimer::singleShot(0, this, [this]() {
        // Find the toolbar button for our action
        QList<QToolButton*> toolButtons = ui->toolBar->findChildren<QToolButton*>();
        for (QToolButton* button : toolButtons) {
            if (button->defaultAction() == ui->actionDemoClient) {
                button->setContextMenuPolicy(Qt::CustomContextMenu);
                connect(button, &QToolButton::customContextMenuRequested, 
                        this, &MainWindow::showDemoClientContextMenu);
                break;
            }
        }
    });
}

void MainWindow::openDemoClient()
{
    QString demoUrl = getDemoClientUrl();
    QDesktopServices::openUrl(QUrl(demoUrl));
}

void MainWindow::showDemoClientContextMenu()
{
    QMenu contextMenu(tr("Demo Client"), this);
    
    QAction *openAction = contextMenu.addAction("🌐 Open Demo Client");
    QAction *copyAction = contextMenu.addAction("📋 Copy Link to Clipboard");
    
    // Style the menu actions
    openAction->setIcon(QIcon(":/images/demo.png")); // if you have an icon
    
    connect(openAction, &QAction::triggered, this, &MainWindow::openDemoClient);
    connect(copyAction, &QAction::triggered, this, &MainWindow::copyDemoLinkToClipboard);
    
    // Show menu at the toolbar button position
    QPoint globalPos = QCursor::pos();
    contextMenu.exec(globalPos);
}

void MainWindow::copyDemoLinkToClipboard()
{
    QString demoUrl = getDemoClientUrl();
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(demoUrl);
    
    // Show confirmation in status bar
    ui->statusbar->showMessage("Demo client link copied to clipboard", 3000);
}

QString MainWindow::getDemoClientUrl()
{
    return QString("http://%1:%2").arg(covise::Host::getHostaddress().c_str()).arg(demo::port);
}

// free functions

void setStackedWidget(QStackedWidget *stack, int index)
{
	stack->currentWidget()->setEnabled(false);
	stack->currentWidget()->hide();
	stack->setCurrentIndex(index);
	stack->currentWidget()->setEnabled(true);
	stack->currentWidget()->show();
}
