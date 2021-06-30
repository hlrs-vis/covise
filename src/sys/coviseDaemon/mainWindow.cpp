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
#include <QMenu>
#include <QMessageBox>
#include <QScrollBar>
#include <QShortcut>
#include <QSystemTrayIcon>
#include <QTextBrowser>
#include <QTextStream>

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

using namespace vrb;

MainWindow::MainWindow(const vrb::VrbCredentials &credentials, QWidget *parent)
	: QMainWindow(parent), ui(new Ui::MainWindow), cfgTimeout("System.CoviseDaemon.Timeout"), cfgAutostart("System.CoviseDaemon.Autostart"), cfgAutoConnect("System.CoviseDaemon.AutoConnect"), cfgBackground("System.CoviseDaemon.Background"), cfgMinimized("System.CoviseDaemon.Minimized"), cfgArguments("System.CoviseDaemon.Arguments")
{
	qRegisterMetaType<vrb::Program>();
	qRegisterMetaType<std::vector<std::string>>();

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
	cdConfig->save();
	m_progressBarTimer.stop();
	m_isConnecting = false;
	delete ui;
}

//private slots
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
	cfgTimeout = val;
	Guard g(m_mutex);
	ui->timeoutLabel->setText(QString("timeout: ") + QString::number(val) + QString("s"));
}

void MainWindow::on_autostartCheckBox_clicked()
{
	cfgAutostart = ui->autostartCheckBox->isChecked();
}
void MainWindow::on_autoconnectCheckBox_clicked()
{
	cfgAutoConnect = ui->autoconnectCheckBox->isChecked();
}
void MainWindow::on_backgroundCheckBox_clicked()
{
	cfgBackground = ui->backgroundCheckBox->isChecked();
}
void MainWindow::on_minimizedCheckBox_clicked()
{
	cfgMinimized = ui->minimizedCheckBox->isChecked();
}
void MainWindow::on_cmdArgsInput_textChanged()
{
	cfgArguments = ui->cmdArgsInput->text();
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
		if (ui->progressBar->maximum() > 0 && newVal >= ui->progressBar->maximum()) //timeout
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

//private functions
//----------------------------------------------------------------------------------

void MainWindow::initUi(const vrb::VrbCredentials &credentials)
{
	ui->setupUi(this);
	ui->tcpInput->setValue(credentials.tcpPort);
	ui->udpInput->setValue(credentials.udpPort);
	ui->hostIpf->setText(QString(credentials.ipAddress.c_str()));
	ui->progressBar->setVisible(false);
	connect(ui->exitBtn, &QPushButton::clicked, QApplication::quit);
	connect(&m_progressBarTimer, &QTimer::timeout, this, &MainWindow::updateStatusBar);
	setStackedWidget(ui->InfoStackedWidget, 1);
	m_askForPermissionDiag = new covise::NonBlockingDialogue{this};
	m_askForPermissionDiag->setWindowTitle("Application execution request");
	m_askForPermissionDiag->setQuestion("Do you want to execute this application?");
	m_askForPermissionOk = m_askForPermissionDiag->addOption("Ok");
	m_askForPermissionAbort = m_askForPermissionDiag->addOption("Abort");
}

void MainWindow::initConfigSettings()
{
	cdConfig = new covise::coConfigGroup("CoviseDaemon");
	cdConfig->addConfig(covise::coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "coviseDaemon.xml", "local", true);
	covise::coConfig::getInstance()->addConfig(cdConfig);

	cfgTimeout.setAutoUpdate(true);
	cfgAutostart.setAutoUpdate(true);
	cfgAutoConnect.setAutoUpdate(true);
	cfgBackground.setAutoUpdate(true);
	cfgMinimized.setAutoUpdate(true);
	cfgArguments.setAutoUpdate(true);

	cfgTimeout.setSaveToGroup(cdConfig);
	cfgAutostart.setSaveToGroup(cdConfig);
	cfgAutoConnect.setSaveToGroup(cdConfig);
	cfgBackground.setSaveToGroup(cdConfig);
	cfgMinimized.setSaveToGroup(cdConfig);
	cfgArguments.setSaveToGroup(cdConfig);

	ui->timeoutSlider->setValue(cfgTimeout);
	on_timeoutSlider_sliderMoved(cfgTimeout);
	ui->autostartCheckBox->setChecked(cfgAutostart);
	ui->autoconnectCheckBox->setChecked(cfgAutoConnect);
	ui->backgroundCheckBox->setChecked(cfgBackground);
	ui->minimizedCheckBox->setChecked(cfgMinimized);
	ui->cmdArgsInput->setText(cfgArguments);
}

void MainWindow::setRemoteLauncherCallbacks()
{
	connect(&m_remoteLauncher, &CoviseDaemon::connectedSignal, this, &MainWindow::setStateConnected);
	connect(&m_remoteLauncher, &CoviseDaemon::disconnectedSignal, this, [this]()
			{
				m_clientList->clear();
				setStateDisconnected();
				if (ui->autoconnectCheckBox->isChecked())
					onConnectBtnClicked();
			});
	connect(&m_remoteLauncher, &CoviseDaemon::updateClient, this, &MainWindow::updateClient);
	connect(&m_remoteLauncher, &CoviseDaemon::removeClient, this, &MainWindow::removeClient);
	connect(&m_remoteLauncher, &CoviseDaemon::childProgramOutput, this, [this](const QString &childId, const QString &txt)
			{
				auto childOutput = std::find(m_childOutputs.begin(), m_childOutputs.end(), childId);
				if (childOutput == m_childOutputs.end())
				{
					childOutput = m_childOutputs.emplace(m_childOutputs.end(), childId, ui->childTabs);
				}
				childOutput->addText(txt);
			});

	connect(&m_remoteLauncher, &CoviseDaemon::childTerminated, this, [this](const QString &childId)
			{ m_childOutputs.erase(std::remove(m_childOutputs.begin(), m_childOutputs.end(), childId), m_childOutputs.end()); });
	connect(&m_remoteLauncher, &CoviseDaemon::askForPermission, this, [this](vrb::Program p, int clientID, const QString &description)
			{
				disconnect(m_askForPermissionConn);
				m_askForPermissionConn = connect(m_askForPermissionDiag, &covise::NonBlockingDialogue::answer, this, [p, clientID, this](int answer)
												 { m_remoteLauncher.answerPermissionRequest(p, clientID, answer == m_askForPermissionOk); });
				askForPermission(p, clientID, description);
			});
}

void MainWindow::initClientList()
{
	m_clientList = new ClientWidgetList(ui->clientsScrollArea, ui->clientsScrollArea);
	connect(m_clientList, &ClientWidgetList::requestProgramLaunch, this, [this](vrb::Program programID, int clientID)
			{
				std::cerr << "launching " << vrb::programNames[programID] << " on client " << clientID << std::endl;

				m_remoteLauncher.sendLaunchRequest(programID, clientID, parseCmdArgsInput());
			});
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
				}
			});
	connect(escape, &QShortcut::activated, this, [this]()
			{
				if (m_isConnecting)
				{
					onCancelBtnClicked();
				}
				else
				{
					onDisconnectBtnClicked();
				}
			});
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
	m_tray->hide();
	show();
}

void MainWindow::hideThis()
{
	if (!m_tray)
		createTrayIcon();
	m_tray->show();
	m_tray->showMessage("COVISE Daemon", " running in background");
	hide();
}

void MainWindow::createTrayIcon()
{
	m_tray = new QSystemTrayIcon(QIcon(":/images/coviseDaemon.png"), this);
	//tray->showMessage("coviseDaemon", "blablabla");
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
					showThis();
					break;
				default:
					break;
				}
			});
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

void MainWindow::askForPermission(vrb::Program p, int clientID, const QString &description)
{
	if (ui->autostartCheckBox->isChecked())
		m_remoteLauncher.answerPermissionRequest(p, clientID, true);
	else
	{
		bool wasVisible = isVisible();
		show(); //if main window is not visible showing the message box may crash

		m_askForPermissionDiag->setInfo(description);
		m_askForPermissionDiag->show();
	}
}

std::vector<std::string> MainWindow::parseCmdArgsInput()
{
	return covise::parseCmdArgString(ui->cmdArgsInput->text().toStdString());
}

//free functions

void setStackedWidget(QStackedWidget *stack, int index)
{
	stack->currentWidget()->setEnabled(false);
	stack->currentWidget()->hide();
	stack->setCurrentIndex(index);
	stack->currentWidget()->setEnabled(true);
	stack->currentWidget()->show();
}
