#include "mainWindow.h"
#include "ui_mainWindow.h"
#include "clientWidget.h"

#include <QMessageBox>
#include <QShortcut>
#include <QTextStream>

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <csignal>
#include <cassert>

using namespace vrb;

MainWindow::MainWindow(const vrb::VrbCredentials &credentials, QWidget *parent)
	: QMainWindow(parent), ui(new Ui::MainWindow)
{
	qRegisterMetaType<vrb::Program>();
	qRegisterMetaType<std::vector<std::string>>();

	ui->setupUi(this);
	setRemoteLauncherCallbacks();
	ui->tcpInput->setValue(credentials.tcpPort);
	ui->udpInput->setValue(credentials.udpPort);
	ui->hostIpf->setText(QString(credentials.ipAddress.c_str()));
	ui->progressBar->setVisible(false);
	connect(this, &MainWindow::updateStatusBarSignal, this, &MainWindow::updateStatusBar);
	readOptions();
	m_clientList = new ClientWidgetList(ui->clientsScrollArea, ui->clientsScrollArea);
	connect(m_clientList, &ClientWidgetList::requestProgramLaunch, this, [this](vrb::Program programID, int clientID) {
		std::cerr << "launching " << vrb::programNames[programID] << " on client " << clientID << std::endl;

		m_remoteLauncher.sendLaunchRequest(programID, clientID, parseCmdArgsInput());
	});

	setHotkeys();


	if (ui->autoconnectCheckBox->isChecked())
	{
		onConnectBtnClicked();
	}
	else
	{
		setStateDisconnected();
	}
}

MainWindow::~MainWindow()
{
	dumpOptions();
	m_isConnecting = false;
	if (m_waitFuture.valid())
	{
		m_waitFuture.get();
	}
	delete ui;
}

void MainWindow::setHotkeys(){
		auto escape = new QShortcut(QKeySequence(Qt::Key_Escape), this);
	auto enter = new QShortcut(QKeySequence(Qt::Key_Return), this);
	auto sideMenu = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Tab), this);
	connect(enter, &QShortcut::activated, this, [this]() {
		if (!m_isConnecting)
		{
			onConnectBtnClicked();
		}
	});
	connect(escape, &QShortcut::activated, this, [this]() {
		if (m_isConnecting)
		{
			onCancelBtnClicked();
		}
		else
		{
			onDisconnectBtnClicked();
		}
		
	});
	connect(sideMenu, &QShortcut::activated, this, [this]() {
		on_actionSideMenuAction_triggered();
	});
}

void MainWindow::on_actionSideMenuAction_triggered()
{
	Guard g(m_mutex);
	int curr = ui->MenuStackedWidget->currentIndex();
	curr == 0 ? curr = 1 : curr = 0;
	setStackedWidget(ui->MenuStackedWidget, curr);
}

void MainWindow::on_timeoutSlider_sliderMoved(int val)
{
	Guard g(m_mutex);
	ui->timeoutLabel->setText(QString("timeout: ") + QString::number(val) + QString("s"));
}

void MainWindow::setRemoteLauncherCallbacks()
{
	connect(&m_remoteLauncher, &VrbRemoteLauncher::connectedSignal, this, &MainWindow::setStateConnected);
	connect(&m_remoteLauncher, &VrbRemoteLauncher::disconnectedSignal, this, [this]() {
		setStateDisconnected();
		if (ui->autoconnectCheckBox->isChecked())
			onConnectBtnClicked();
	});
	connect(&m_remoteLauncher, &VrbRemoteLauncher::updateClient, this, &MainWindow::updateClient);
	connect(&m_remoteLauncher, &VrbRemoteLauncher::removeClient, this, &MainWindow::removeClient);
	connect(&m_remoteLauncher, &VrbRemoteLauncher::launchSignal, this, &MainWindow::launchProgram);
}

void MainWindow::showConnectionProgressBar(int seconds)
{
	int resolution = 2;
	ui->progressBar->reset();
	ui->progressBar->setRange(0, resolution * seconds);
	ui->progressBar->setVisible(true);
	if (m_waitFuture.valid())
	{
		m_waitFuture.get();
	}

	m_waitFuture = std::async(std::launch::async, [this, seconds, resolution]() {
		while (m_isConnecting)
		{
			QMetaObject::invokeMethod(this, "updateStatusBarSignal", Qt::QueuedConnection);
			std::this_thread::sleep_for(std::chrono::milliseconds(1000 / resolution));
		}
	});
}

void MainWindow::setStateDisconnected()
{
	m_remoteLauncher.disconnect();
	ui->progressBar->setVisible(false);
	m_isConnecting = false;
	ui->connectBtn->setText("connect");
	ui->connectBtn->disconnect();
	ui->conncetedLabel->setStyleSheet(QString("image: url(:/images/redCircle.png);"));
	connect(ui->connectBtn, &QPushButton::clicked, this, &MainWindow::onConnectBtnClicked);
	setStackedWidget(ui->InfoStackedWidget, 1);
}

void MainWindow::setStateConnecting()
{
	showConnectionProgressBar(ui->timeoutSlider->value());
	ui->connectBtn->setText("cancel");
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
		}
		else
		{
			ui->progressBar->setValue(newVal);
		}
	}
}

void MainWindow::updateClient(int clientID, QString clientInfo)
{
	m_clientList->addClient(clientID, clientInfo);
}

void MainWindow::removeClient(int clientID)
{
	m_clientList->removeClient(clientID);
}

void MainWindow::launchProgram(vrb::Program programID, const std::vector<std::string> &args)
{
	bool execute = ui->autostartCheckBox->isChecked();
	if (!execute)
	{
		QMessageBox msgBox{this};
		QString text;
		QTextStream ss(&text);
		ss << "Start of " << vrb::programNames[programID] << " requested!";
		msgBox.setText(text);
		msgBox.setInformativeText("Do you want to execute that program?");
		msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Ok);
		int ret = msgBox.exec();
		execute = ret == QMessageBox::Ok ? true : false;
	}
	if (execute)
	{
		std::cerr << "launching " << vrb::programNames[programID] << std::endl;
		spawnProgram(programID, args);
	}
}

void MainWindow::dumpOptions()
{
	std::string path = getenv("COVISE_PATH");
	path += "/VrbRemoteLauncher.settings";
	std::fstream file(path, std::ios_base::binary | std::ios_base::out);
	if (file.is_open())
	{
		covise::TokenBuffer tb;
		tb << __DATE__ << __TIME__;
		tb << ui->timeoutSlider->value();
		tb << ui->autostartCheckBox->isChecked();
		tb << ui->autoconnectCheckBox->isChecked();
		tb << ui->cmdArgsInput->text().toStdString();
		int size = tb.getData().length();
		file.write((char *)&size, sizeof(size));
		file.write(tb.getData().data(), size);
	}
	else
	{
		std::cerr << "failed to dump settings to " << path << std::endl;
	}
}

void MainWindow::readOptions()
{
	std::string path = getenv("COVISE_PATH");
	path += "/VrbRemoteLauncher.settings";
	std::fstream file(path, std::ios_base::binary | std::ios_base::in);
	if (file.is_open())
	{
		int l;
		file.read((char *)&l, sizeof(l));
		covise::DataHandle dh{(size_t)l};
		file.read(dh.accessData(), l);
		covise::TokenBuffer tb(dh);
		int timeout;
		bool autostart, autoconnect;
		char *date, *time;
		std::string args;
		tb >> date >> time;
		if (strcmp(date, __DATE__) != 0 | strcmp(time, __TIME__) != 0)
		{
			std::cerr << "failed to load settings: different compilation!" << std::endl;
			return;
		}

		tb >> timeout >> autostart >> autoconnect >> args;
		ui->timeoutSlider->setValue(timeout);
		on_timeoutSlider_sliderMoved(timeout);
		ui->autostartCheckBox->setChecked(autostart);
		ui->autoconnectCheckBox->setChecked(autoconnect);
		ui->cmdArgsInput->setText(args.c_str());
	}
}

std::vector<std::string> MainWindow::parseCmdArgsInput()
{
	std::string argsString = ui->cmdArgsInput->text().toStdString();
	std::vector<std::string> args;
	auto end = argsString.begin();
	while (true)
	{
		auto begin = end;
		end = std::find(begin, argsString.end(), ' ');
		if (begin != end)
		{
			args.emplace_back(argsString.substr(begin - argsString.begin(), end - begin));
		}
		if (end == argsString.end())
		{
			return args;
		}

		while (*end == ' ')
		{
			++end;
			if (end == argsString.end())
			{
				return args;
			}
		}
	}
}

void setStackedWidget(QStackedWidget *stack, int index)
{
	stack->currentWidget()->setEnabled(false);
	stack->currentWidget()->hide();
	stack->setCurrentIndex(index);
	stack->currentWidget()->setEnabled(true);
	stack->currentWidget()->show();
}
