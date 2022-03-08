/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "mainWindow.h"

#include <stdio.h>
#include <string.h>
#include "math.h"

#include "windows.h"
#include "mmsystem.h"
#include <QMessageBox>
#include <QTimer>
#include "remoteSoundMessages.h"
#include <config/CoviseConfig.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <qfiledialog.h>

// internal prototypes (required for the Metrowerks CodeWarrior compiler)
int main(int argc, char *argv[]);

void mainWindow::updateValues()
{
	speedValue = speedSlider->value();
	CarRPM->setValue(speedValue);
	carSpeed = velocitySlider->value();
	WheelVelocity->setValue(carSpeed);
	slipValue = SlipSlider->value() / 100.0;
	Wheelslip->setValue(slipValue);
	eventsystem->update();
}
void mainWindow::speedChanged(int val)
{
	updateValues();
}

void mainWindow::velocityChanged(int val)
{
	updateValues();
}

void mainWindow::removeClient(soundClient *c)
{
    connections.remove(c->toClient);
    delete c;
}
bool mainWindow::handleClient(covise::Message* msg)
{
    if ((msg->type == covise::COVISE_MESSAGE_SOCKET_CLOSED) || (msg->type == covise::COVISE_MESSAGE_CLOSE_SOCKET))
    {
        std::cerr << "mainWindow: socket closed" << std::endl;

        for (const auto& c : clients)
        {
            if (msg->conn == c->toClient)
            {
                removeClient(c);
                break;
            }
        }
        return true; // we have been deleted, exit immediately
    }
    covise::TokenBuffer tb(msg);
    switch (msg->type)
    {
    case covise::COVISE_MESSAGE_SOUND:
    {
        soundClient* currentClient = nullptr;
        for (const auto& c : clients)
        {
            if (msg->conn == c->toClient)
            {
                currentClient = c;
                break;
            }
        }
        if (currentClient)
        {
            int type;
            tb >> type;
            switch (type)
            {
            case SOUND_CLIENT_INFO:
            {
                std::string Application;
                std::string user;
                std::string host;
                std::string IP;
                tb >> Application;
                tb >> user;
                tb >> host;
                tb >> IP;
                currentClient->setClientInfo(Application, user, host, IP);
                break;
            }
            case SOUND_NEW_SOUND:
            {
                std::string fileName;
                uint64_t fileSize;
                time_t fileTime;
                tb >> fileName;
                tb >> fileSize;
                tb >> fileTime;
                currentClient->addSound(fileName,fileSize,fileTime);
                break;
            }
            case SOUND_DELETE_SOUND:
            {
                break;
            }
            case SOUND_SOUND_FILE:
            {
                break;
            }
            }
        }
    }
    break;
    default:
    {
        if (msg->type >= 0 && msg->type < covise::COVISE_MESSAGE_LAST_DUMMY_MESSAGE)
            std::cerr << "TUIApplication::handleClient err: unknown COVISE message type " << msg->type << " " << covise::covise_msg_types_array[msg->type] << std::endl;
        else
            std::cerr << "TUIApplication::handleClient err: unknown COVISE message type " << msg->type << std::endl;
    }
    break;
    }
    return false;
}

void mainWindow::slipChanged(int val)
{
	updateValues();
}

void mainWindow::dataReceived(int socket)
{
    messageReceived = true;
    int msgSize = udpclient->readMessage();
    if (msgSize > sizeof(int))
    {
        const int* rawBuffer = (int*)udpclient->rawBuffer();
        const char* payload = udpclient->rawBuffer() + sizeof(int);
        unsigned int msgType = *rawBuffer;
        if (msgType == TypeCarSound)
        {
            float* floatValues = (float*)payload;
            speedValue = floatValues[0];
            carSpeed = floatValues[1];
            engineTorque = floatValues[2];
            slipValue = floatValues[3];
            CarRPM->setValue(speedValue);
            CarLoad->setValue(engineTorque);
            WheelVelocity->setValue(carSpeed);
            Wheelslip->setValue(slipValue);
        }
        else if (msgType == TypeSimpleSound) // two bytes start/stop/ soundnumber
        {
            int soundNumber = payload[1];
            char action = payload[0];
            if (soundNumber < eventSounds.size())
            {
                if (action == '\0')
                {
                    eventSounds[soundNumber]->stop();
                }
                else if (action == '\1')
                {
                    eventSounds[soundNumber]->start();
                }
                else if (action == '\2')
                {
                    eventSounds[soundNumber]->continuePlaying();
                }
                else if (action == '\3')
                {
                    eventSounds[soundNumber]->loop(true);
                }
                else if (action == '\4')
                {
                    eventSounds[soundNumber]->loop(false);
                }
                else if (action == '\5')
                {
                    eventSounds[soundNumber]->rewind();
                }
            }
        }
        else if (msgType == TypeRemoteSoundDelay) // 
        {
            RemoteSoundDelayData* rsdd = (RemoteSoundDelayData*)rawBuffer;
            for (const auto& c : clients)
            {
                for (const auto& s : c->sounds)
                {
                    if (s->ID == rsdd->soundID)
                    {
                        s->setDelay(rsdd->startValue, rsdd->endValue, rsdd->stopChannel);
                        break;
                    }
                }
            }
        }
        else if (msgType == TypeRemoteSound) // action and flot value
        {
            RemoteSoundData* rsd = (RemoteSoundData*)rawBuffer;
            for (const auto& c : clients)
            {
                for (const auto& s : c->sounds)
                {
                    if (s->ID == rsd->soundID)
                    {
                        if (rsd->action == (unsigned char)remoteSoundActions::Stop)
                        {
                            s->stop();
                        }
                        else if (rsd->action == (unsigned char)remoteSoundActions::Start)
                        {
                            s->start();
                        }
                        else if (rsd->action == (unsigned char)remoteSoundActions::enableLoop)
                        {
                            s->loop(true,(int)rsd->value);
                        }
                        else if (rsd->action == (unsigned char)remoteSoundActions::disableLoop)
                        {
                            s->loop(false, 0);
                        }
                        else if (rsd->action == (unsigned char)remoteSoundActions::Volume)
                        {
                            s->volume(rsd->value);
                        }
                        else if (rsd->action == (unsigned char)remoteSoundActions::Pitch)
                        {
                            s->pitch(rsd->value);
                        }
                        break;
                    }
                }
            }
        }
    }
	eventsystem->update();
}

void mainWindow::watchdog()
{
    if (!messageReceived)
    {
        speedValue = 0.0;
        carSpeed = 0.0;
        engineTorque = 0.0;
        slipValue = 0.0;

		CarRPM->setValue(speedValue);
		CarLoad->setValue(engineTorque);
		WheelVelocity->setValue(carSpeed);
		Wheelslip->setValue(slipValue);
    }
    messageReceived = false;
	eventsystem->update();
}

void mainWindow::startAnlasser()
{
    anlasser->start();
}
void mainWindow::selectClient(QTreeWidgetItem*)
{
}
void mainWindow::selectSound(QTreeWidgetItem *item)
{
    currentSound = nullptr;
    for (const auto& c : clients)
    {
        for (const auto& s : c->sounds)
        {
            if (s->myItem == item)
            {
                currentSound = s;
                break;
            }
        }
    }
}
void mainWindow::startHupe()
{
    hupe->start();
}
void mainWindow::stopHupe()
{
    hupe->stop();
}

mainWindow *theWindow;

mainWindow::mainWindow(QWidget* parent) : cacheDir("System.RemoteSound.CacheDir")
{

    //CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    theWindow = this;
    setupUi(this);
    speedSlider->setMaximum(8000);
    speedSlider->setMinimum(90);
    speedSlider->setValue(90);
    velocitySlider->setMaximum(100);
    velocitySlider->setMinimum(0);
    velocitySlider->setValue(0);
    carSpeed = 0;
    SlipSlider->setMaximum(100);
    SlipSlider->setMinimum(0);
    SlipSlider->setValue(0);
    slipValue = 0;
    engineTorque = 0.0;
    messageReceived = true;

    // automatic storage of config values
    soundConfig = new covise::coConfigGroup("RemoteSound");
    soundConfig->addConfig(covise::coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "RemoteSound.xml", "local", true);
    covise::coConfig::getInstance()->addConfig(soundConfig);
    cacheDir.setAutoUpdate(true);
    cacheDir.setSaveToGroup(soundConfig);
    if (!cacheDir.hasValidValue())
    {
        cacheDir = "c:/tmp";

        std::cerr << std::string(cacheDir) << std::endl;
    }
    cacheDirectory->setText(std::string(cacheDir).c_str());
    std::cerr << std::string(cacheDir) << std::endl;

    //clientTable->setFont(smallFont);
    QStringList labels;
    labels << "ID"
        << "Application"
        << "User"
        << "Host"
        << "IP";
    clientTable->setHeaderLabels(labels);
    clientTable->setMinimumSize(clientTable->sizeHint());
    clientTable->header()->resizeSections(QHeaderView::ResizeToContents);
    connect(clientTable, SIGNAL(itemClicked(QTreeWidgetItem*, int)),
        this, SLOT(selectClient(QTreeWidgetItem*)));

    QStringList soundLabels;
    soundLabels << "ID"
        << "State"
        << "Client"
        << "FileName"
        << "Volume"
        << "Pitch";
    soundTable->setHeaderLabels(soundLabels);
    soundTable->setMinimumSize(soundTable->sizeHint());
    soundTable->header()->resizeSections(QHeaderView::ResizeToContents);
    connect(soundTable, SIGNAL(itemClicked(QTreeWidgetItem*, int)),
        this, SLOT(selectSound(QTreeWidgetItem*)));
    connect(dirBrowser, SIGNAL(clicked(bool)), this, SLOT(onDirBrowser()));
    connect(cacheDirectory, SIGNAL(textEdited(const QString &)), this, SLOT(onDirChanged(const QString&)));

    UDPPort = covise::coCoviseConfig::getInt("UDPPort", "RemoteSound", 31804);
    udpclient = new UDPComm(UDPPort);
    socketNotifier = new QSocketNotifier(udpclient->getReceiveSocket(), QSocketNotifier::Read);
    QObject::connect(socketNotifier, SIGNAL(activated(int)), this, SLOT(dataReceived(int)));

	QObject::connect(speedSlider, SIGNAL(sliderMoved(int)), this, SLOT(speedChanged(int)));
	QObject::connect(velocitySlider, SIGNAL(sliderMoved(int)), this, SLOT(velocityChanged(int)));
	QObject::connect(SlipSlider, SIGNAL(sliderMoved(int)), this, SLOT(slipChanged(int)));
    QObject::connect(anlasserButton, SIGNAL(pressed()), this, SLOT(startAnlasser()));
    QObject::connect(hupeButton, SIGNAL(pressed()), this, SLOT(startHupe()));
    QObject::connect(hupeButton, SIGNAL(released()), this, SLOT(stopHupe()));

    QObject::connect(playButton, SIGNAL(pressed()), this, SLOT(play()));
    QObject::connect(stopButton, SIGNAL(pressed()), this, SLOT(stop()));
    QObject::connect(rewindButton, SIGNAL(pressed()), this, SLOT(rewind()));
    QObject::connect(volumeDial, SIGNAL(valueChanged(int)), this, SLOT(volume(int)));
    QObject::connect(pitchDial, SIGNAL(valueChanged(int)), this, SLOT(pitch(int)));

    watchdogTimer = new QTimer(this);
    connect(watchdogTimer, SIGNAL(timeout()), this, SLOT(watchdog()));
    watchdogTimer->start(2000);


    TCPPort = covise::coCoviseConfig::getInt("TCPPort", "RemoteSound", 31805);
    show();
    myInstance = this;

    //FMOD
    FMOD_RESULT result;

    eventsystem = 0;
    system = 0;

    result = FMOD::Studio::System::create(&eventsystem);
    if (result != FMOD_OK)
    {
        printf("FMOD error! (%d) \n", result);
        exit(-1);
    }

    result = eventsystem->initialize(256, FMOD_STUDIO_INIT_NORMAL,FMOD_INIT_NORMAL, 0);
    if (result != FMOD_OK)
    {
        printf("FMOD error! (%d) \n", result);
        exit(-1);
    }
	result = eventsystem->getLowLevelSystem(&system);
	if (result != FMOD_OK)
	{
		printf("FMOD error! (%d) \n", result);
		exit(-1);
	}

    hupe = new EventSoundSample("c:\\data\\Porsche\\sounds\\horn.wav");
    hupe->loop(true);
    anlasser = new EventSoundSample("c:\\data\\Porsche\\sounds\\AnlasserInnen.wav");
    anlasser->loop(false);

    eventSounds.push_back(hupe);
    eventSounds.push_back(anlasser);

	result = eventsystem->loadBankFile("c:\\data\\Porsche\\sounds\\Desktop\\MasterBank.bank", NULL, &MasterBank);
	if (result != FMOD_OK)
	{
		printf("FMOD error! (%d)  couldnot load Master\n", result);
		exit(-1);
	}
	result = eventsystem->loadBankFile("c:\\data\\Porsche\\sounds\\Desktop\\MasterBank.strings.bank", NULL, &MasterStringBank);
	if (result != FMOD_OK)
	{
		printf("FMOD error! (%d)  couldnot load Master strings\n", result);
		exit(-1);
	}
	result = eventsystem->loadBankFile("c:\\data\\Porsche\\sounds\\Desktop\\porsche911.bank", NULL, &porsche911Bank);
	if (result != FMOD_OK)
	{
		printf("FMOD error! (%d) could not load poersche911 bank\n", result);
		exit(-1);
	}
	//FMOD::Studio::EventDescription *array[100];
/*	int num;
	porsche911Bank->getEventList(array,100, &num);
	for (int i = 0; i < num; i++)
	{
		char path[1000];
		int len;
		array[i]->getPath(path, 1000, &len);
		fprintf(stderr, "%s\n", path);
	}*/

    if (openServer() < 0)
    {
        return;
    }

	CarEventInstance = NULL;
	CarEventDescription = NULL;
	WheelEventInstance = NULL;
	WheelEventDescription = NULL;
	result = eventsystem->getEvent("event:/Car/CarEngine", &CarEventDescription);
	result = eventsystem->getEvent("event:/Car/Wheel", &WheelEventDescription);

	result = CarEventDescription->createInstance(&CarEventInstance);
	result = WheelEventDescription->createInstance(&WheelEventInstance);

	result = CarEventInstance->getParameter("rpm", &CarRPM);
	result = CarEventInstance->getParameter("load", &CarLoad);
	result = WheelEventInstance->getParameter("Velocity", &WheelVelocity); // in m/s 0-100 --> 0 - 360km/h
	result = WheelEventInstance->getParameter("slip", &Wheelslip);  // 0-1.0

	CarRPM->setValue(0);
	CarLoad->setValue(1.0);
	WheelVelocity->setValue(0.0);
	Wheelslip->setValue(0.0);

    result = CarEventInstance->start();
	result = CarEventInstance->setVolume(1.0);
	result = WheelEventInstance->start();
	result = WheelEventInstance->setVolume(1.0);
	eventsystem->update();

}

//------------------------------------------------------------------------
void mainWindow::closeServer()
//------------------------------------------------------------------------
{
    delete serverSN;
    serverSN = NULL;

    if (sConn)
    {
        connections.remove(sConn);
        sConn = NULL;
    }

}

//------------------------------------------------------------------------
int mainWindow::openServer()
//------------------------------------------------------------------------
{
    connections.remove(sConn);
    sConn = connections.tryAddNewListeningConn<covise::ServerConnection>(TCPPort, 0, 0);
    if (!sConn)
    {
        return (-1);
    }

    msg = new covise::Message;

    serverSN = new QSocketNotifier(sConn->get_id(NULL), QSocketNotifier::Read);

    //cerr << "listening on port " << port << endl;
// weil unter windows manchmal Messages verloren gehen
// der SocketNotifier wird nicht oft genug aufgerufen)
#if defined(_WIN32) || defined(__APPLE__)
    m_periodictimer = new QTimer;
    QObject::connect(m_periodictimer, SIGNAL(timeout()), this, SLOT(processMessages()));
    m_periodictimer->start(1000);
#endif

    QObject::connect(serverSN, SIGNAL(activated(int)), this, SLOT(processMessages()));
    return 0;
}

//------------------------------------------------------------------------
bool mainWindow::serverRunning()
//------------------------------------------------------------------------
{
    return sConn && sConn->is_connected();
}

//------------------------------------------------------------------------
void mainWindow::processMessages()
//------------------------------------------------------------------------
{
    //qDebug() << "process message called";
    const covise::Connection* conn;
    while ((conn = connections.check_for_input(0.0001f)))
    {
        if (conn == sConn) // connection to server port
        {
            auto conn = sConn->spawn_connection();
            struct linger linger;
            linger.l_onoff = 0;
            linger.l_linger = 0;
            setsockopt(conn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char*)&linger, sizeof(linger));
            const covise::Connection* clientConn = connections.add(std::move(conn)); //add new connection;
            soundClient *sc = new soundClient(clientConn);
            clients.push_back(sc);
        }
        else
        {
            if (conn->recv_msg(msg))
            {
                if (msg)
                {
                    if (handleClient(msg))
                    {
                        return; // we have been deleted, exit immediately
                    }
                }
            }
        }
    }
}

void mainWindow::onDirBrowser()
{
    QStringList fileNames;
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::Directory);
    if (dialog.exec())
    {
        fileNames = dialog.selectedFiles();
        cacheDirectory->setText(fileNames[0]);
        cacheDir = fileNames[0].toStdString();
        soundConfig->save();
    }
}
void mainWindow::onDirChanged(const QString& d)
{
    QString dirName = cacheDirectory->text();
    if (QDir(dirName).exists())
    {
        cacheDir = dirName.toStdString();
        soundConfig->save();
    }
    if (!cacheDirectory->hasFocus())
    {
        cerr << "no focus" << endl;
    }
    /*else
    {
        QMessageBox::StandardButton reply;
        reply = QMessageBox::question(this, "Directory" + dirName + "does not exist,", "Create it?",
            QMessageBox::Yes | QMessageBox::No);
        if (reply == QMessageBox::Yes) {
            if (QDir().mkpath(dirName))
            {
                cacheDir = dirName;
                soundConfig->save();
            }
        }
        else {
            qDebug() << "Yes was *not* clicked";
        }
    }*/
}

mainWindow *mainWindow::myInstance = NULL;
mainWindow::~mainWindow()
{
    while (clients.size())
    {
        delete clients.front();
    }
}
