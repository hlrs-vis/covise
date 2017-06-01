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

void mainWindow::slipChanged(int val)
{
	updateValues();
}

void mainWindow::dataReceived(int socket)
{
    messageReceived = true;
    int msgSize = udpclient->readMessage();
    if (msgSize >= 12)
    {
        float *floatValues = (float *)udpclient->rawBuffer();
        speedValue = floatValues[0];
        carSpeed = floatValues[1];
        engineTorque = floatValues[2];
        slipValue = floatValues[3];
		CarRPM->setValue(speedValue);
		CarLoad->setValue(engineTorque);
		WheelVelocity->setValue(carSpeed);
		Wheelslip->setValue(slipValue);
    }
    else if (msgSize == 2) // two bytes start/stop/ soundnumber
    {
        int soundNumber = udpclient->rawBuffer()[1];
        char action = udpclient->rawBuffer()[0];
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
void mainWindow::startHupe()
{
    hupe->start();
}
void mainWindow::stopHupe()
{
    hupe->stop();
}

mainWindow *theWindow;

mainWindow::mainWindow(QWidget *parent)
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

    udpclient = new UDPComm(31804);
    socketNotifier = new QSocketNotifier(udpclient->getReceiveSocket(), QSocketNotifier::Read);
    QObject::connect(socketNotifier, SIGNAL(activated(int)), this, SLOT(dataReceived(int)));

	QObject::connect(speedSlider, SIGNAL(sliderMoved(int)), this, SLOT(speedChanged(int)));
	QObject::connect(velocitySlider, SIGNAL(sliderMoved(int)), this, SLOT(velocityChanged(int)));
	QObject::connect(SlipSlider, SIGNAL(sliderMoved(int)), this, SLOT(slipChanged(int)));
    QObject::connect(anlasserButton, SIGNAL(pressed()), this, SLOT(startAnlasser()));
    QObject::connect(hupeButton, SIGNAL(pressed()), this, SLOT(startHupe()));
    QObject::connect(hupeButton, SIGNAL(released()), this, SLOT(stopHupe()));

    watchdogTimer = new QTimer(this);
    connect(watchdogTimer, SIGNAL(timeout()), this, SLOT(watchdog()));
    watchdogTimer->start(2000);

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
	FMOD::Studio::EventDescription *array[100];
/*	int num;
	porsche911Bank->getEventList(array,100, &num);
	for (int i = 0; i < num; i++)
	{
		char path[1000];
		int len;
		array[i]->getPath(path, 1000, &len);
		fprintf(stderr, "%s\n", path);
	}*/


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

mainWindow *mainWindow::myInstance = NULL;
mainWindow::~mainWindow()
{
}
