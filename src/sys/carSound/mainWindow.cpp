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

void mainWindow::speedChanged(int val)
{
    speedValue = speed->value();
    myRPM->setValue(speedValue);
    system->update();
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
        myLoad->setValue(engineTorque);
        myRPM->setValue(speedValue);
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
    system->update();
}

void mainWindow::watchdog()
{
    if (!messageReceived)
    {
        speedValue = 0.0;
        carSpeed = 0.0;
        engineTorque = 0.0;
        slipValue = 0.0;
        myLoad->setValue(engineTorque);
        myRPM->setValue(speedValue);
    }
    messageReceived = false;
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
    theWindow = this;
    setupUi(this);
    speedSlider->setMaximum(8000);
    speedSlider->setMinimum(90);
    speedSlider->setValue(90);
    speed->setMaximum(8000);
    speed->setMinimum(90);
    speed->setValue(90);
    speedValue = 90;
    pitchOffset->setMaximum(10.0);
    pitchOffset->setMinimum(0.01);
    pitchOffset->setSingleStep(0.01);
    volumeOffset->setMaximum(2.0);
    volumeOffset->setMinimum(0.1);
    volumeOffset->setSingleStep(0.01);
    engineTorque = 0.0;
    messageReceived = true;

    udpclient = new UDPComm(31804);
    socketNotifier = new QSocketNotifier(udpclient->getReceiveSocket(), QSocketNotifier::Read);
    QObject::connect(socketNotifier, SIGNAL(activated(int)), this, SLOT(dataReceived(int)));

    QObject::connect(upDown, SIGNAL(toggled(bool)), this, SLOT(animateToggle(bool)));
    QObject::connect(speedSlider, SIGNAL(sliderMoved(int)), this, SLOT(speedChanged(int)));
    QObject::connect(this, SIGNAL(setPitch(double)), pitchOffset, SLOT(setValue(double)));
    QObject::connect(this, SIGNAL(setVolume(double)), volumeOffset, SLOT(setValue(double)));
    QObject::connect(anlasserButton, SIGNAL(pressed()), this, SLOT(startAnlasser()));
    QObject::connect(hupeButton, SIGNAL(pressed()), this, SLOT(startHupe()));
    QObject::connect(hupeButton, SIGNAL(released()), this, SLOT(stopHupe()));

    QObject::connect(speed, SIGNAL(valueChanged(int)), this, SLOT(speedChanged(int)));

    watchdogTimer = new QTimer(this);
    connect(watchdogTimer, SIGNAL(timeout()), this, SLOT(watchdog()));
    watchdogTimer->start(2000);

    show();
    myInstance = this;

    //FMOD
    FMOD_RESULT result;

    eventsystem = 0;
    system = 0;

    result = FMOD::EventSystem_Create(&eventsystem);
    if (result != FMOD_OK)
    {
        printf("FMOD error! (%d) \n", result);
        exit(-1);
    }

    result = eventsystem->getSystemObject(&system);
    if (result != FMOD_OK)
    {
        printf("FMOD error! (%d) \n", result);
        exit(-1);
    }

    //result = system->init(100, FMOD_INIT_NORMAL, 0);	// Initialize FMOD.
    result = eventsystem->init(256, FMOD_INIT_NORMAL, 0);
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

    result = eventsystem->load("c:\\data\\Porsche\\sounds\\porsche911.fev", NULL, &project);

    //FMOD::EventGroup *myGroup=NULL;
    /*  int numev=0;
   eventsystem->getNumEvents(&numev);
      for(int i=0;i<numev;i++)
      {
         result = eventsystem->getEventBySystemID(i,FMOD_EVENT_DEFAULT,&myEvent);
      }*/
    //result = eventsystem->getGroup("AdvancedTechniques",false,&myGroup);

    myEvent = NULL;
    myRPM = NULL;
    myLoad = NULL;
    result = eventsystem->getEvent("porsche911/Car/CarEngine", FMOD_EVENT_DEFAULT, &myEvent);
    result = myEvent->getParameter("rpm", &myRPM);
    result = myEvent->getParameter("load", &myLoad);
    myRPM->setValue(0);
    myLoad->setValue(1.0);
    result = myEvent->start();
    myEvent->getPaused(false);
    myEvent->setVolume(1.0);
}

mainWindow *mainWindow::myInstance = NULL;
mainWindow::~mainWindow()
{
}
