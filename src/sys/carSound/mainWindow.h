/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ui_carSound.h"
#include "EventSoundSample.h"
#include "UDPComm.h"
#include <QSocketNotifier>
#include <fmod_studio.hpp>

class mainWindow : public QMainWindow, public Ui::MainWindow
{
    Q_OBJECT

public:
    mainWindow(QWidget *parent = 0);
    ~mainWindow();
    static mainWindow *instance()
    {
        return myInstance;
    };
    float oldPitch;
    int speedValue;
    float carSpeed;
    float engineTorque;
    float slipValue;
    bool animUp;
    bool messageReceived;
    QTimer *watchdogTimer;
    UDPComm *udpclient;
    QSocketNotifier *socketNotifier;
    std::vector<EventSoundSample *> eventSounds;
    EventSoundSample *hupe;
    EventSoundSample *anlasser;

	FMOD::Studio::System *eventsystem;
    FMOD::System *system;
	FMOD::Studio::Bank* porsche911Bank;
	FMOD::Studio::Bank* MasterBank;
	FMOD::Studio::Bank* MasterStringBank;

	FMOD::Studio::EventInstance *CarEventInstance;
	FMOD::Studio::EventDescription *CarEventDescription;
	FMOD::Studio::EventInstance *WheelEventInstance;
	FMOD::Studio::EventDescription *WheelEventDescription;

	FMOD::Studio::ParameterInstance *CarRPM;
	FMOD::Studio::ParameterInstance *CarLoad;
	FMOD::Studio::ParameterInstance *WheelVelocity;
	FMOD::Studio::ParameterInstance *Wheelslip;

private:
    static mainWindow *myInstance;
	void updateValues();
private slots:
    virtual void speedChanged(int val);
	virtual void velocityChanged(int val);
	virtual void slipChanged(int val);
    virtual void dataReceived(int socket);
    virtual void startAnlasser();
    virtual void watchdog();
    virtual void startHupe();
    virtual void stopHupe();

signals:
};

extern mainWindow *theWindow;