/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ui_carSound.h"
#include "EventSoundSample.h"
#include "UDPComm.h"
#include <QSocketNotifier>
#include <fmod.hpp>
#include <fmod_event.hpp>

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
    void updatePitch(double val)
    {
        emit setPitch(val);
    };
    void updateVolume(double val)
    {
        emit setVolume(val);
    };

    FMOD::EventSystem *eventsystem;
    FMOD::System *system;
    FMOD::EventProject *project;

    FMOD::Event *myEvent;
    FMOD::EventParameter *myRPM;
    FMOD::EventParameter *myLoad;

private:
    static mainWindow *myInstance;
private slots:
    virtual void speedChanged(int val);
    virtual void dataReceived(int socket);
    virtual void startAnlasser();
    virtual void watchdog();
    virtual void startHupe();
    virtual void stopHupe();

signals:
    void setPitch(double val);
    void setVolume(double val);
};

extern mainWindow *theWindow;