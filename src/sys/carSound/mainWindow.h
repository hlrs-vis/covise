/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ui_carSound.h"
#include "EventSoundSample.h"
#include <util/UDPComm.h>
#include <QSocketNotifier>
#include <fmod_studio.hpp>
#include <net/covise_connect.h>
#include <config/coConfig.h>
#include "soundClient.h"
class QTreeWidgetItem;

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


    int openServer();
    int TCPPort = 31805;
    int UDPPort = 31804;
    QSocketNotifier* serverSN;
    const covise::ServerConnection* sConn = nullptr;
    covise::ConnectionList connections;
    covise::Message* msg = nullptr;
    QTimer* m_periodictimer = nullptr;
    std::list<soundClient*>clients;
    ClientSoundSample* currentSound = nullptr;
    void removeClient(soundClient* c);
    covise::coConfigGroup *soundConfig;

    covise::coConfigString cacheDir;

private:
    static mainWindow *myInstance;
	void updateValues();
private slots:
    void closeServer();
    void processMessages();
    bool handleClient(covise::Message* msg);
    bool serverRunning();
    void selectClient(QTreeWidgetItem*);
    void selectSound(QTreeWidgetItem*);
    virtual void speedChanged(int val);
	virtual void velocityChanged(int val);
	virtual void slipChanged(int val);
    virtual void dataReceived(int socket);
    virtual void startAnlasser();
    virtual void watchdog();
    virtual void startHupe();
    virtual void stopHupe();
    virtual void onDirBrowser();
    void onDirChanged(const QString& d);
    void play() { if (currentSound) currentSound->start(); };
    void stop() { if (currentSound) currentSound->stop(); };
    void rewind() { if (currentSound) currentSound->rewind(); };
    void volume(int v) 
    {
        if (currentSound) 
            currentSound->volume(2*v/99.0); 
    };
    void pitch(int p) { if (currentSound) currentSound->pitch(0.7+(0.6*p/99.0)); };

signals:
};

extern mainWindow *theWindow;