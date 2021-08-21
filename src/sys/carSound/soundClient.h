#pragma once

#include <net/covise_connect.h>
#include <QObject>
#include <ClientSoundSample.h>

class QSocketNotifier;
class QTreeWidgetItem;

enum  Columns {
    ID,
    Application,
    User,
    Host,
    IP,
};


class soundClient: public QObject
{

    Q_OBJECT

public:
    soundClient(const covise::Connection * conn);
    ~soundClient();
    void setClientInfo(const std::string Application, const std::string user, const std::string host, std::string IP);
    const covise::Connection* toClient;
    unsigned int ID;
    void addSound(std::string fileName);

    bool send(covise::TokenBuffer& tb);
    std::list<ClientSoundSample*> sounds;
private:
    QSocketNotifier *clientSN;
    static unsigned int IDCounter;
    std::string application;
    std::string user;
    std::string host;
    std::string IP;
    QTreeWidgetItem* myItem;
private slots:

};

