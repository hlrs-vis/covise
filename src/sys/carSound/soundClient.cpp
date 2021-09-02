#include "soundClient.h"
#include <QSocketNotifier>
#include "mainWindow.h"
#include "ClientSoundSample.h"
#include "remoteSoundMessages.h"
#include "net/message_types.h"

soundClient::soundClient(const covise::Connection* conn)
{
    ID = IDCounter++;
    clientSN = new QSocketNotifier(conn->get_id(NULL), QSocketNotifier::Read);
    QObject::connect(clientSN, SIGNAL(activated(int)),
        mainWindow::instance(), SLOT(processMessages()));

    myItem = new QTreeWidgetItem(mainWindow::instance()->clientTable);
    myItem->setText(Columns::ID, QString::number(ID));
    toClient = conn;

}


void soundClient::setClientInfo(const std::string &a, const std::string &u, const std::string &h, std::string &ip)
{
    application = a;
    user = u;
    host = h;
    IP = ip;
    myItem->setText(Columns::Application, application.c_str());
    myItem->setText(Columns::User, user.c_str());
    myItem->setText(Columns::Host, host.c_str());
    myItem->setText(Columns::IP, IP.c_str());
}

void soundClient::addSound(const std::string &fileName,size_t fileSize,time_t fileTime)
{
    sounds.push_back(new ClientSoundSample(fileName, fileSize, fileTime, this));
}

soundClient::~soundClient()
{
    delete clientSN;
    delete myItem;
    mainWindow::instance()->clients.remove(this);
    for (const auto& s : sounds)
    {
        delete s;
    }
    sounds.clear();
}

bool soundClient::send(covise::TokenBuffer& tb)
{
    covise::Message* m = new covise::Message(tb);
    m->type = covise::COVISE_MESSAGE_SOUND;
    if(toClient)
    return toClient->sendMessage(m);
}

covise::Message* soundClient::receiveMessage()
{
    covise::Message* m = new covise::Message();
    m->type = covise::COVISE_MESSAGE_QUIT;
    int res = toClient->recv_msg(m);
    return m;
}

unsigned int soundClient::IDCounter = 0;
