#ifndef MAPEDITOR_REMOTE_PARTNER_H
#define MAPEDITOR_REMOTE_PARTNER_H

#include <QWidget>
#include <QDialog>
#include <QString>
#include <comsg/coviseLaunchOptions.h>
#include <comsg/NEW_UI.h>
namespace Ui{
    class MERemotePartner;
}
class QScrollArea;
class QPushButton;
class QVBoxLayout;


Q_DECLARE_METATYPE(covise::LaunchStyle);
Q_DECLARE_METATYPE(covise::ClientInfo);
Q_DECLARE_METATYPE(std::vector<int>);

class ClientWidget : public QWidget
{
    Q_OBJECT
public:
    ClientWidget(const covise::ClientInfo &partner, QWidget *parent);

    std::map<covise::LaunchStyle, QPushButton *> m_clientActions;
signals:
    void clientAction(covise::ClientInfo client);

private:
    const covise::ClientInfo m_partner;
};

class ClientWidgetList : public QWidget
{
    Q_OBJECT
public:
    ClientWidgetList(QScrollArea *scrollArea, QWidget *parent);
    void addClient(const covise::ClientInfo& partner);
    void removeClient(int clientID);
    std::vector<int> getSelectedClients(covise::LaunchStyle launchStyle);
signals:
    void clientAction(const covise::ClientInfo &client);

private:
    QVBoxLayout *m_layout = nullptr;
    QScrollArea *m_scrollArea = nullptr;
    std::map<int, ClientWidget *> m_clients;
};


class MERemotePartner : public QDialog
{
    Q_OBJECT 
public:
    explicit MERemotePartner(QWidget *parent = nullptr);
    void setPartners(const covise::ClientList &partners);
signals:
    void clientAction(const covise::ClientInfo &client);


private:
    const QWidget *m_parent = nullptr;
    Ui::MERemotePartner *m_ui;
    ClientWidgetList *m_clients;

};

#endif