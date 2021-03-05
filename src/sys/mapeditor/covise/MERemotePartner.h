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
class QCheckBox;
class QVBoxLayout;


Q_DECLARE_METATYPE(covise::LaunchStyle);
Q_DECLARE_METATYPE(std::vector<int>);

class ClientWidget : public QWidget
{
    Q_OBJECT
public:
    ClientWidget(const covise::ClientInfo &partner, QWidget *parent);

    std::map<covise::LaunchStyle, QCheckBox *> m_clientActions;
signals:
    void partnerSelected(bool selected);

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
    void atLeastOneClientSelected(covise::LaunchStyle launchStyle, bool state);
private slots:
    void checkClientsSelected(covise::LaunchStyle launchStyle);

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
    void takeAction(covise::LaunchStyle launchStyle, std::vector<int> clientIds);

private:
    Ui::MERemotePartner *m_ui;
    ClientWidgetList *m_clients;
    std::map<covise::LaunchStyle, QPushButton *> m_actions;

    void createParnerWidget(const covise::ClientInfo& partner);
};

#endif