#include "MERemotePartner.h"
#include "ui_MERemotePartner.h"

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

using namespace covise;
bool enablePartnerObtions(covise::LaunchStyle partnerStyle, covise::LaunchStyle action) //enable disconnect of host or partner, enable partner + host if disconnect
{
    if (action == covise::LaunchStyle::Disconnect)
    {
        return partnerStyle != covise::LaunchStyle::Disconnect;
    }
    return partnerStyle == covise::LaunchStyle::Disconnect;
}

ClientWidget::ClientWidget(const covise::ClientInfo &partner, QWidget *parent)
    : QWidget(parent), m_partner(partner)
{

    this->setMaximumHeight(60);
    QString s = partner.hostName.c_str();
    s += ", id:" + QString::number(partner.id);

    auto *layout = new QVBoxLayout(this);
    layout->setDirection(QVBoxLayout::LeftToRight);
    auto *content = new QLabel(s, this);
    layout->addWidget(content, 0, Qt::AlignTop);

    if (partner.style == LaunchStyle::Disconnect)
    {
        auto partnerBtn = new QPushButton(this);
        partnerBtn->setText("Add partner");
        layout->addWidget(partnerBtn);
        connect(partnerBtn, &QPushButton::clicked, this, [this, &partner]() {
            ClientInfo i = partner;
            i.style = LaunchStyle::Partner;
            emit clientAction(i);
        });

        auto hostBtn = new QPushButton(this);
        hostBtn->setText("Add host");
        layout->addWidget(hostBtn);
        connect(hostBtn, &QPushButton::clicked, this, [this, &partner]() {
            ClientInfo i = partner;
            i.style = LaunchStyle::Host;
            emit clientAction(i);
        });
    }
    else
    {
        QString btnText;
        if (partner.style == LaunchStyle::Host)
        {
            btnText = "Disconnect host";
        }
        else if (partner.style == LaunchStyle::Partner)
        {
            btnText = "Disconnect partner";
        }
        auto btn = new QPushButton(this);
        btn->setText(btnText);
        layout->addWidget(btn);
        connect(btn, &QPushButton::clicked, this, [this, &partner]() {
            ClientInfo i = partner;
            i.style = LaunchStyle::Disconnect;
            emit clientAction(i);
        });
    }
}

ClientWidgetList::ClientWidgetList(QScrollArea *scrollArea, QWidget *parent)
    : QWidget(parent), m_layout(new QVBoxLayout(this))
{
    m_layout->setDirection(QVBoxLayout::TopToBottom);
    scrollArea->setWidget(this);
}

void ClientWidgetList::addClient(const covise::ClientInfo &partner)
{
    auto cw = new ClientWidget(partner, this);
    m_layout->addWidget(cw);
    removeClient(partner.id);
    m_clients[partner.id] = cw;
    connect(cw, &ClientWidget::clientAction, this, &ClientWidgetList::clientAction);
}

void ClientWidgetList::removeClient(int clientID)
{
    auto cl = m_clients.find(clientID);
    if (cl != m_clients.end())
    {
        m_layout->removeWidget(cl->second);
        delete cl->second;
        cl->second = nullptr;
        m_clients.erase(cl);
    }
}

std::vector<int> ClientWidgetList::getSelectedClients(covise::LaunchStyle launchStyle)
{
    std::vector<int> retval;
    for (const auto &cl : m_clients)
    {
        if (cl.second->m_clientActions[launchStyle]->isChecked())
        {
            retval.push_back(cl.first);
        }
    }
    return retval;
}

MERemotePartner::MERemotePartner(QWidget *parent)
    : QDialog(parent), m_ui(new Ui::MERemotePartner)
{
    qRegisterMetaType<covise::LaunchStyle>();
    qRegisterMetaType<std::vector<int>>();
    qRegisterMetaType<covise::ClientInfo>();

    m_ui->setupUi(this);
    m_clients = new ClientWidgetList(m_ui->partnersArea, this);
    connect(m_clients, &ClientWidgetList::clientAction, this, &MERemotePartner::clientAction);

    connect(m_ui->cancelBtn, &QPushButton::clicked, this, [this]() {
        hide();
    });
}

void MERemotePartner::setPartners(const covise::ClientList &partners)
{
    for (const auto &p : partners)
    {
        m_clients->addClient(p);
    }
}
