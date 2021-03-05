#include "MERemotePartner.h"
#include "ui_MERemotePartner.h"

#include <QLabel>
#include <QCheckBox>
#include <QVBoxLayout>

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

    std::array<covise::LaunchStyle, 3> actions{covise::LaunchStyle::Partner, covise::LaunchStyle::Host, covise::LaunchStyle::Disconnect};
    for (auto style : actions)
    {
        auto box = m_clientActions.insert(std::make_pair(style, new QCheckBox(this))).first->second;
        box->setEnabled(enablePartnerObtions(partner.style, style));
        layout->addWidget(box);
    }
}

ClientWidgetList::ClientWidgetList(QScrollArea *scrollArea, QWidget *parent)
    : QWidget(parent), m_layout(new QVBoxLayout(this))
{
    m_layout->setDirection(QVBoxLayout::TopToBottom);
    scrollArea->setWidget(this);

    //auto headlineWidget = new QWidget(this);
    //headlineWidget->setMaximumHeight(60);
    //auto headlineLayout = new QVBoxLayout(headlineWidget);
    //headlineLayout->setDirection(QVBoxLayout::LeftToRight);
    //auto clients = new QLabel("Clients", headlineWidget);
    //headlineLayout->addWidget(clients);
    //auto partners = new QLabel("Partners", headlineWidget);
    //headlineLayout->addWidget(partners);
    //auto hosts = new QLabel("Hosts", headlineWidget);
    //headlineLayout->addWidget(hosts);
}

void ClientWidgetList::addClient(const covise::ClientInfo &partner)
{
    auto cw = new ClientWidget(partner, this);
    m_layout->addWidget(cw);
    removeClient(partner.id);
    m_clients[partner.id] = cw;
    for (const auto &clientAction : cw->m_clientActions)
    {
        connect(clientAction.second, &QCheckBox::stateChanged, this, [this, &clientAction]() {
            checkClientsSelected(clientAction.first);
        });
    }
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

void ClientWidgetList::checkClientsSelected(covise::LaunchStyle launchStyle)
{
    auto it = std::find_if(m_clients.begin(), m_clients.end(), [launchStyle](const std::pair<int, ClientWidget *> &cl) {
        return cl.second->m_clientActions[launchStyle]->isChecked();
    });
    emit atLeastOneClientSelected(launchStyle, it != m_clients.end());
}

MERemotePartner::MERemotePartner(QWidget *parent)
    : QDialog(parent), m_ui(new Ui::MERemotePartner)
{
    qRegisterMetaType<covise::LaunchStyle>();
    qRegisterMetaType<std::vector<int>>();

    m_ui->setupUi(this);
    m_actions[covise::LaunchStyle::Partner] = m_ui->addPartnersBtn;
    m_actions[covise::LaunchStyle::Host] = m_ui->addHostsBtn;
    m_actions[covise::LaunchStyle::Disconnect] = m_ui->disconnectBtn;
    m_clients = new ClientWidgetList(m_ui->partnersArea, this);
    for (const auto &action : m_actions)
    {
        action.second->setEnabled(false);
        connect(action.second, &QPushButton::clicked, this, [this, &action]() {
            emit takeAction(action.first, m_clients->getSelectedClients(action.first));
        });
    }
    connect(m_ui->cancelBtn, &QPushButton::clicked, this, [this]() {
        for (auto btn : m_actions)
        {
            btn.second->setEnabled(false);
        }
        this->hide();
    });
    connect(m_clients, &ClientWidgetList::atLeastOneClientSelected, this, [this](covise::LaunchStyle launchStyle, bool state) {
        m_actions[launchStyle]->setEnabled(state);
    });
}

void MERemotePartner::setPartners(const covise::ClientList &partners)
{
    for (const auto &action : m_actions)
    {
        action.second->setEnabled(false);
    }
    for (const auto &p : partners)
    {
        createParnerWidget(p);
    }
}

void MERemotePartner::createParnerWidget(const covise::ClientInfo &partner)
{
    m_clients->addClient(partner);
}
