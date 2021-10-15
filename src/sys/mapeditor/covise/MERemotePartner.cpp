#include "MERemotePartner.h"
#include "ui_MERemotePartner.h"

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QProgressBar>
#include <sstream>

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

    std::stringstream ss;
    ss << partner.userName << "@" << partner.hostName <<
    ", id: " << partner.id;

    auto *layout = new QVBoxLayout(this);
    layout->setDirection(QVBoxLayout::LeftToRight);
    auto *content = new QLabel(ss.str().c_str(), this);
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
    else if(partner.style == LaunchStyle::Pending)
    {
        auto progress = new QProgressBar(this);
        progress->setMinimum(0);
        progress->setMaximum(0);
        progress->setToolTip("Waiting for connection");
        layout->addWidget(progress);
        
        auto cancelBtn = new QPushButton(this);
        cancelBtn->setText("Cancel");
        layout->addWidget(cancelBtn);
        connect(cancelBtn, &QPushButton::clicked, this, [this, &partner]() {
            ClientInfo i = partner;
            i.style = LaunchStyle::Disconnect;
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

ClientWidgetList::ClientWidgetList(QScrollArea *scrollArea, QWidget *parent, const covise::ClientList &partners)
    : QWidget(parent), m_layout(new QVBoxLayout(this))
{
    m_layout->setDirection(QVBoxLayout::TopToBottom);
    scrollArea->setWidget(this);
    for (const auto &partner : partners)
    {
        auto cw = new ClientWidget(partner, this);
        m_layout->addWidget(cw);
        m_clients[partner.id] = cw;
        connect(cw, &ClientWidget::clientAction, this, &ClientWidgetList::clientAction);
    }
}

MERemotePartner::MERemotePartner(QWidget *parent)
    : QDialog(parent), m_ui(new Ui::MERemotePartner), m_parent(parent)
{

    if (parent)
    {
        move(parent->rect().center() - rect().center());
    }
    qRegisterMetaType<covise::LaunchStyle>();
    qRegisterMetaType<std::vector<int>>();
    qRegisterMetaType<covise::ClientInfo>();

    m_ui->setupUi(this);

}

void MERemotePartner::setPartners(const covise::ClientList &partners)
{
    if (m_parent)
    {
        move(m_parent->pos() + QPoint{m_parent->rect().width() / 2, m_parent->rect().height() / 2} - rect().center());
    }
    m_clients = new ClientWidgetList(m_ui->partnersArea, this, partners);
    connect(m_clients, &ClientWidgetList::clientAction, this, &MERemotePartner::clientAction);
    connect(m_ui->cancelBtn, &QPushButton::clicked, this, [this]() {
        hide();
    });
}
