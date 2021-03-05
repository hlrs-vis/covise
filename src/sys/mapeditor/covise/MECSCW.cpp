/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QComboBox>
#include <QGridLayout>
#include <QPushButton>
#include <QHBoxLayout>
#include <QLabel>
#include <QDialogButtonBox>
#include <QListWidget>
#if QT_VERSION >= 0x040400
#include <QFormLayout>
#endif

#include <config/CoviseConfig.h>
#include <covise/covise_msg.h>
#include <net/covise_host.h>

#include "MECSCW.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"

/*!
   \class MECSCW
   \brief This class select a host for adding
*/

MECSCW::MECSCW(QWidget *parent)
    : QDialog(parent)
{
    std::cerr << "MECSCW constructed" << std::endl;
    setWindowIcon(MEMainHandler::instance()->pm_logo);

    // read config file for available hosts

    int i = 0;
    covise::coCoviseConfig::ScopeEntries entries = covise::coCoviseConfig::getScopeEntries("System.HostConfig");
    const char **line = entries.getValue();
    if (line)
    {
        while (line[i] != NULL)
        {
            QString tmp = (char *)line[i];
            m_configHosts << tmp.section(":", -1);
            i = i + 2;
        }
    }

    // make the layout for the first dialog box

    QVBoxLayout *fbox = new QVBoxLayout(this);
    fbox->setMargin(2);
    fbox->setSpacing(2);

    QLabel *label = new QLabel(this);
    label->setText("Available hosts");
    label->setAlignment(Qt::AlignCenter);
    label->setFont(MEMainHandler::s_boldFont);
    fbox->addWidget(label);

    m_listbox = new QListWidget();
    m_listbox->setSelectionMode(QAbstractItemView::SingleSelection);
    m_listbox->setModelColumn(10);
    m_listbox->addItems(m_configHosts);
    connect(m_listbox, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(setHostCB(QListWidgetItem *)));
    connect(m_listbox, SIGNAL(itemDoubleClicked(QListWidgetItem *)), this, SLOT(accepted2(QListWidgetItem *)));
    fbox->addWidget(m_listbox);

    label = new QLabel(this);
    label->setText("Selected host");
    label->setAlignment(Qt::AlignCenter);
    label->setFont(MEMainHandler::s_boldFont);
    fbox->addWidget(label);

    m_selectionHost = new QLineEdit(this);
    m_selectionHost->setModified(true);
    connect(m_selectionHost, SIGNAL(returnPressed()), this, SLOT(accepted()));
    fbox->addWidget(m_selectionHost);

    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(bb, SIGNAL(rejected()), this, SLOT(rejected()));
    connect(bb, SIGNAL(accepted()), this, SLOT(accepted()));
    fbox->addWidget(bb);

    // set the logo
    setWindowIcon(MEMainHandler::instance()->pm_logo);
}

MECSCW::~MECSCW()
{
    std::cerr << "MECSCW destroyed" << std::endl;
}

void MECSCW::setVrbPartner(const QStringList &vrbPartner){
    m_listbox->clear();
    m_listbox->addItems(m_configHosts);
    m_listbox->addItems(vrbPartner);
}

//!
//! accept the host selection
//!
void MECSCW::accepted()
{
    // request host information
    QString tmp = "HOSTINFO\n" + m_selectionHost->text() + "\n";
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, tmp);

    // clear text field and hide widget
    m_selectionHost->clear();
    hide();
}

//!
//! accept the host selection
//!
void MECSCW::accepted2(QListWidgetItem *item)
{
    // request host information
    QString tmp = "HOSTINFO\n" + item->text() + "\n";
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, tmp);

    hide();
}

//!
//! reject the host delection
//!
void MECSCW::rejected()
{
    m_selectionHost->clear();
    hide();
}

//!
//! set the selected host in textfield
//!
void MECSCW::setHostCB(QListWidgetItem *item)
{
    if (item)
        m_selectionHost->setText(item->text());
}

#if QT_VERSION >= 0x040400

#define makeLine(text, widget, type)           \
    label = new QLabel(text);                  \
    label->setFont(MEMainHandler::s_boldFont); \
    widget = new type();                       \
    grid->addRow(label, widget);

#else

#define makeLine(text, widget, type)                                     \
    label = new QLabel(text);                                            \
    label->setFont(MEMainHandler::s_boldFont);                           \
    widget = new type();                                                 \
    grid->addWidget(label, index, 0, Qt::AlignVCenter | Qt::AlignRight); \
    grid->addWidget(widget, index, 1);                                   \
    ++index;

#endif

/*!
   \class MECSCWParam
   \brief This class handles the parameter for adding a host or partner
*/

MECSCWParam::MECSCWParam(QWidget *parent)
    : QDialog(parent)
{

    // set the logo
    setWindowIcon(MEMainHandler::instance()->pm_logo);
    setWindowTitle(MEMainHandler::instance()->generateTitle("Connection Parameters"));

    QVBoxLayout *fbox = new QVBoxLayout(this);

    // make main content
    QLabel *label;
    QStringList text;
    text << "rexec"
         << "rsh"
         << "ssh"
         << "nqs"
         << "manual"
         << "remoteDaemon"
         << "SSL"
         << "VRB";

#if QT_VERSION >= 0x040400
    QFormLayout *grid = new QFormLayout();
#else
    QGridLayout *grid = new QGridLayout();
    int index = 0;
#endif

    makeLine("Hostname", hostname, QLineEdit);
    hostname->setText(MEMainHandler::instance()->localHost);
    makeLine("Username", username, QLineEdit);
    username->setText(MEMainHandler::instance()->localUser);
    makeLine("ExecModes", connectionModes, QComboBox);
    connectionModes->addItems(text);
#ifdef WIN32
    setMode(5);
#else
    setMode(2);
#endif
    makeLine("Password", passwd, QLineEdit);
    passwd->setEchoMode(QLineEdit::Password);
    makeLine("Display", display, QLineEdit);
    makeLine("Timeout", timeout, QLineEdit);

    fbox->addLayout(grid);

    // add dialog buttons
    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(bb, SIGNAL(rejected()), this, SLOT(rejected()));
    connect(bb, SIGNAL(accepted()), this, SLOT(accepted()));
    fbox->addWidget(bb);
}

MECSCWParam::~MECSCWParam()
{
}

//!
//! set the timeout
//!
void MECSCWParam::setTimeout(QString time)
{
    timeout->setText(time);
}

//!
//! set the host delection
//!
void MECSCWParam::setMode(int index)
{
    connectionModes->setCurrentIndex(index);
}

//!
//! set the host delection
//!
void MECSCWParam::setHost(QString host)
{
    hostname->setText(host);
}

//!
//! set the user delection
//!
void MECSCWParam::setUser(QString user)
{
    username->setText(user);
}

//!
//! set the host selection
//!
void MECSCWParam::accepted()
{
    QStringList list;



    std::string ip = covise::Host::lookupIpAddress(hostname->text().toLatin1().data());
    list << ip.c_str() << username->text();

    list << passwd->text();

    int curr = connectionModes->currentIndex();
    QString mm;
    mm.setNum(curr + 1);
    list << mm;

    if (timeout->text().length() == 0)
        list << "5";
    else
        list << timeout->text();

    list << display->text();

    QString tmp = list.join("\n");

    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, tmp);

    hide();
}

//!
//! reject the host delection
//!
void MECSCWParam::rejected()
{
    hide();
}

//!
//! set the default parameter if a ADDHOST from reading a map happens
//!
void MECSCWParam::setDefaults(QString name)
{
    QStringList item;
    QString hostname, text;

    // get the hostname for a given IP address

    hostname = QString::fromStdString(covise::Host::lookupHostname(name.toLatin1().data()));

    // read line in covise.config
    // look for hostname or ip address

    std::string line = covise::coCoviseConfig::getEntry("System.HostConfig." + std::string(hostname.toLatin1().data()));

    if (line.empty())
    {
        line = covise::coCoviseConfig::getEntry("System.HostConfig." + std::string(name.toLatin1().data()));
        if (line.empty())
        {
            //hostname might be a hostname with domain, so try it without the domain
            // s == "myapp"
            QString shortName = hostname.section('.', 0, 0);
            line = covise::coCoviseConfig::getEntry("System.HostConfig." + std::string(shortName.toLatin1().data()));
        }
    }
    if (!line.empty())
    {
        text = line.c_str();
        item = text.split(" ", QString::SkipEmptyParts);
        connectionModes->setCurrentIndex(connectionModes->findText(item[1]));
        timeout->setText(item[2]);
    }
}
