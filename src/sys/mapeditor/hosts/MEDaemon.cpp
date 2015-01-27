/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QTreeWidgetItem>

#include "hosts/MEDaemon.h"
#include "widgets/MEModuleTree.h"
#include "widgets/MEUserInterface.h"

/*****************************************************************************
 *
 * Class coDeamon
 *
 *****************************************************************************/

MEDaemon::MEDaemon(QString name, QString user)
    : QObject()
    , m_hostname(name)
    , m_username(user)
{

    // get the "short" name, name without domain suffiy
    m_shortname = m_hostname.section(".", 0, 0);

    // make name for lists
    m_text = m_username;
    m_text.append("@");
    m_text.append(m_shortname);

    // make root entry in module tree
    m_modroot = new QTreeWidgetItem(MEUserInterface::instance()->getModuleTree());
    m_modroot->setText(0, m_text);
    m_modroot->setExpanded(false);
}

MEDaemon::~MEDaemon()
{
    delete m_modroot;
}

void MEDaemon::setVisible(int state)
{
    m_modroot->setHidden(!state);
}

QString &MEDaemon::getShortname()
{
    return m_shortname;
}

void MEDaemon::setShortname(const QString &paramShortname)
{
    m_shortname = paramShortname;
}

QString &MEDaemon::getUsername()
{
    return m_username;
}

void MEDaemon::setUsername(const QString &paramUsername)
{
    m_username = paramUsername;
}

QString &MEDaemon::getHostname()
{
    return m_hostname;
}

void MEDaemon::setHostname(const QString &paramHostname)
{
    m_hostname = paramHostname;
}
