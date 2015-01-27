/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_DAEMON_H
#define ME_DAEMON_H

#include <QObject>

class QString;
class QTreeWidgetItem;

namespace covise
{
class coRecvBuffer;
}

//================================================
class MEDaemon : public QObject
//================================================
{
    Q_OBJECT

public:
    MEDaemon(QString, QString);
    ~MEDaemon();

    QString &getShortname();
    QString &getUsername();
    QString &getHostname();

    void setVisible(int state);
    void addHostItems(covise::coRecvBuffer &);

    void setShortname(const QString &shortname);
    void setUsername(const QString &username);
    void setHostname(const QString &hostname);

    QTreeWidgetItem *getModuleRoot()
    {
        return m_modroot;
    };

private:
    QString m_hostname; // full hostname
    QString m_username; // user id, not used
    QString m_shortname; // short hostname
    QString m_text; // text in lists
    QTreeWidgetItem *m_modroot;
};
#endif
