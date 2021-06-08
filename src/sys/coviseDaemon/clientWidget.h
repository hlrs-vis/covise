/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_REMOTE_LAUNCHER_CLIENT_WIDGET_H
#define VRB_REMOTE_LAUNCHER_CLIENT_WIDGET_H

#include "metaTypes.h"

#include <vrb/ProgramType.h>

#include <QWidget>
#include <QScrollArea>


class QVBoxLayout;

class ClientWidget : public QWidget
{
    Q_OBJECT
public:
    ClientWidget(int clientID, const QString &clientInfo, QWidget *parent);

signals:
    void requestProgramLaunch(vrb::Program programID, int clientID);

private:
    int m_clientID;
};

class ClientWidgetList : public QWidget
{
    Q_OBJECT
public:
    ClientWidgetList(QScrollArea *scrollArea, QWidget *parent);
    void addClient(int clientID, const QString &clientInfo);
    void removeClient(int clientID);
    void clear();
signals:
    void requestProgramLaunch(vrb::Program programID, int clientID);

private:
    QVBoxLayout *m_layout = nullptr;
    QScrollArea *m_scrollArea = nullptr;
    std::map<int, ClientWidget *> m_clients;
};
  

#endif