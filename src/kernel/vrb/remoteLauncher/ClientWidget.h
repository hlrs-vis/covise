#ifndef VRB_REMOTE_LAUNCHER_CLIENT_WIDGET_H
#define VRB_REMOTE_LAUNCHER_CLIENT_WIDGET_H

#include "MessageTypes.h"

#include "export.h"

#include <QWidget>
#include <QScrollArea>




class QVBoxLayout;
namespace vrb{  
namespace launcher
{
class REMOTELAUNCHER_EXPORT ClientWidget : public QWidget
{
    Q_OBJECT
public:
    ClientWidget(int clientID, const QString &clientInfo, QWidget *parent);

signals:
    void requestProgramLaunch(Program programID, int clientID);

private:
    int m_clientID;
};

class REMOTELAUNCHER_EXPORT ClientWidgetList : public QWidget
{
    Q_OBJECT
public:
    ClientWidgetList(QScrollArea *scrollArea, QWidget *parent);
    void addClient(int clientID, const QString &clientInfo);
    void removeClient(int clientID);
signals:
    void requestProgramLaunch(Program programID, int clientID);

private:
    QVBoxLayout *m_layout = nullptr;
    QScrollArea *m_scrollArea = nullptr;
    std::map<int, ClientWidget *> m_clients;
};
} // namespace launcher
}   //vrb    


#endif