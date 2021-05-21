/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef COVISE_DAEMON_CHILD_PROCESS_h
#define COVISE_DAEMON_CHILD_PROCESS_h



#include <QObject>
#include <QSocketNotifier>
#include <QString>

#include <memory>
#include <string>
#include <vector>

#include <util/coSignal.h>

#ifdef WIN32
#include <QThread>
#include <windows.h> 
struct  ProcessThread : QThread
{

    Q_OBJECT
public: 
    ProcessThread(const std::vector<std::string>& args, QObject* parent);
    void run() override;
signals:
    void output(const QString& msg);
    void died();
public slots:
    void terminate();
private:
    std::atomic_bool m_terminate{ false };
    const std::vector<std::string> m_args;
    HANDLE readHandle = NULL;
    HANDLE writeHandle = NULL;
    HANDLE terminateHandle = NULL;
    void ReadFromPipe();


};
#endif

class SigChildHandler : public QObject, public covise::coSignalHandler
{
    Q_OBJECT
public:
signals:
    void childDied(int pid);

private:
    virtual void sigHandler(int sigNo); //  catch SIGTERM

    virtual const char *sigHandlerName();
};

struct ChildProcess : QObject
{
    Q_OBJECT
public:
    ChildProcess(const char *path, const std::vector<std::string> &args);
    ChildProcess(const ChildProcess &) = delete;
    const ChildProcess &operator=(const ChildProcess &) = delete;
    ~ChildProcess();
    bool operator<(const ChildProcess &other) const;
    bool operator==(const ChildProcess& other) const;


signals:
    void died();
    void output(const QString &msg);
    void destructor();
private:
    int m_pid = 0;
    std::unique_ptr<QSocketNotifier> m_outputNotifier;
#ifdef WIN32
    void createWindowsProcess(const std::vector<std::string>& args);
#endif

};
#endif // !COVISE_DAEMON_CHILD_PROCESS_h
