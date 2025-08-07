#ifndef COVISE_DAEMON_DEMO_H
#define COVISE_DAEMON_DEMO_H

#include <nlohmann/json.hpp>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <atomic>

#include <crow.h>

class DemoServer
{
public:
    void run();
    void stop();

private:
    std::atomic<bool> m_running{true};

    struct RunningProcess {
        std::atomic<int> pid{-1};
        std::string program;
        std::string headline;
        int id = -1;
    } m_runningProcess;

    nlohmann::json findDemoById(int id);
    int launchProcess(const std::string& program, const std::vector<std::string>& args);
    bool isPidRunning(int pid);
    bool terminateProcess(int pid);
    void monitorProcess(int pid, const std::string& appName);

    void setupRoutes(crow::SimpleApp& app);
};

#endif // COVISE_DAEMON_DEMO_H