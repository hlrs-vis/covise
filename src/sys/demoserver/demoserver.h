#ifndef COVISE_DAEMON_DEMO_H
#define COVISE_DAEMON_DEMO_H

#include <nlohmann/json.hpp>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <atomic>
#include <mutex>

#include <crow.h>

class DemoServer
{
public:
    void run();
    void stop();

private:
    std::atomic<bool> m_running{true};

    struct RunningDemo {
        std::vector<int> pids;
        std::string program;
        std::string headline;
        int id = -1;
        bool toTerminate = false;
        bool terminated = false;
        std::chrono::steady_clock::time_point terminationTime;
        // termination notification
        bool terminationNotificationPending = false; // set true when all processes terminated
    } m_runningDemo;

    nlohmann::json findDemoById(int id);
    int launchProcess(const std::string& program, const std::vector<std::string>& args);
    void setupRoutes(crow::SimpleApp& app);
    std::unique_ptr<crow::SimpleApp> m_app;
};

#endif // COVISE_DAEMON_DEMO_H