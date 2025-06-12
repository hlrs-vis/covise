#pragma once

#include <QString>
#include <QStringList>
#include <QWidget>
#include <QProcess>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
class DemoLauncher : public QObject
{
public:
    DemoLauncher() = default;
    ~DemoLauncher();

    // Launches the demo, returns vector of started PIDs
    bool launchDemo(const nlohmann::json &demo);

    // Terminates all running demos for a widget
    void terminateDemo(int id);

private:
    std::map<int, std::vector<std::unique_ptr<QProcess>>> runningProcesses;


    void terminateAll();
};