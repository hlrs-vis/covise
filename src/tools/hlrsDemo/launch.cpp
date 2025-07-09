#include "launch.h"
#include <QMessageBox>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#endif


DemoLauncher::~DemoLauncher()
{
    terminateAll();
}

bool DemoLauncher::launchDemo(const nlohmann::json &demo)
{
    std::vector<qint64> startedPids;
    auto id = demo.value("id", 0);

    for (const auto &launch : demo["launch"])
    {
        auto program = launch.value("program", "");
        std::vector<std::string> args = launch.value("args", nlohmann::json::array());
        QStringList qArgs;
        for(const auto &arg : args)
            qArgs << QString::fromStdString(arg);
        if (program.empty())
        {
            QMessageBox::warning(nullptr, "Error", "No program specified for this demo.");
            return false;
        }
        auto proc = std::make_unique<QProcess>();
        proc->setProgram(program.c_str());
        proc->setArguments(qArgs);
        proc->start();
        qint64 pid = proc->processId();
        if (pid > 0)
        {
            runningProcesses[id].push_back(std::move(proc));
            startedPids.push_back(pid);
        }
        else
        {
            QMessageBox::critical(nullptr, "Error", "Failed to start: " + QString::fromStdString(program));
            terminateDemo(id);
            return false;
        }
    }
    return true;
}

void terminate(int pid)
{
#ifdef _WIN32
    HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, FALSE, static_cast<DWORD>(pid));
    if (hProcess)
    {
        TerminateProcess(hProcess, 0);
        CloseHandle(hProcess);
    }
#else
    ::kill(pid, SIGTERM);
#endif
}

void DemoLauncher::terminateDemo(int id)
{
    auto it = runningProcesses.find(id);
    if (it != runningProcesses.end())
    {
        for (auto &proc : it->second)
        {
            proc->terminate();
            // terminate(proc->processId());
        }
        runningProcesses.erase(it);
    }
}

void DemoLauncher::terminateAll()
{
    for (auto &pair : runningProcesses)
    {
        for (auto &proc : pair.second)
        {
            terminate(proc->processId());
        }
    }
    runningProcesses.clear();
}