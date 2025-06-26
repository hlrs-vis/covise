#include "demoserver.h"
#include <demo.h>

#include <fstream>
#include <thread>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

DemoServer::DemoServer()
{
    m_apps = {
        {"covise", "covise"},
        {"OpenCOVER", "OpenCOVER"},
        {"opencover", "OpenCOVER"},
        {"Vistle", "vistle"},
        {"vistle", "vistle"},
        {"sumo", "sumo"},
        {"notepad", "notepad"}};
}

json DemoServer::loadDemos()
{
    std::ifstream f(demo::collection);
    if (!f)
        return json::object();
    json j;
    f >> j;
    return j;
}

json DemoServer::findDemoById(int id)
{
    json demos_by_category = loadDemos();
    for (auto &[category, demo_list] : demos_by_category.items())
    {
        for (auto &demo : demo_list)
        {
            if (demo["id"] == id)
                return demo;
        }
    }
    return nullptr;
}

int DemoServer::launchProcess(const std::string &program, const std::vector<std::string> &args)
{
#ifdef _WIN32
    std::string cmd = program;
    for (const auto &arg : args)
        cmd += " " + arg;
    STARTUPINFOA si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    if (CreateProcessA(NULL, (LPSTR)cmd.c_str(), NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &pi))
    {
        CloseHandle(pi.hThread);
        int pid = (int)pi.dwProcessId;
        CloseHandle(pi.hProcess);
        return pid;
    }
    return -1;
#else
    pid_t pid = fork();
    if (pid == 0)
    {
        std::vector<char *> argv;
        argv.push_back(const_cast<char *>(program.c_str()));
        for (const auto &arg : args)
            argv.push_back(const_cast<char *>(arg.c_str()));
        argv.push_back(nullptr);
        execvp(program.c_str(), argv.data());
        std::exit(1);
    }
    return pid;
#endif
}

bool DemoServer::isPidRunning(int pid)
{
#ifdef _WIN32
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
    DWORD ret = WaitForSingleObject(process, 0);
    CloseHandle(process);
    return ret == WAIT_TIMEOUT;
#else
    return (kill(pid, 0) == 0);
#endif
}

bool DemoServer::terminateProcess(int pid)
{
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    if (!h)
        return false;
    BOOL ok = TerminateProcess(h, 0);
    CloseHandle(h);
    return ok;
#else
    return kill(pid, SIGTERM) == 0;
#endif
}

void DemoServer::monitorProcess(int pid, const std::string &appName)
{
#ifdef _WIN32
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (h)
    {
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
    }
#else
    int status;
    waitpid(pid, &status, 0);
#endif
    m_runningProcess.pid = -1;
    m_runningProcess.program.clear();
    m_runningProcess.headline.clear();
    m_runningProcess.id = -1;
}

void DemoServer::setupRoutes(crow::SimpleApp &app)
{
    // GET /demos
    CROW_ROUTE(app, "/demos").methods("GET"_method)([this]()
                                                    {
        json demos_by_category = loadDemos();
        json demos = json::array();
        std::set<int> seen_ids;
        for (auto& [category, demo_list] : demos_by_category.items()) {
            for (auto& demo : demo_list) {
                demo["category"] = category;
                if (demo.contains("id") && !seen_ids.count(demo["id"].get<int>())) {
                    seen_ids.insert(demo["id"].get<int>());
                    demos.push_back(demo);
                }
            }
        }
        json result;
        result["demos"] = demos;
        std::vector<std::string> categories;
        for (auto& [cat, _] : demos_by_category.items())
            categories.push_back(cat);
        result["categories"] = categories;
        return crow::response(result.dump()); });

    // POST /launch_demo
    CROW_ROUTE(app, "/launch_demo").methods("POST"_method)([this](const crow::request &req)
                                                           {
        auto data = json::parse(req.body, nullptr, false);
        if (!data.is_object() || !data.contains("id"))
            return crow::response(400, R"({"status":"error","message":"Missing id"})");

        int demo_id = data["id"].get<int>();
        json demo = findDemoById(demo_id);
        if (demo.is_null())
            return crow::response(404, R"({"status":"error","message":"Demo not found"})");

        // Log launch event
        std::ofstream logf(demo::logFile, std::ios::app);
        json log_entry = { {"timestamp", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())}, {"demo_id", demo_id} };
        logf << log_entry.dump() << "\n";

        // Use first launch entry
        if (!demo.contains("launch") || !demo["launch"].is_array() || demo["launch"].empty())
            return crow::response(400, R"({"status":"error","message":"No launch entry"})");
        auto entry = demo["launch"][0];
        std::string program = entry["program"];
        std::vector<std::string> args;
        if (entry.contains("args"))
            for (auto& a : entry["args"]) args.push_back(a);

        int pid = launchProcess(program, args);
        if (pid < 0)
            return crow::response(500, R"({"status":"error","message":"Failed to launch"})");

        m_runningProcess.pid = pid;
        m_runningProcess.program = program;
        m_runningProcess.headline = demo.value("headline", "");
        m_runningProcess.id = demo_id;

        std::thread([this, pid, program]() { monitorProcess(pid, program); }).detach();

        return crow::response(R"({"status":"success"})"); });

    // GET /running_process
    CROW_ROUTE(app, "/running_process").methods("GET"_method)([this]()
                                                              {
        json resp;
        if (m_runningProcess.pid > 0 && isPidRunning(m_runningProcess.pid)) {
            resp["running"] = true;
            resp["program"] = m_runningProcess.program;
            resp["headline"] = m_runningProcess.headline;
            resp["id"] = m_runningProcess.id;
        } else {
            resp["running"] = false;
        }
        return crow::response(resp.dump()); });

    // POST /terminate_process
    CROW_ROUTE(app, "/terminate_process").methods("POST"_method)([this](const crow::request &)
                                                                 {
        if (m_runningProcess.pid > 0 && isPidRunning(m_runningProcess.pid)) {
            if (terminateProcess(m_runningProcess.pid)) {
                m_runningProcess.pid = -1;
                m_runningProcess.program.clear();
                m_runningProcess.headline.clear();
                m_runningProcess.id = -1;
                return crow::response(R"({"status":"terminated"})");
            } else {
                return crow::response(500, R"({"status":"error","message":"Failed to terminate"})");
            }
        }
        return crow::response(R"({"status":"no_process"})"); });

    // GET /apps
    CROW_ROUTE(app, "/apps").methods("GET"_method)([this]()
                                                   {
        json apps;
        for (const auto& [k, _] : m_apps)
            apps["apps"].push_back(k);
        return crow::response(apps.dump()); });

    // POST /update_description
    CROW_ROUTE(app, "/update_description").methods("POST"_method)([this](const crow::request &req)
                                                                  {
        auto data = json::parse(req.body, nullptr, false);
        if (!data.is_object() || !data.contains("id") || !data.contains("description"))
            return crow::response(400, R"({"status":"error","message":"Missing id or description"})");

        int demo_id = data["id"].get<int>();
        std::string new_desc = data["description"];
        json demos_by_category = loadDemos();

        bool updated = false;
        for (auto& [cat, demo_list] : demos_by_category.items()) {
            for (auto& demo : demo_list) {
                if (demo["id"] == demo_id) {
                    demo["description"] = new_desc;
                    updated = true;
                    break;
                }
            }
        }
        if (updated) {
            std::ofstream f(demo::collection);
            f << demos_by_category.dump(2);
            return crow::response(R"({"status":"success"})");
        } else {
            return crow::response(404, R"({"status":"error","message":"Demo not found"})");
        } });

    // GET /hot_demos
    CROW_ROUTE(app, "/hot_demos").methods("GET"_method)([this]()
                                                        {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto cutoff = now - hours(24 * 30);

        std::map<int, int> counts;
        std::ifstream logf(demo::logFile);
        std::string line;
        while (std::getline(logf, line)) {
            try {
                auto entry = json::parse(line);
                auto ts = system_clock::from_time_t(entry["timestamp"].get<time_t>());
                if (ts >= cutoff) {
                    int demo_id = entry["demo_id"];
                    counts[demo_id]++;
                }
            } catch (...) {}
        }

        json demos_by_category = loadDemos();
        std::vector<json> all_demos;
        for (auto& [cat, demo_list] : demos_by_category.items())
            for (auto& demo : demo_list)
                all_demos.push_back(demo);

        json hot = json::array();
        for (auto& demo : all_demos) {
            int id = demo["id"];
            if (counts.count(id)) {
                demo["launch_count"] = counts[id];
                hot.push_back(demo);
            }
        }
        std::sort(hot.begin(), hot.end(), [](const json& a, const json& b) {
            return a["launch_count"].get<int>() > b["launch_count"].get<int>();
        });
        return crow::response(json{{"hot", hot}}.dump()); });

    // Serve static files
    CROW_ROUTE(app, "/<path>")
    .name("static_files")
    ([this](const crow::request &req, std::string path)
     {
        auto p = (fs::path(demo::root) / path);
        std::ifstream file(p, std::ios::binary);
        if (!file)
            return crow::response(404);
        std::ostringstream contents;
        contents << file.rdbuf();
        return crow::response(contents.str()); });

    CROW_ROUTE(app, "/")
    ([this]()
     {
    std::ifstream file(demo::indexHtml);
    if (!file)
        return crow::response(404);
    std::ostringstream contents;
    contents << file.rdbuf();
    return crow::response(contents.str()); });
}

void DemoServer::run() {
    crow::SimpleApp app;
    setupRoutes(app);
    // Run the server in a way that allows stopping
    auto server_future = std::async(std::launch::async, [&](){
        app.port(demo::port).multithreaded().run();
    });
    while (m_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    app.stop(); // Stop Crow server
}

void DemoServer::stop() {
    m_running = false;
}