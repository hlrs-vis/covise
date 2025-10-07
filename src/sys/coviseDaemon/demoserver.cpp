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


json loadDemos()
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
        setenv("COVER_TERMINATE_SESSION", "1", 1); // Used for demos
        execvp(program.c_str(), argv.data());
        std::exit(1);
    }
    return pid;
#endif
}

bool isPidRunning(int pid)
{
#ifdef _WIN32
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
    DWORD ret = WaitForSingleObject(process, 0);
    CloseHandle(process);
    return ret == WAIT_TIMEOUT;
#else
    int status; 
    auto r = waitpid(pid, &status, WNOHANG); // Reap zombie processes
    if (r == pid) {
        // child was reaped -> not running anymore
        return false;
    }
    // r == 0 => still running, r == -1 => error (treat as not running if ESRCH)
    if (r == -1) {
        return false; // no such process
    }    
    return (kill(pid, 0) == 0);
#endif
}

bool isAnyPidRunning(const std::vector<int> &pids)  // Changed parameter type
{
    for (const auto &pid : pids)
    {
        if (isPidRunning(pid))  // No need for .load() anymore
            return true;
    }
    return false;
}

bool terminateProcess(int pid, bool force = false)
{
    #ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    if (!h)
    return false;
    BOOL ok = TerminateProcess(h, 0);
    CloseHandle(h);
    return ok;
    #else
    return kill(pid, force ? SIGKILL : SIGTERM) == 0;
    #endif
}

bool terminateProcesses(const std::vector<int> &pids, bool force = false)  // Changed parameter type
{
    bool retval = true;
    for (const auto &pid : pids)
    {
        if (!terminateProcess(pid, force))
            retval = false;
    }
    return retval;
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

    // Check if launch entries exist
    if (!demo.contains("launch") || !demo["launch"].is_array() || demo["launch"].empty())
        return crow::response(400, R"({"status":"error","message":"No launch entry"})");

    // Clear previous running demo info
    m_runningDemo.pids.clear();
    m_runningDemo.program.clear();
    m_runningDemo.headline = demo.value("headline", "");
    m_runningDemo.id = demo_id;

    // Launch ALL entries in the launch array
    std::vector<std::string> launched_programs;
    bool any_failed = false;
    
    for (const auto& entry : demo["launch"]) {
        if (!entry.contains("program")) {
            std::cerr << "Warning: Launch entry missing 'program' field" << std::endl;
            continue;
        }
        
        std::string program = entry["program"];
        std::vector<std::string> args;
        
        if (entry.contains("args")) {
            for (const auto& a : entry["args"]) {
                args.push_back(a);
            }
        }
        
        int pid = launchProcess(program, args);
        if (pid < 0) {
            std::cerr << "Failed to launch: " << program << std::endl;
            any_failed = true;
        } else {
            m_runningDemo.pids.emplace_back(pid);
            launched_programs.push_back(program);
            std::cerr << "Launched " << program << " with PID " << pid << std::endl;
        }
    }
    
    if (m_runningDemo.pids.empty()) {
        return crow::response(500, R"({"status":"error","message":"Failed to launch any process"})");
    }
    
    // Set the program name to a combination of all launched programs
    m_runningDemo.program = launched_programs.empty() ? "Unknown" : 
                           (launched_programs.size() == 1 ? launched_programs[0] : 
                            "Multiple (" + std::to_string(launched_programs.size()) + " processes)");


    if (any_failed) {
        return crow::response(200, R"({"status":"partial_success","message":"Some processes failed to launch"})");
    } else {
        return crow::response(R"({"status":"success"})");
    }
});

    // GET /running_process
    CROW_ROUTE(app, "/running_process").methods("GET"_method)([this]()
                                                              {
        json resp;
        if (!m_runningDemo.pids.empty() && isAnyPidRunning(m_runningDemo.pids)) {
            resp["running"] = true;
            resp["program"] = m_runningDemo.program;
            resp["headline"] = m_runningDemo.headline;
            resp["id"] = m_runningDemo.id;
        } else {
            resp["running"] = false;
        }
        return crow::response(resp.dump()); });

    // POST /terminate_process
    CROW_ROUTE(app, "/terminate_process").methods("POST"_method)([this](const crow::request &)
                                                                 {
        std::cerr << "Terminate request received for demo id " << m_runningDemo.id << std::endl;
        if (!m_runningDemo.pids.empty() && isAnyPidRunning(m_runningDemo.pids)) {
            std::cerr << "Terminating processes for demo id " << m_runningDemo.id << std::endl;
            if (terminateProcesses(m_runningDemo.pids)) {
                m_runningDemo.toTerminate = true;
                m_runningDemo.terminationNotificationPending = true;
                m_runningDemo.terminationTime = std::chrono::steady_clock::now();
                // Keep program/headline/id until processes actually exit so client can still show info
                return crow::response(R"({"status":"terminating"})");
            } else {
                return crow::response(500, R"({"status":"error","message":"Failed to terminate"})");
            }
        }
        return crow::response(R"({"status":"no_process"})"); });

    // GET /termination_notification - client polls this while a demo is/was running
    CROW_ROUTE(app, "/termination_notification").methods("GET"_method)([this]()
                                                                        {

        json resp;
        auto anyRunning = isAnyPidRunning(m_runningDemo.pids);
        resp["terminated"] = !anyRunning;
        resp["id"] = m_runningDemo.id;

        if (anyRunning &&
            m_runningDemo.toTerminate && std::chrono::steady_clock::now() - m_runningDemo.terminationTime > std::chrono::seconds(10)) {
                terminateProcesses(m_runningDemo.pids, true);
        }

        if (!anyRunning && m_runningDemo.terminationNotificationPending) {
            m_runningDemo = {};
        }
        std::cerr << "Termination notification polled, terminated=" << resp["terminated"] << ", id=" << resp["id"] << std::endl;
        for (auto id : m_runningDemo.pids)
        {
            std::cerr << " - " << id << std::endl;
        }
        std::cerr << std::endl;
        
        return crow::response(resp.dump()); });


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

void updateDemoIds(){
    auto demos = loadDemos();

    // Collect all existing IDs
    std::set<int> existingIds;
    bool needsUpdate = false;
    
    // First pass: collect all existing IDs
    for (auto& [category, demo_list] : demos.items()) {
        for (auto& demo : demo_list) {
            if (demo.contains("id") && demo["id"].is_number_integer()) {
                existingIds.insert(demo["id"].get<int>());
            }
        }
    }
    
    // Find the next available ID
    int nextId = 1;
    while (existingIds.count(nextId)) {
        nextId++;
    }
    
    // Second pass: assign IDs to demos that don't have them
    for (auto& [category, demo_list] : demos.items()) {
        for (auto& demo : demo_list) {
            if (!demo.contains("id") || !demo["id"].is_number_integer()) {
                demo["id"] = nextId;
                existingIds.insert(nextId);
                std::cerr << "Assigned ID " << nextId << " to demo: " << demo["headline"] << std::endl;
                needsUpdate = true;
                
                // Find next available ID
                do {
                    nextId++;
                } while (existingIds.count(nextId));
            }
        }
    }
    
    // Write back to file if any changes were made
    if (needsUpdate) {
        std::ofstream f(demo::collection);
        if (f) {
            f << demos.dump(2);
        }
    }
}

void DemoServer::run() {
    updateDemoIds();
    m_app = std::make_unique<crow::SimpleApp>();
    crow::SimpleApp &app = *m_app;

    // Disable Crow's logging completely
    app.loglevel(crow::LogLevel::Critical);

    setupRoutes(app);

    // Directly run the server here. main() already runs DemoServer::run()
    // in a separate thread, so we don't need to spawn another thread.
    try {
        std::cerr << "Starting server on port " << demo::port << std::endl;
        app.port(demo::port).multithreaded().run(); // blocks until stop() causes it to return
    } catch (const std::exception &e) {
        std::cerr << "Server run exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Server run unknown exception" << std::endl;
    }

    // ensure any launched demo processes are terminated
    terminateProcesses(m_runningDemo.pids);
    // Destroy app to close sockets and other resources
    m_app.reset();

    // small pause to allow OS to release the port (if needed)
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cerr << "Stopped server..." << std::endl;
}

void DemoServer::stop() {
    m_running = false;
    // Request Crow to stop; app.stop() should cause run() to return
    if (m_app) {
        try {
            m_app->stop();
        } catch (const std::exception &e) {
            std::cerr << "Exception calling app.stop(): " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception calling app.stop()" << std::endl;
        }
    }
}