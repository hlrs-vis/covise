#include "tui.h"
#include <iostream>
#include <iomanip>

#include <csignal>

// void handle_sigint(int)
// {
// }


// ANSI color codes
const char *BLUE =  "\033[1;34m";
const char *GREEN = "\033[1;32m";
const char *RESET = "\033[0m";


void listDemos(const nlohmann::json& demos)
{

    for (const auto& [categoryName, demoList] : demos.items()) {
        std::cout << "----- " << BLUE << categoryName << RESET << " -----\n";
        // Sort demos by id
        std::vector<nlohmann::json> sorted = demoList;
        std::sort(sorted.begin(), sorted.end(), [](const nlohmann::json& a, const nlohmann::json& b) {
            return a.value("id", 0) < b.value("id", 0);
        });

        // Print in columns (3 columns)
        size_t cols = 3;
        size_t rows = (sorted.size() + cols - 1) / cols;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                size_t idx = c * rows + r;
                if (idx < sorted.size()) {
                    const auto& demo = sorted[idx];
                    int id = demo.value("id", 0);
                    std::string name = demo.value("headline", demo.value("name", ""));
                    std::cout << std::setw(3) << GREEN << id << RESET << ") " << name;
                    // Pad to column width
                    int pad = 30 - (int)name.size();
                    for (int i = 0; i < pad; ++i) std::cout << " ";
                }
            }
            std::cout << "\n";
        }
    }
    std::cout << GREEN << "q" << RESET << ") exit this menu" << std::endl;

}

nlohmann::json findDemo(const nlohmann::json& demos, int id)
{
    for (const auto& [categoryName, demoList] : demos.items())
    {
        for (const auto& demo : demoList)
        {
            if (demo.value("id", 0) == id)
            {
                return demo;
            }
        }
    }
    return nlohmann::json(); // Return empty if not found
}


DemoTui::DemoTui(const nlohmann::json &demos, int demoId)
{
    if(demoId > 0)
    {
        auto demo = findDemo(demos, demoId);
        if (demo.is_null())
        {
            std::cout << "Demo with ID " << demoId << " not found.\n";
            return;
        }
        std::cout << "Starting " << demo.value("headline", "") << "...\n";
        if(launcher.launchDemo(demo))
            runningDemoId = demoId;
    } else
    {
        listDemos(demos);
    }
    while(true)
    {
        std::string input;
        std::cin >> input;
        if(runningDemoId > 0)
        {
           launcher.terminateDemo(runningDemoId);
           runningDemoId = -1;
           listDemos(demos);
        }
        if (input == "q" || input == "exit" || input == "quit") {
            break;
        } else {
            try {
                int id = std::stoi(input);
                auto demo = findDemo(demos, id);
                if (demo.is_null()) {
                    std::cout << "Demo with ID " << id << " not found.\n";
                    continue;
                }
                std::cout << "Starting " << demo.value("headline", "") << "...\n";
                if(launcher.launchDemo(demo))
                    runningDemoId = id;
            } catch (const std::invalid_argument&) {
                if(runningDemoId < 0)
                    std::cout << "Invalid input. Please enter a valid demo ID or 'list' to see available demos.\n";
            }
        }
    }
}