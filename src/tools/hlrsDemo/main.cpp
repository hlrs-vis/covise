#include "flowlayout.h"
#include "verticallabel.h"
#include "demowindow.h"
#include "tui.h"
#include <demo.h>
#include <QApplication>
#include <QFile>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

int main(int argc, char *argv[])
{
    // Check for --tui
    bool gui = false;
    int id = -1;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--gui")
        {
            gui = true;
            break;
        } else
        {
            try
            {
                id = std::stoi(argv[i]);    
            }
            catch(const std::exception& e)
            {
                std::cerr << "Invalid argument: " << argv[i] << ". Use --gui to launch GUI or provide a demo ID." << std::endl;
                return -1;
            }
            
        }
    }
    if (!QFile::exists(demo::collection.c_str()))
    {
        std::cerr << "Demos file not found: " << demo::collection << std::endl;
        std::cerr << "Please set the HLRS_DEMO_DIR environment variable to the correct path." << std::endl;
        return -1;
    }
    nlohmann::json demos = readDemosJson(demo::collection.c_str());
    if (gui)
    {
        QApplication app(argc, argv);
        DemoWindow window(demos);
        window.setWindowTitle("HLRS Demo Launcher");
        window.resize(900, 700);
        window.show();
        return app.exec();
    }
    DemoTui tui(demos, id);
    return 0;
}