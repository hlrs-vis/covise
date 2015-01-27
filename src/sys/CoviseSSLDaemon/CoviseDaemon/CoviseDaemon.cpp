/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// CoviseDaemon.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include <QApplication>
#include "frmMainWindow.h"
#include <iostream>
#include <string>
#include <vector>

using namespace ::covise;

int containsOption(std::vector<std::string> optionList, std::string option)
{
    std::vector<std::string>::iterator listIter;
    int position = 0;

    listIter = optionList.begin();
    while (listIter != optionList.end())
    {
        if ((*listIter) == option)
        {
            return position;
        }
        position++;
        listIter++;
    }
    return 0;
}

void showMessage()
{
    std::cout << "Covise Remote Daemon rev" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Usage: CoviseDaemon <option>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << "  -g  --gui      Use GUI enabled version, starting maximized (default)" << std::endl;
    std::cout << "  -m  --minimize Use GUI enabled version, starting minimized" << std::endl;
}

int startUp(int argc, char *argv[])
{
    //GUI-Mode maximized
    // Need to distinguish between running daemon or complete start
    // For now only consider complete start.
    QApplication app(argc, argv, true);
    frmMainWindow *main = new frmMainWindow(&app);
    main->setupUi();
    main->show();
    app.setActiveWindow(main);
    app.connect(&app, SIGNAL(lastWindowClosed()), &app, SLOT(quit()));

    return app.exec();
}

int main(int argc, char *argv[])
{

    //seperate commandline parameters
    std::vector<std::string> list;
    for (int i = 0; i < argc; i++)
    {
        list.push_back(argv[i]);
    }

    if (containsOption(list, "-m") || containsOption(list, "--minimized"))
    {
        //GUI-Mode minimized
    }
    else if (containsOption(list, "-g") || containsOption(list, "--gui"))
    {

        return startUp(argc, argv);
    }
    else
    {
        showMessage();
        return startUp(argc, argv);
    }
    int c = getchar();
    std::cerr << c;
    return 0;
}
