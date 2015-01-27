/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// CoviseClient.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include <iostream>
#include <string>
#include <vector>
#include "SSLClient.h"
#include <stdlib.h>

using namespace std;

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
    std::cout << "Covise Remote Client" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Usage: CoviseClient <option>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << "  -s <ip>       --server  <ip>        Specifiy server ip address to connect to" << std::endl;
    std::cout << "  -p <port>     --port    <port>      Specifiy port to connect to" << std::endl;
    std::cout << "  -c <command>  --command <command>   Specifiy port to connect to" << std::endl;

    char c;
    std::cin >> c;
}

int main(int argc, char *argv[])
{
    int posItem = 0;
    int port = 0;
    std::string sPort = "0";
    std::string server = "";
    std::string command = "";
    SSLClient *client = NULL;

    if (argc < 5)
    {
        showMessage();
        return 0;
    }

    //seperate commandline parameters
    std::vector<std::string> list;
    for (int i = 0; i < argc; i++)
    {
        list.push_back(argv[i]);
    }

    posItem = containsOption(list, "-s");
    if (posItem)
    {
        server = list.at(posItem + 1);
    }
    else
    {
        posItem = containsOption(list, "--server");
        if (posItem)
        {
            server = list.at(posItem + 1);
        }
        else
        {
            showMessage();
            return 0;
        }
    }

    posItem = 0;
    posItem = containsOption(list, "-p");
    if (posItem)
    {
        sPort = list.at(posItem + 1);
    }
    else
    {
        posItem = containsOption(list, "--port");
        if (posItem)
        {
            sPort = list.at(posItem + 1);
        }
        else
        {
            showMessage();
            return 0;
        }
    }

    posItem = 0;
    posItem = containsOption(list, "-c");
    if (posItem)
    {
        command = list.at(posItem + 1);
    }
    else
    {
        posItem = containsOption(list, "--command");
        if (posItem)
        {
            command = list.at(posItem + 1);
        }
        else
        {
            showMessage();
            return 0;
        }
    }

    port = atoi(sPort.c_str());

    try
    {
        client = new SSLClient(server, port);
        client->run(command);

        delete client;
        client = NULL;
    }
    catch (std::exception &ex)
    {
        cerr << "Exception while setting up and running SSLClient!" << endl;
        cerr << "Exception Detail: " << endl;
        cerr << ex.what() << endl;
        return (1);
    }

    return 0;
}
