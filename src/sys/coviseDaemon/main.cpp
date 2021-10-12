/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coverDaemon.h"
#include "mainWindow.h"
#include "tui.h"

#include <QApplication>
#include <QThread>
#include <boost/program_options.hpp>
#include <future>
#include <iostream>

namespace po = boost::program_options;




vrb::VrbCredentials readCredentials(const po::variables_map &vm)
{
        vrb::VrbCredentials cr{};
        std::string ip = cr.ipAddress();
        auto tcp = cr.tcpPort();
        auto udp = cr.udpPort();
        if (vm.count("host"))
        {
                ip = vm["host"].as<std::string>();
        }
        if (vm.count("port"))
        {
                tcp = vm["port"].as<unsigned int>();
        }
        if (vm.count("udp"))
        {
                udp = vm["udp"].as<unsigned int>();
        }
        return vrb::VrbCredentials{ip, tcp, udp};
}

int runGuiDaemon(int argc, char **argv, const po::variables_map &vars)
{
        QApplication a(argc, argv);
        a.setWindowIcon(QIcon(":/images/coviseDaemon.png"));
        CoverDaemon d;
        MainWindow mw{readCredentials(vars)};
        return a.exec();
}

int runCommandlineDaemon(int argc, char **argv, const po::variables_map &vars)
{
        QCoreApplication a(argc, argv);
        CoverDaemon d;
        CommandLineUi tui{ readCredentials(vars), vars.count("autostart") != 0 };
        auto retval = a.exec();
        return retval;
}

int main(int argc, char **argv)
{
        po::options_description desc("usage");
        desc.add_options()("help", "show this message")("host,h", po::value<std::string>(), "VRB address")("port,p", po::value<unsigned int>(), "VRB tcp port")("udp,u", po::value<unsigned int>(), "VRB udp port")("tui, t", "start command line interface")("autostart, a", "launch programs without asking for permission");
#ifdef _WIN32
        _putenv_s("COVISEDEAMONSTART", "true"); //tells the started programms that they are started by this daemon
#else
        setenv("COVISEDEAMONSTART", "true", true); //tells the started programms that they are started by this daemon
#endif
        po::variables_map vm;
        try
        {
                po::positional_options_description popt;
                popt.add("name", 1);
                po::store(po::command_line_parser(argc, argv).options(desc).positional(popt).run(), vm);
                po::notify(vm);
        }
        catch (std::exception &e)
        {
                std::cerr << e.what() << std::endl;
                std::cerr << desc << std::endl;
                return 1;
        }

        if (vm.count("help"))
        {
                std::cerr << desc << std::endl;
                return 0;
        }

        if (vm.count("tui"))
        {
                runCommandlineDaemon(argc, argv, vm);
        }
        else
        {
                runGuiDaemon(argc, argv, vm);
        }
}
