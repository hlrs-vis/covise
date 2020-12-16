
#include "mainWindow.h"
#include "tui.h"
#include <QApplication>

#include <boost/program_options.hpp>

#include <future>
#include <iostream>

namespace po = boost::program_options;

vrb::VrbCredentials readCredentials(const po::variables_map &vm)
{
        vrb::VrbCredentials cr{};
        std::string ip = cr.ipAddress;
        auto tcp = cr.tcpPort;
        auto udp = cr.udpPort;
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

int main(int argc, char **argv)
{
        po::options_description desc("usage");
        desc.add_options()("help", "show this message")("host,h", po::value<std::string>(), "VRB address")("port,p", po::value<unsigned int>(), "VRB tcp port")("udp,u", po::value<unsigned int>(), "VRB udp port")("tui, t", "start command line interface")("autostart, a", "launch programs without asking for permission");

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
        bool autostart = false;
        if (vm.count("autostart"))
        {
                autostart = true;
        }
        QApplication a(argc, argv);

        if (!vm.count("tui"))
        {
                MainWindow mw{readCredentials(vm)};
                mw.setWindowTitle("VrbRemoteLauncher");
                mw.show();
                return a.exec();
        }
        else
        {
                Tui tui(readCredentials(vm), autostart);
                std::thread s{
                    [&tui]() {
                            tui.run();
                    }};
                auto retval = a.exec();
                s.join();
                return retval;
        }
}
