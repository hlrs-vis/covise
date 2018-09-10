// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include "server_manager.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvtoolshed.h>
#include <virvo/vvvirvo.h>

#include <iostream>
#include <limits>
#include <string>

#ifndef _WIN32
#include <signal.h>
#include <syslog.h>
#include <unistd.h>
#endif

static const unsigned short DEFAULT_PORT = 31050;

//---- Command line options ------------------------------------------------------------------------

// port the server renderer uses to listen for incoming connections
static unsigned short port = DEFAULT_PORT;

// indicating current server mode (default: single server)
static vvServerManager::Mode serverMode = vvServerManager::SERVER;

// indicating the use of bonjour
static bool useBonjour = false;

// run in background as a unix daemon
static bool daemonize = false;

// name of the daemon for reference in syslog
static std::string daemonName = "voxserver";

//--------------------------------------------------------------------------------------------------

static void help()
{
    std::cerr << "Syntax:" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  vserver [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Available options:" << std::endl;
    std::cerr << std::endl;
    std::cerr << "-port" << std::endl;
    std::cerr << " Don't use the default port (" << DEFAULT_PORT << "), but the specified one" << std::endl;
    std::cerr << std::endl;

    if (virvo::hasFeature("bonjour"))
    {
        std::cerr << "-mode" << std::endl;
        std::cerr << " Start server-manager with one of the following modes:" << std::endl;
        std::cerr << " s     single server (default)" << std::endl;
        std::cerr << " rm    resource manager" << std::endl;
        std::cerr << " rm+s  server and resource manager simultanously" << std::endl;
        std::cerr << std::endl;
        std::cerr << "-bonjour" << std::endl;
        std::cerr << " use bonjour to broadcast this service. options:" << std::endl;
        std::cerr << " on" << std::endl;
        std::cerr << " off (default)" << std::endl;
        std::cerr << std::endl;
    }

#ifndef _WIN32
    std::cerr << "-daemon" << std::endl;
    std::cerr << " Start in background as a daemon" << std::endl;
    std::cerr << std::endl;
#endif

    std::cerr << "-debug" << std::endl;
    std::cerr << " Set debug level" << std::endl;
    std::cerr << std::endl;
}

bool parseCommandLine(int argc, char* argv[])
{
    bool HaveBonjour = virvo::hasFeature("bonjour");

    for (int arg = 1; arg < argc; ++arg)
    {
        if (vvToolshed::strCompare(argv[arg], "-help") == 0 ||
            vvToolshed::strCompare(argv[arg], "-h") == 0 ||
            vvToolshed::strCompare(argv[arg], "-?") == 0 ||
            vvToolshed::strCompare(argv[arg], "/?") == 0)
        {
            help();
            return false;
        }
        else if (vvToolshed::strCompare(argv[arg], "-port") == 0)
        {
            if ((++arg)>=argc)
            {
                std::cerr << "No port specified" << std::endl;
                return false;
            }
            else
            {
                int inport = atoi(argv[arg]);
                if (inport > std::numeric_limits<ushort>::max() || inport <= std::numeric_limits<ushort>::min())
                {
                    std::cerr << "Specified port is out of range. Falling back to default: " << DEFAULT_PORT << std::endl;
                    port = DEFAULT_PORT;
                }
                else
                {
                    port = static_cast<unsigned short>(inport);
                }
            }
        }
        else if (HaveBonjour && vvToolshed::strCompare(argv[arg], "-mode") == 0)
        {
            if ((++arg) >= argc)
            {
                std::cerr << "Mode type missing." << std::endl;
                return false;
            }

            if (vvToolshed::strCompare(argv[arg], "s") == 0)
            {
                serverMode = vvServerManager::SERVER;
            }
            else if (vvToolshed::strCompare(argv[arg], "rm") == 0)
            {
                serverMode = vvServerManager::RM;
            }
            else if (vvToolshed::strCompare(argv[arg], "rm+s") == 0)
            {
                serverMode = vvServerManager::RM_WITH_SERVER;
            }
            else
            {
                std::cerr << "Unknown mode type." << std::endl;
                return false;
            }
        }
        else if (HaveBonjour && vvToolshed::strCompare(argv[arg], "-bonjour") == 0)
        {
            if ((++arg) >= argc)
            {
                std::cerr << "Bonjour setting missing." << std::endl;
                return false;
            }
            if (vvToolshed::strCompare(argv[arg], "on") == 0)
            {
                useBonjour = true;
            }
            else if(vvToolshed::strCompare(argv[arg], "off") == 0)
            {
                useBonjour = false;
            }
            else
            {
                std::cerr << "Unknown bonjour setting." << std::endl;
                return false;
            }
        }
#ifndef _WIN32
        else if (vvToolshed::strCompare(argv[arg], "-daemon") == 0)
        {
            daemonize = true;
        }
#endif
        else if (vvToolshed::strCompare(argv[arg], "-debug") == 0)
        {
            if ((++arg) >= argc)
            {
                std::cerr << "Debug level missing." << std::endl;
                return false;
            }

            int level = atoi(argv[arg]);
            if (level >= 0 && level <= 3)
            {
                vvDebugMsg::setDebugLevel(level);
            }
            else
            {
                std::cerr << "Invalid debug level." << std::endl;
                return false;
            }
        }
        else
        {
            std::cerr << "Unknown option/parameter: \"" << argv[arg] << "\", use -help for instructions" << std::endl;
            return false;
        }
    }

    return true;
}

#ifndef _WIN32
static void handleSignal(int sig)
{
    switch (sig)
    {
    case SIGHUP:
        syslog(LOG_WARNING, "Got SIGHUP, quitting.");
        break;
    case SIGTERM:
        syslog(LOG_WARNING, "Got SIGTERM, quitting.");
        break;
    default:
        syslog(LOG_WARNING, "Got unhandled signal %s, quitting.", strsignal(sig));
        break;
    }
}
#endif

//--------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    //virvo::debug::installUnhandledExceptionFilter();

    std::cerr << "Virvo server " << virvo::version() << std::endl;
    std::cerr << "(c) " << VV_VERSION_YEAR << " Juergen Schulze (schulze@cs.brown.edu)" << std::endl;
    std::cerr << "Brown University" << std::endl << std::endl;

    if (!parseCommandLine(argc, argv))
        return EXIT_FAILURE;

#ifndef _WIN32
    if (daemonize)
    {
        signal(SIGHUP, handleSignal);
        signal(SIGTERM, handleSignal);
        signal(SIGINT, handleSignal);
        signal(SIGQUIT, handleSignal);

        setlogmask(LOG_UPTO(LOG_INFO));
        openlog(daemonName.c_str(), LOG_CONS, LOG_USER);

        pid_t pid;
        pid_t sid;

        syslog(LOG_INFO, "Starting %s.", daemonName.c_str());

        pid = fork();
        if (pid < 0)
        {
            return EXIT_FAILURE;
        }

        if (pid > 0)
        {
            return EXIT_SUCCESS;
        }

        umask(0);

        sid = setsid();
        if (sid < 0)
        {
            return EXIT_FAILURE;
        }

        // change working directory to /, which cannot be unmounted
        if (chdir("/") < 0)
        {
            return EXIT_FAILURE;
        }

        // close file descriptors for standard output
        close(STDIN_FILENO);
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
    }
#endif

    try
    {
        // Create a new server manager
        vvServerManager server(port, useBonjour);

        // Start a new accept operation
        server.accept();

        // Run the server
        boost::thread runner(&vvServerManager::run, &server);

        // Wait for the server to quit
        runner.join();
    }
    catch (std::exception& e)
    {
        std::cout << "vvserver_asio: exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
