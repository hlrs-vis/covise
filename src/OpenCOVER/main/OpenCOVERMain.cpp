/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *  (C) 1996-2002                                                       *
 *  Computer Centre University of Stuttgart                             *
 *  Allmandring 30                                                      *
 *  D-70550 Stuttgart                                                   *
 *  Germany                                                             *
 *                                                                      *
 *  Vircinity GmbH                                                      *
 *  Nobelstrasse 15                                                      *
 *  D-70550 Stuttgart                                                   *
 *  Germany                                                             *
 *                                                                      *
 *	Description		main for COVER                                      *
 *                                                                      *
 *	Author			D. Rainer                                           *
 *                                                                      *
 *                                                                      *
 ************************************************************************/

#include <util/common.h>
#include <util/unixcompat.h>
#include <net/covise_socket.h>
#ifdef DOTIMING
#include <covise/coTimer.h>
#endif

#include <OpenConfig/access.h>
#include <config/CoviseConfig.h>
#include <config/coConfigConstants.h>
#include <cover/coCommandLine.h>

#include <cover/OpenCOVER.h>
#include <util/environment.h>

#include <boost/algorithm/string.hpp>
#ifdef HAS_MPI
#include <mpi.h>
#endif

int main(int argc, char *argv[])
{
    covise::Socket::initialize();

#ifdef _WIN32
    // disable "debug dialog": it prevents the application from exiting,
    // but still all sockets remain open
    DWORD dwMode = SetErrorMode(SEM_NOGPFAULTERRORBOX);
    SetErrorMode(dwMode | SEM_NOGPFAULTERRORBOX);
#endif
#ifdef MPI_COVER
    int mpiinit = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
#endif
    bool forceMpi = false;
#ifndef WIN32
    signal(34, SIG_IGN);
    signal(13, SIG_IGN); // SIGPIPE //would be generated if you write to a closed socket
#endif
    covise::setupEnvironment(argc, argv);
    opencover::coCommandLine(argc, argv);
    char my_hostname[256];
    gethostname(my_hostname, 256);
    int myID = 0;
    std::string mastername(my_hostname);
    if ((opencover::coCommandLine::argc() >= 5) && (!strcmp(opencover::coCommandLine::argv(1), "-c")))
    {
        myID = atoi(opencover::coCommandLine::argv(2));
        if (opencover::coCommandLine::argc() >= 6)
        {
            mastername = opencover::coCommandLine::argv(5);
        }
    }
    if (boost::icontains(argv[0], ".mpi"))
    {
#ifdef MPI_COVER
        MPI_Initialized(&mpiinit);
        if (!mpiinit)
            MPI_Init(&argc, &argv);
        forceMpi = true;
        MPI_Comm_rank(comm, &myID);
        int len = (int)mastername.size();
        MPI_Bcast(&len, 1, MPI_INT, 0, comm);
        if (myID != 0)
            mastername.resize(len);
        MPI_Bcast(&mastername[0], len, MPI_BYTE, 0, comm);
#else
        std::cerr << "OpenCOVER: not compiled with MPI support" << std::endl;
        exit(1);
#endif
    }
    opencover::config::Access config(my_hostname, mastername, myID);
    if (auto covisedir = getenv("COVISEDIR"))
    {
        config.setPrefix(covisedir);
    }
    covise::coConfigConstants::setRank(myID);
    covise::coConfigConstants::setMaster(mastername);

    if (argc > 1 && 0 == strcmp(argv[1], "-d"))
    {

        // find the module's name
        const char *modname = argv[0];
        const char *lastSlash = strrchr(modname, '/');
        if (lastSlash)
            modname = lastSlash + 1;

        // create port and parameter output if called with the option -d
        cout << "Module:      \"" << modname << "\"" << endl;
        cout << "Desc:        \""
             << "VR-Renderer"
             << "\"" << endl;

        cout << "Parameters:   " << 4 << endl;
        cout << "  \"Viewpoints\" \"Browser\" \"./default.vwp\" \"Viewpoints\" \"IMM\"" << endl;
        cout << "  \"Viewpoints___filter\" \"BrowserFilter\" \"./default.vwp *.vwp\" \"Viewpoints\" \"IMM\"" << endl;
        cout << "  \"Plugins\" \"String\" \"\" \"Additional plugins\" \"START\"" << endl;
        cout << "  \"WindowID\" \"IntScalar\" \"0\" \"window ID to render to\" \"START\"" << endl;
        cout << "OutPorts:     " << 0 << endl;
        cout << "InPorts:     " << 1 << endl;
        cout << "  \""
             << "RenderData"
             << "\" \""
             << "Geometry|UnstructuredGrid|Points|StructuredGrid|Polygons|Triangles|Quads|TriangleStrips|Lines|Spheres"
             << "\" \""
             << "render geometry"
             << "\" \""
             << "req" << '"' << endl;
        exit(EXIT_SUCCESS);
    }

    bool useVirtualGL = false;
    if (getenv("VGL_ISACTIVE"))
    {
        useVirtualGL = true;
    }

    if (!forceMpi && !useVirtualGL)
    {
        //   sleep(30);
        if (covise::coCoviseConfig::getEntry("COVER.MultiPC.SyncMode") == "MPI")
        {
            const char *coviseDir = getenv("COVISEDIR");
            const char *archsuffix = getenv("ARCHSUFFIX");
            if (coviseDir == 0)
            {
                std::cerr << "COVISEDIR not set" << std::endl;
                exit(0);
            }
            if (archsuffix == 0)
            {
                std::cerr << "ARCHSUFFIX not set" << std::endl;
                exit(0);
            }
            std::string mpiExecutable = std::string(coviseDir).append("/bin/OpenCOVER.mpi");
            std::string mpiRun = covise::coCoviseConfig::getEntry("mpirun", "COVER.MultiPC.SyncMode", "mpirun");

            std::string hostlist = covise::coCoviseConfig::getEntry("hosts", "COVER.MultiPC.SyncMode", "");
            if (hostlist == "")
            {
                std::cerr << "hosts not set in COVER.MultiPC.SyncMode" << std::endl;
                exit(0);
            }

            std::stringstream noOfHosts;
            noOfHosts << (std::count(hostlist.begin(), hostlist.end(), ',') + 1);

            char **argv_mpi = new char *[argc + 8];
            argv_mpi[0] = strdup(mpiRun.c_str());
            argv_mpi[1] = strdup("-n");
            argv_mpi[2] = strdup(noOfHosts.str().c_str());
#if defined(OMPI_MAJOR_VERSION)
            argv_mpi[3] = strdup("-H");
#elif defined(MPICH_VERSION)
            argv_mpi[3] = strdup("-hosts");
#elif defined(MSMPI_VER)
            argv_mpi[3] = strdup("-hosts");
#elif defined(HAS_MPI)
#error "unknown MPI flavor"
#endif
            argv_mpi[4] = strdup(hostlist.c_str());
#if defined(OMPI_MAJOR_VERSION)
            argv_mpi[5] = strdup("-x");
#elif defined(MPICH_VERSION)
            argv_mpi[5] = strdup("-genvlist");
#elif defined(MSMPI_VER)
            argv_mpi[5] = strdup("-env");
#endif
            argv_mpi[6] = strdup("COCONFIG,COCONFIG_LOCAL,COCONFIG_DIR,COCONFIG_SCHEMA,COCONFIG_DEBUG,COVISE_CONFIG,COVISEDIR,COVISE_PATH,ARCHSUFFIX,LD_LIBRARY_PATH");
            argv_mpi[7] = strdup(mpiExecutable.c_str());
            for (int ctr = 1; ctr < argc; ++ctr)
                argv_mpi[ctr + 7] = argv[ctr];
            argv_mpi[argc + 7] = 0;

            std::cerr << "starting ";
            for (int ctr = 0; argv_mpi[ctr] != 0; ++ctr)
                std::cerr << argv_mpi[ctr] << " ";
            std::cerr << std::endl;

            execvp(argv_mpi[0], argv_mpi);
            exit(0);
        }
    }

#ifdef _WIN32
    // note: console has to be allocated after possible handling of argument '-d',
    //    otherwise output of module definition is written onto a volatile console
    if (covise::coCoviseConfig::isOn("COVER.Console", true))
    {
        std::string filebase = covise::coCoviseConfig::getEntry("file", "COVER.Console");
        if (!filebase.empty())
        {
            char *filename = new char[strlen(filebase.c_str()) + 100];
            sprintf(filename, "%s%d.err.txt", filebase.c_str(), 0);
            freopen(filename, "w", stderr);
            sprintf(filename, "%s%d.out.txt", filebase.c_str(), 0);
            freopen("conout$", "w", stdout);
            delete[] filename;
        }
        else if(!getenv("COVISEDEAMONSTART")) //if the coviseDaemon starts OpenCOVER it pipes STDOUT and STDERR in its ui
        {
            AllocConsole();

            freopen("conin$", "r", stdin);
            freopen("conout$", "w", stdout);
            freopen("conout$", "w", stderr);
        }
    }
#else //_WIN32
    if (covise::coCoviseConfig::isOn("COVER.Console", false))
    {
        std::string filename = covise::coCoviseConfig::getEntry("file", "COVER.Console");
        if (!filename.empty())
        {
            if (!freopen((filename + ".stderr").c_str(), "w", stderr))
            {
                std::cerr << "error reopening stderr" << std::endl;
            }
            if (!freopen((filename + ".stdout").c_str(), "w", stdout))
            {
                std::cerr << "error reopening stdout" << std::endl;
            }
        }
    }
#ifdef MPI_COVER
    else
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        if (rank > 0)
        {
            std::string filename = covise::coCoviseConfig::getEntry("slavelog", "COVER.Console");
            if (filename.empty())
            {
                fclose(stderr);
                fclose(stdout);
            }
            else
            {
                if (!freopen((filename + ".stderr").c_str(), "w", stderr))
                {
                    std::cerr << "error reopening stderr" << std::endl;
                }
                if (!freopen((filename + ".stdout").c_str(), "w", stdout))
                {
                    std::cerr << "error reopening stdout" << std::endl;
                }
            }
        }
    }
#endif
#endif

// timing printouts only if enabled in covise.config
#ifdef DOTIMING
    if (coCoviseConfig::isOn("COVER.Timer", false))
    {
        coTimer::init("COVER", 2000);
    }
#endif

    //MARK2("COVER STARTING UP on host %s with pid %d", my_hostname, getpid());

    int dl = covise::coCoviseConfig::getInt("COVER.DebugLevel", 0);

    if (dl >= 1)
        fprintf(stderr, "OpenCOVER: Starting up\n\n");
    opencover::OpenCOVER *Renderer = NULL;
#ifdef MPI_COVER
    if (forceMpi)
    {
        Renderer = new opencover::OpenCOVER(&comm, nullptr);
    }
    else
#endif
    {
        Renderer = new opencover::OpenCOVER();
    }
    Renderer->run();
    config.save();
    delete Renderer;


#ifdef MPI_COVER
    if (!mpiinit)
        MPI_Finalize();
#endif

    return 0;
}
