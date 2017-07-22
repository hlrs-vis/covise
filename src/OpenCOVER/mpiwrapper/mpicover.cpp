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
#ifdef DOTIMING
#include <covise/coTimer.h>
#endif

#include <config/CoviseConfig.h>
#include <config/coConfigConstants.h>
#include <cover/coCommandLine.h>

#include <cover/OpenCOVER.h>
#include <util/environment.h>

#include <mpi.h>

#include <util/coExport.h>

#ifdef WIN32
static char *strcasestr(char *source, char *target)
{
    size_t i = 0, len = 0;
    unsigned char after_space = 1;

    len = strlen(target);
    for (; source[i] != '\0'; i++)
    {

        if (!after_space && source[i] != ' ')
            continue;
        if (source[i] == ' ')
        {
            after_space = 1;
            continue;
        }
        after_space = 0;
        if (!strncasecmp((source + i), target, len))
            return (source + i);
    }
    return NULL;
}
#endif

extern "C" COEXPORT int mpi_main(MPI_Comm comm, int argc, char *argv[])
{
    int mpiinit = 0;
    MPI_Initialized(&mpiinit);
    assert(mpiinit);

#ifdef _WIN32
    // disable "debug dialog": it prevents the application from exiting,
    // but still all sockets remain open
    DWORD dwMode = SetErrorMode(SEM_NOGPFAULTERRORBOX);
    SetErrorMode(dwMode | SEM_NOGPFAULTERRORBOX);
#else
    signal(34, SIG_IGN);
    signal(13, SIG_IGN); // SIGPIPE //would be generated if you write to a closed socket
#endif
    covise::setupEnvironment(argc, argv);
    opencover::coCommandLine(argc, argv);
    char my_hostname[256];
    gethostname(my_hostname, 256);
    std::string mastername(my_hostname);

    int myID = 0;
    MPI_Comm_rank(comm, &myID);
    int len = mastername.size();
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    if (myID != 0)
        mastername.resize(len);
    MPI_Bcast(&mastername[0], len, MPI_BYTE, 0, comm);

    covise::coConfigConstants::setRank(myID);
    covise::coConfigConstants::setMaster(QString::fromStdString(mastername));

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
        else
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
    else
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        if (rank > 0)
        {
            std::string filename = covise::coCoviseConfig::getEntry("slavelog", "COVER.Console");
            if (filename.empty())
            {
#if 0
                fclose(stderr);
                fclose(stdout);
#endif
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
    opencover::OpenCOVER *Renderer = new opencover::OpenCOVER(&comm);
    Renderer->run();
    delete Renderer;

    return 0;
}
