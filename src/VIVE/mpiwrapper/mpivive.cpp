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

#include <OpenConfig/access.h>
#include <config/CoviseConfig.h>
#include <config/coConfigConstants.h>
#include <core/vvCommandLine.h>

#include <core/vvVIVE.h>
#include <util/environment.h>

#ifdef HAS_MPI
#include <mpi.h>
#else
#endif

#include <util/coExport.h>

#include "mpivive.h"
#include "export.h"

extern "C" COEXPORT int mpi_main(MPI_Comm comm, int shmGroupRoot, pthread_barrier_t *shmBarrier, int argc, char *argv[])
{
    mpi_main_t *typecheck = mpi_main;
    (void)typecheck;

#ifdef HAS_MPI
    int mpiinit = 0;
    MPI_Initialized(&mpiinit);
    if (!mpiinit)
    {
        std::cerr << "MPI has not been initialized" << std::endl;
        return -1;
    }
    assert(mpiinit);
#endif

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
    vive::vvCommandLine(argc, argv);
    char my_hostname[256];
    gethostname(my_hostname, 256);
    std::string mastername(my_hostname);

    int myID = 0;
#ifdef HAS_MPI
    MPI_Comm_rank(comm, &myID);
    int len = (int)mastername.size();
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    if (myID != 0)
        mastername.resize(len);
    MPI_Bcast(&mastername[0], len, MPI_BYTE, 0, comm);
#endif

    covise::coConfigConstants::setRank(myID, shmGroupRoot);
    covise::coConfigConstants::setMaster(mastername);

    std::unique_ptr<vive::config::Access> config;
    if (vive::config::Access::isInitialized())
    {
        config = std::make_unique<vive::config::Access>();
    }
    else
    {
        config = std::make_unique<vive::config::Access>(my_hostname, mastername, myID);
        if (auto covisedir = getenv("COVISEDIR"))
        {
            config->setPrefix(covisedir);
        }
    }

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
        fprintf(stderr, "vive: Starting up\n\n");
#ifdef HAS_MPI
    vive::vvVIVE *Renderer = new vive::vvVIVE(&comm, shmBarrier);
#else
    vive::vvVIVE*Renderer = new vive::vvVIVE();
#endif
    Renderer->run();
    config->save();
    delete Renderer;

    return 0;
}
