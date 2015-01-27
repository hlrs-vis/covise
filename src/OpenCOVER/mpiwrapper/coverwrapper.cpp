/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* wrapper for COVER to distribute environment variables to slaves before
 * initialization of static OpenSceneGraph objects happens */

#include <mpi.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#include <util/findself.h>

#ifndef RENAME_MAIN
#error "RENAME_MAIN not defined"
#endif

#define xstr(s) #s
#define str(s) xstr(s)

int main(int argc, char *argv[])
{

    const char libcover[] = "libmpicover.so";
    const char mainname[] = str(RENAME_MAIN);

    MPI_Init(&argc, &argv);

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::string> vars;
    vars.push_back("PATH");
    vars.push_back("LD_LIBRARY_PATH");
    vars.push_back("DYLD_LIBRARY_PATH");
    vars.push_back("DYLD_FRAMEWORK_PATH");
    vars.push_back("LANG");
    vars.push_back("COCONFIG");
    vars.push_back("COCONFIG_LOCAL");
    vars.push_back("COCONFIG_DEBUG");
    vars.push_back("COVISE_CONFIG");
    //vars.push_back("COVISE_HOST");
    vars.push_back("COVISEDIR");
    vars.push_back("COVISE_PATH");
    vars.push_back("ARCHSUFFIX");
    vars.push_back("OSGFILEPATH");
    vars.push_back("OSG_FILE_PATH");
    vars.push_back("OSG_NOTIFY_LEVEL");
    vars.push_back("OSG_LIBRARY_PATH");
    vars.push_back("OSG_LD_LIBRARY_PATH");
    for (size_t i = 0; i < vars.size(); ++i)
    {
        std::vector<char> buf;
        int len = -1;
        if (rank == 0)
        {
            const char *val = getenv(vars[i].c_str());
            if (val)
            {
                len = strlen(val) + 1;
                buf.resize(len);
                strcpy(&buf[0], val);
            }
        }
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (len >= 0)
        {
            if (rank > 0)
                buf.resize(len);
            MPI_Bcast(&buf[0], buf.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            setenv(vars[i].c_str(), &buf[0], 1 /* overwrite */);
            //std::cerr << vars[i].c_str() << "=" << &buf[0] << std::endl;
        }
    }

    typedef int (*main_t)(int, char *[]);
    main_t realmain = NULL;
    int ret = 0;
    std::string bindir = vistle::getbindir(argc, argv);
    std::string abslib = bindir + "/../../lib/" + libcover;
    void *handle = dlopen(abslib.c_str(), RTLD_LAZY);
    if (!handle)
    {
        std::cerr << "failed to dlopen " << abslib << std::endl;
        ret = 1;
        goto finish;
    }

    realmain = (main_t)dlsym(handle, mainname);
    if (!realmain)
    {
        std::cerr << "could not find " << mainname << " in " << libcover << std::endl;
        ret = 1;
        goto finish;
    }

    ret = realmain(argc, argv);

finish:
    if (handle)
        dlclose(handle);
    MPI_Finalize();

    return ret;
}
