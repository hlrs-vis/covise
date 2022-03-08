/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGCONSTANTS_H
#define COCONFIGCONSTANTS_H

#define CO_CONFIG_TRANSFORM_FILE "transform.xml"

#include <util/coTypes.h>
#include <set>
#include <string>
#include <vector>
namespace covise
{

    extern char pathSeparator;
    class CONFIGEXPORT coConfigConstants
    {

    public:
        enum ConfigScope
        {
            Default = -1,
            Global = 1,
            Cluster,
            Host,
            Additional
        };

        static const std::set<std::string> &getArchList();
        static const std::string &getHostname();
        static const std::string &getBackend();

        static void setMaster(const std::string &hostname);
        static const std::string &getMaster();

        static void setBackend(const std::string &backend);
        static void setRank(int rank, int shmGroupRootRank = -1);
        static int getRank();
        static int getShmGroupRootRank();

    protected:
        coConfigConstants();
        virtual ~coConfigConstants();

    private:
        std::string backend;
        std::string hostname;
        std::string master;
        std::set<std::string> archlist;
        int rank = -1;
        int shmGroupRoot = -1;

        static coConfigConstants *instance;

        void initXerces();
        void terminateXerces();
    };

    class CONFIGEXPORT coConfigDefaultPaths
    {

    private:
        coConfigDefaultPaths();
        ~coConfigDefaultPaths();

    public:
        static const std::string &getDefaultTransformFileName();

        static const std::string &getDefaultLocalConfigFileName();
        static const std::string &getDefaultGlobalConfigFileName();

        static const std::string &getDefaultLocalConfigFilePath();
        static const std::string &getDefaultGlobalConfigFilePath();

        static const std::set<std::string> &getSearchPath();

    private:
        static coConfigDefaultPaths *getInstance()
        {

            if (instance)
            {
                return instance;
            }
            else
            {
                return new coConfigDefaultPaths();
            }
        }

        static coConfigDefaultPaths *instance;

        void setNames();
        std::string configGlobal, configLocal, defaultTransform, configGlobalPath, configLocalPath;
        std::set<std::string> searchPath;
    };
}
#endif
