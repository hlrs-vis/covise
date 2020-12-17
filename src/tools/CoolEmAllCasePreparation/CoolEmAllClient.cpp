/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <CoolEmAllClient.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include <osg/Vec3d>
#ifndef WIN32
#include <unistd.h>
#endif
#include <string.h>

using namespace std;

extern CoolEmAllClient *bc_U;

CoolEmAllClient::CoolEmAllClient(std::string h)
{
    hostname = h;
    databaseHost = new Host(hostname.c_str());
}
CoolEmAllClient::~CoolEmAllClient()
{
    delete databaseHost;
}
double CoolEmAllClient::getValue(std::string path, std::string var)
{
    conn = new SimpleClientConnection(databaseHost, 9451);
    if (!conn->is_connected())
    {
        std::cerr << "could not connect to: " << hostname << std::endl;
        _exit(-1);
    }
    std::string command;
    command = "getLastMetricByMetricName " + path + " " + var + "\n";
    fprintf(stderr, "%s\n", command.c_str());
    conn->send(command.c_str(), command.length());
    char line[1000];
    line[0] = '\0';
    line[999] = '\0';
    int numRead = conn->getSocket()->Read(line, 1000);
    if (numRead >= 0)
        line[numRead] = '\0';
    double value = -1.0;
    char *valstr = strstr(line, "value=");
    fprintf(stderr, "%s\n", line);
    if (valstr)
    {
        //fprintf(stderr,"%s\n",valstr+7);
        sscanf(valstr + 7, "%lf", &value);
    }
    fprintf(stderr, "%f\n", (float)value);
    return value;
}
bool CoolEmAllClient::setValue(std::string path, std::string var, std::string value)
{
    conn = new SimpleClientConnection(databaseHost, 9451);
    if (!conn->is_connected())
    {
        std::cerr << "could not connect to: " << hostname << std::endl;
        _exit(-1);
    }
    std::string command;
    time_t timei = time(NULL);
    char timec[1000];
    sprintf(timec, "%d", static_cast<int>(timei));
    std::string times = timec;
    std::string experiment;
    std::string trial;
    size_t posSim = path.find("sim");
    size_t posFirstSlash = path.find_first_of("/");
    size_t posSecondSlash = path.find("/", posFirstSlash + 1);
    posSim = posSim + 4;
    experiment = path.substr(posFirstSlash + 1, (posSecondSlash - posFirstSlash - 1));
    size_t posThirdSlash = path.find("/", posSecondSlash + 1);
    trial = path.substr(posSecondSlash + 1, (posThirdSlash - posSecondSlash - 1));
    //std::cerr << "posSim= " << posSim << std::endl;
    //std::cerr << "posFirstSlash= " << posFirstSlash << std::endl;
    //std::cerr << "posSecondSlash= " << posSecondSlash << std::endl;
    //std::cerr << "posThridSlash= " << posThirdSlash << std::endl;
    //std::cerr << "experiment= " << experiment << std::endl;
    //std::cerr << "trial= " << trial << std::endl;
    // example path sim/forDaniel/trial_1/dcworms/hw/testbed/psnc/hpc/hw/rack1/recs_i7/Inlet_1
    // get experiment and trial from path

    command = "putMetricDB \"experimentID:" + experiment + ",trialID:" + trial + ",name:" + var + ",time:" + times + ",value:" + value + ",object_path:" + path + ",source:CFD,output:OK\"\n";
    //std::cerr << "command_DB= " << command << std:: endl;
    conn->send(command.c_str(), command.length());
    char line[1000];
    line[0] = '\0';
    line[999] = '\0';
    int numRead = conn->getSocket()->Read(line, 1000);
    if (numRead >= 0)
        line[numRead] = '\0';
    if (strncmp(line, "None", 4) != 0)
    {
        fprintf(stderr, "error writing to CoolEmAllDB on %s : %s\n command=%s", hostname.c_str(), line, command.c_str());
        return false;
    }
    return true;
}
