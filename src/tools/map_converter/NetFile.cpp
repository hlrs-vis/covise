/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class NetFile                         ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:  16.01.2001                                                   ++
// ++**********************************************************************/

#include "NetFile.h"
#include "Connection.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <util/string_util.h>
#include <fstream>

//
// Constructor
//
NetFile::NetFile()
    :

    fileRead_(0)
    , sDiag_(cerr)
    , skels_(NULL)
    , allocInc_(64)
    , allocSize_(allocInc_)
    , hosts_(NULL)
    , users_(NULL)
    , numHosts_(0)
    , version_(0)
{
}

NetFile::NetFile(Skeletons *skels, ostream &sDiag)
    :

    fileRead_(0)
    , sDiag_(sDiag)
    , skels_(skels)
    , allocInc_(64)
    , allocSize_(allocInc_)
    , hosts_(NULL)
    , users_(NULL)
    , numHosts_(0)
    , version_(0)
{
}

//
// Method
//
int
NetFile::read(const std::string &filename)
{
    string line;
    ifstream file;
    file.open(filename.c_str());
    if (!file.fail() && file.good())
    {
        bool firstline = true;
        while (!file.eof() && file.good())
        {
            getline(file, line);
            if ((line.length() == 0) || (line[0] != '#'))
                inLines_.push_back(line);
            else if (firstline && line.length() > 0 && line[0] == '#')
            {
                version_ = atoi(line.c_str() + 1);
            }
            firstline = false;
        }
        file.close();
        fileRead_ = 1;
        return 1;
    }
    fprintf(stderr, "NetFile::read(): could not open file '%s'\n", filename.c_str());
    fileRead_ = 0;
    return 0;
}

void
NetFile::check(CheckMode mode)
{
    const int lBorder = 70; // use to format output

    // here we check the module informations in current map-file
    // output to stdout;

    // if module skeletons are not present our work ends here
    if (skels_ == NULL)
    {
        fprintf(stderr, "NetFile::check(): no module skeleton information available\n");
        return;
    }

    int fPtr = 0;

    // we read the number of hosts

    int numHosts = 0;

    std::string numHostsStr(inLines_[fPtr]);

    // surely the first checkpoint - if we read NO valid i.e. non negative int
    // we have to give up
    if (isIntNumber(numHostsStr))
    {
        numHosts = atoi(numHostsStr.c_str());
    }
    else
    {
        cerr << "Fatal Error in line "
             << fPtr
             << " of COVISE workflow-file: non negative integer expected "
             << endl;
        exit(EXIT_FAILURE);
    }

    fPtr++;

    int cnt = 0;
    for (cnt = 0; cnt < numHosts; ++cnt)
    {
        std::string host(inLines_[fPtr]);
        fPtr++;
        std::string user(inLines_[fPtr]);
        fPtr++;
        add(host, user);
    }

    // next checkpoint - if we read NO valid i.e. non negative int
    // we have to give up
    int numModules;
    std::string numModulesStr(inLines_[fPtr].c_str());
    if (isIntNumber(numModulesStr))
    {
        numModules = atoi(numModulesStr.c_str());
    }
    else
    {
        cerr << "Fatal Error in line "
             << fPtr
             << " of COVISE workflow-file: non negative integer expected "
             << endl;
        exit(EXIT_FAILURE);
    }

    //sOut_ << inLines_[fPtr] << endl;
    sDiag_ << "================================================================" << endl;
    fprintf(stderr, "map contains %d modules\n", numModules);
    fPtr++;

    int i;
    // loop over all modules
    for (i = 0; i < numModules; ++i)
    {
        std::string modName(inLines_[fPtr]);
        sDiag_ << "MODULE <"
               << modName
               << ">";
        int msgLen = 9 + (int)modName.size();
        fPtr++;
        // index in net-file
        int netIndex = atoi(inLines_[fPtr].c_str());
        fPtr++;
        // host
        std::string modHost(inLines_[fPtr]);
        fPtr++;
        // group
        std::string modGroup(inLines_[fPtr]);
        fPtr++;
        // desc
        std::string modDesc(inLines_[fPtr]);
        fPtr++;
        // x-pos
        int Xpos = atoi(inLines_[fPtr].c_str());
        fPtr++;
        // y-pos
        int Ypos = atoi(inLines_[fPtr].c_str());
        fPtr++;
        // num InPorts
        int numInPorts = atoi(inLines_[fPtr].c_str());
        //	    fprintf(stderr, "  module %s contains %d InPorts \n",modName.c_str(),numInPorts);
        fPtr++;

        // we get a module skeleton from the skeletons if no skeleton with  adequate
        // name and group exist an empty skeleton is returned

        ModuleSkeleton actMod(skels_->get(modName.c_str(), modGroup.c_str()));

        int modSignature = 0;
        int numParamsMod = 0;

        int modEmpty = actMod.empty();
        if (modEmpty && mode != NO_REM)
        {
            int kk;
            for (kk = msgLen; kk < lBorder; ++kk)
                sDiag_ << ".";
            sDiag_ << "REMOVED!**" << endl;
        }

        int j;
        for (j = 0; j < numInPorts; ++j)
        {
            std::string portName(inLines_[fPtr]);
            //cerr << "NetFile::check(..)  get port(PIN) " << portName << endl;
            if (!modEmpty)
                actMod.checkPortPolicy(portName, 0);
            fPtr += 5;
        }

        // num OutPorts
        int numOutPorts = atoi(inLines_[fPtr].c_str());
        //	    fprintf(stderr, "  module %s contains %d OutPorts \n",modName.c_str(),numOutPorts);
        fPtr++;

        for (j = 0; j < numOutPorts; ++j)
        {
            std::string portName(inLines_[fPtr]);
            //cerr << "NetFile::check(..)  get port(POUT) " << portName << endl;
            if (!modEmpty)
                actMod.checkPortPolicy(portName, 0);
            fPtr += 5;
        }

        // reset the port replace policy if needed
        if (!modEmpty)
        {
            std::string dummy;
            actMod.checkPortPolicy(dummy, 1);
        }
        // num Params
        int numParams = atoi(inLines_[fPtr].c_str());
        //	    fprintf(stderr, "  module %s contains %d parameters \n",modName.c_str(),numParams);
        fPtr++;

        if (!modEmpty || mode == NO_REM)
        {
            numParamsMod = actMod.getNumParams();

            if (actMod.getNumInPorts() != numInPorts)
            {
                modSignature++;
                // 			sDiag_ << " WARNING: the number of input-ports in module <"
                // 			      << modName
                // 			      << "> differs! ("
                // 			      << numInPorts
                // 			      << " InPorts <--> "
                // 			      << actMod.getNumInPorts()
                // 			      << " InPorts in recent version of module" << endl;
            }
            if (actMod.getNumOutPorts() != numOutPorts)
            {
                modSignature++;
                // 			sDiag_ << " WARNING: the number of output-ports in module <"
                // 			      << modName
                // 			      << " differs! ("
                // 			      << numOutPorts
                // 			      << " OutPorts <--> "
                // 			      << actMod.getNumOutPorts()
                // 			      << " OutPorts in recent version of module" << endl;
            }
            if (actMod.getNumParams() != numParams)
            {
                modSignature++;
                // 			sDiag_ << " WARNING: the number of parameters in module <"
                // 			      << modName
                // 			      << " differs! ("
                // 			      << numParams
                // 			      << " Params <--> "
                // 			      << actMod.getNumParams()
                // 			      << " Params in recent version of module" << endl;
            }

            if (modSignature == 0)
            {
                int kk;
                for (kk = msgLen; kk < lBorder; ++kk)
                    sDiag_ << ".";
                sDiag_ << "OK" << endl;

                // 			sDiag_ << "module <"
                // 			      << modName
                // 			      << "> has the same number of input-ports, output-ports \nand parameters compared to the latest skeleton .. good" << endl;
            }
            else
            {
                int kk;
                for (kk = msgLen; kk < lBorder; ++kk)
                    sDiag_ << ".";
                sDiag_ << "MODIFIED" << endl;
            }
            // set X,Y for output
            actMod.setDesc(modDesc);
            actMod.setHost(modHost);
            actMod.setPos(Xpos, Ypos);
            actMod.setOrgNetIndex(netIndex);
            actMod.setOrgModName(modName);
        }

        // all parameters are written to the output stream which have been found in the
        // input file

        ParamSkel *params = new ParamSkel[numParamsMod];
        int nParam = 0;

        //sOut_ << numParamsMod << endl;
        for (j = 0; j < numParams; ++j)
        {
            std::string parName(inLines_[fPtr]);
            //sDiag_ << "     got parameter " << parName << endl;
            fPtr++;
            std::string parType(inLines_[fPtr]);
            //sDiag_ << "       type: " << parType << endl;
            fPtr++;
            std::string parDesc(inLines_[fPtr]);
            //sDiag_ << "       description: " << parDesc << endl;
            fPtr++;
            std::string parValue(inLines_[fPtr]);
            //sDiag_ << "       value: " << parValue << endl;
            fPtr++;
            std::string parMode(inLines_[fPtr]);
            //sDiag_ << "       mode: " << parMode << endl;
            fPtr++;
            std::string parSelect(inLines_[fPtr]);
            //sDiag_ << "       selection: " << parSelect << endl;
            fPtr++;

            if (!actMod.empty())
            {
                ParamSkel *param;
                if (parType == std::string("Choice"))
                    param = new ChoiceParamSkel(actMod.getParam(parName, parType));
                else
                    param = new ParamSkel(actMod.getParam(parName, parType));

                if (param->empty())
                {
                    delete param;
                    param = 0;
                    string name = parName + "@" + modName;
                    const std::string *replacement = skels_->findReplacement(name);
                    if (replacement)
                    {
                        sDiag_ << " Replacing parameter " << parName << " with parameter " << *replacement << endl;
                        if (parType == std::string("Choice"))
                            param = new ChoiceParamSkel(actMod.getParam(*replacement, parType));
                        else
                            param = new ParamSkel(actMod.getParam(*replacement, parType));
                    }
                    else
                    {
                        sDiag_ << " WARNING: parameter "
                               << parName
                               << " does NOT belong to module <"
                               << modName
                               << "> anymore!"
                               << endl
                               << "         old value: "
                               << parValue << " **" << endl;
                    }
                }

                if (param && !param->empty())
                {
                    if (version_ == 0 && parType == "Browser")
                    {
                        std::string::size_type pos = parValue.find_last_of(' ');
                        if (pos != std::string::npos)
                        {
                            parValue = parValue.substr(0, pos);
                        }
                    }
                    param->setValue(parValue);
                    int sel = atoi(parSelect.c_str());
                    param->setSelect(sel);

                    params[nParam] = *param;
                    delete param;
                    nParam++;
                }
            }
        }
        // if the module description contains parameters which are not yet in the input file
        if (!actMod.empty())
        {
            const int dummy = 1234;
            // call to obtain number of unknown params
            int numUnknwn = actMod.getUnusedParams(NULL, dummy);
            //		  cerr << "==> got # unKnown params " << numUnknwn << endl;
            ParamSkel *paramUnknwn = new ParamSkel[numUnknwn + 1];

            numUnknwn = actMod.getUnusedParams(paramUnknwn, numUnknwn + 1);

            int k;
            for (k = 0; k < numUnknwn; ++k)
            {

                params[nParam] = paramUnknwn[k];
                nParam++;

                sDiag_ << " WARNING: parameter <"
                       << paramUnknwn[k].getName()
                       << "> is NEW in module "
                       << modName << endl
                       << "          it will enter the map-file with default values" << endl;
            }
            delete[] paramUnknwn;
        }

        // delete all parameters in mod
        // add instead all parameters from params
        if (!actMod.empty())
        {
            actMod.deleteAllParams();

            for (int i = 0; i < nParam; ++i)
                actMod.add(params[i]);

            // add module to array of modules in current net-file
            add(actMod);
        }
        delete[] params;
        fPtr++;
    }

    reCnt();

    sDiag_ << "================================================================" << endl;

    // now we will work at the Connections

    int numConn = atoi(inLines_[fPtr].c_str());
    fPtr++;

    int numValidConn = 0;
    int k;
    int frmIdx;
    int toIdx;
    for (k = 0; k < numConn; ++k)
    {
        std::string frmModName(inLines_[fPtr]);
        fPtr++;
        frmIdx = atoi(inLines_[fPtr].c_str());
        fPtr += 2;
        std::string frmPortName(inLines_[fPtr]);
        fPtr++;
        fPtr++;
        std::string toModName(inLines_[fPtr]);
        fPtr++;
        toIdx = atoi(inLines_[fPtr].c_str());
        fPtr += 2;
        std::string toPortName(inLines_[fPtr]);
        fPtr++;
        ModuleSkeleton frmMod(get(frmModName, frmIdx));
        ModuleSkeleton toMod(get(toModName, toIdx));

        Connection con(frmMod, toMod, frmPortName, toPortName);

        sDiag_ << "CONNECTION from "
               << frmModName << ":" << frmPortName
               << " to "
               << toModName << ":" << toPortName;

        size_t msgLen = 20 + frmModName.size() + frmPortName.size() + toModName.size() + toPortName.size();

        if (con.valid() == Connection::Valid)
        {
            size_t kk;
            for (kk = msgLen; kk < lBorder; ++kk)
                sDiag_ << ".";
            sDiag_ << "VALID" << endl;
            ++numValidConn;
            add(con);
        }
        else
        {
            size_t kk;
            for (kk = msgLen; kk < lBorder; ++kk)
                sDiag_ << ".";
            sDiag_ << "*INVALID*" << endl;
        }
    }
}

int
NetFile::getNumModules() const
{
    if (inLines_.size() < 13)
    {
        return 0;
        cerr << "NetFile::getNumModules()  inLinesLen_ < 13 return 0" << endl;
    }
    int fPtr = 0;
    int numHosts = 0;

    std::string numHostsStr(inLines_[fPtr]);

    // surely the first checkpoint - if we read NO valid i.e. non negative int
    // we have to give up
    if (isIntNumber(numHostsStr))
    {
        numHosts = atoi(numHostsStr.c_str());
    }
    else
    {
        cerr << "Fatal Error in line "
             << fPtr
             << " of COVISE workflow-file: non negative integer expected "
             << endl;
        exit(EXIT_FAILURE);
    }

    fPtr += (1 + 2 * numHosts);

    // next checkpoint - if we read NO valid i.e. non negative int
    // we have to give up
    int numModules = 0;
    std::string numModulesStr(inLines_[fPtr].c_str());
    if (isIntNumber(numModulesStr))
    {
        numModules = atoi(numModulesStr.c_str());
    }
    else
    {
        cerr << "Fatal Error in line "
             << fPtr
             << " of COVISE workflow-file: non negative integer expected "
             << endl;
        exit(EXIT_FAILURE);
    }
    cerr << "NetFile::getNumModules() got numModules " << numModules << endl;

    return numModules;
}

void
NetFile::getNameList(std::vector<std::string> &array, int &num)
{
    if (!fileRead_)
    {
        fprintf(stderr, "   NetFile::allocAndGetNameList() input not read\n");
        return;
    }

    int fPtr = 0;

    int numHosts = 0;

    std::string numHostsStr(inLines_[fPtr]);

    // surely the first checkpoint - if we read NO valid i.e. non negative int
    // we have to give up
    if (isIntNumber(numHostsStr))
    {
        numHosts = atoi(numHostsStr.c_str());
    }
    else
    {
        cerr << "Fatal Error in line "
             << fPtr
             << " of COVISE workflow-file: non negative integer expected "
             << endl;
        exit(EXIT_FAILURE);
    }

    fPtr += (1 + 2 * numHosts);

    // next checkpoint - if we read NO valid i.e. non negative int
    // we have to give up
    int numModules;
    std::string numModulesStr(inLines_[fPtr].c_str());
    if (isIntNumber(numModulesStr))
    {
        numModules = atoi(numModulesStr.c_str());
    }
    else
    {
        cerr << "Fatal Error in line "
             << fPtr
             << " of COVISE workflow-file: non negative integer expected "
             << endl;
        exit(EXIT_FAILURE);
    }

    fPtr++;

    int i;
    // loop over all modules
    for (i = 0; i < numModules; ++i)
    {

        std::string modName(inLines_[fPtr]);

        if (i < num)
            array[i] = modName;
        fPtr += 7;

        int numInPorts = atoi(inLines_[fPtr].c_str());
        fPtr += (1 + (5 * numInPorts));

        int numOutPorts = atoi(inLines_[fPtr].c_str());
        fPtr += (1 + (5 * numOutPorts));

        int numParams = atoi(inLines_[fPtr].c_str());
        fPtr += (2 + (6 * numParams));
    }
}

ostream &
operator<<(ostream &s, const NetFile &f)
{
    s << f.numHosts_ << endl;
    int i;
    for (i = 0; i < f.numHosts_; ++i)
    {
        s << f.hosts_[i] << endl;
        s << f.users_[i] << endl;
    }
    s << "#numModules" << endl;
    s << f.modules_.size() << endl;

    for (i = 0; i < f.modules_.size(); ++i)
    {
        s << f.modules_[i];
    }

    s << f.connections_.size() << endl;

    for (i = 0; i < f.connections_.size(); ++i)
    {
        s << f.connections_[i];
    }

    s << 0 << endl;

    return s;
}

void
NetFile::add(const ModuleSkeleton &mod)
{
    modules_.push_back(mod);
}

void
NetFile::add(const Connection &con)
{
    connections_.push_back(con);
}

void
NetFile::add(const std::string &host, const std::string &user)
{
    std::string *tmpU = new std::string[numHosts_ + 1];
    std::string *tmpH = new std::string[numHosts_ + 1];
    int i;
    for (i = 0; i < numHosts_; ++i)
    {
        tmpH[i] = hosts_[i];
        tmpU[i] = users_[i];
    }

    delete[] users_;
    delete[] hosts_;

    tmpH[numHosts_] = host;
    tmpU[numHosts_] = user;
    numHosts_++;

    hosts_ = tmpH;
    users_ = tmpU;
}

const ModuleSkeleton &
NetFile::get(const std::string &name, const int &idx)
{

    int i;
    for (i = 0; i < modules_.size(); ++i)
    {
        if (modules_[i].nameValid(name)
            && (idx == modules_[i].getOrgNetIndex())
            && (name == modules_[i].getOrgModName()))
            return modules_[i];
    }
    return emptyMod_;
}

void
NetFile::reCnt()
{

    const size_t len = modules_.size();

    std::string *names = new std::string[len];
    int *cnts = new int[len];

    memset(cnts, 0, modules_.size() * sizeof(int));

    int cntMax = 0;

    int i;
    for (i = 0; i < modules_.size(); ++i)
    {
        int j;
        int found = 1;

        std::string thisMdNm = modules_[i].getName();
        for (j = 0; j < cntMax; ++j)
        {
            if (names[j] == thisMdNm)
            {
                cnts[j]++;
                found = 0;
            }
        }
        if (found)
        {
            cnts[cntMax] = 1;
            names[cntMax] = thisMdNm;
            cntMax++;
        }
    }

    for (size_t i = modules_.size() - 1; i >= 0; --i)
    {
        int idx;
        std::string thisMdNm = modules_[i].getName();
        int j;
        for (j = 0; j < cntMax; ++j)
        {
            if (names[j] == thisMdNm)
            {
                int xx = cnts[j];
                idx = xx;
                xx--;
                cnts[j] = xx;
                break;
            }
        }
        modules_[i].setNetIndex(idx);
    }

    delete[] cnts;
    delete[] names;
}

int
NetFile::replaceUserHost(const std::string &userHostStr)
{

    cerr << "NetFile::replaceUserHost(..) got user-host str: "
         << userHostStr
         << endl;

    // parse userHostStr
    size_t sep = userHostStr.find_first_of(":");
    std::string oldInfo;
    std::string newInfo;
    if (sep != std::string::npos)
    {
        oldInfo = userHostStr.substr(0, sep);
        newInfo = userHostStr.substr(sep + 1);
    }
    else
    {
        // we return because the user-host str is not complete
        return -1;
    }

    sep = oldInfo.find_first_of("@");
    std::string oldUser(oldInfo.substr(0, sep));
    std::string oldHost(oldInfo.substr(sep + 1));
    sep = newInfo.find_first_of("@");
    std::string newUser(newInfo.substr(0, sep));
    std::string newHost(newInfo.substr(sep + 1));

    cerr << "NetFile::replaceUserHost(..) got old host "
         << oldHost << " old user " << oldUser << endl;

    cerr << "NetFile::replaceUserHost(..) got new host "
         << newHost << " new user " << newUser << endl;

    // replace user, host in the netFile
    int i;
    int sucess = 0;
    for (i = 0; i < numHosts_; ++i)
    {
        if (hosts_[i] == oldHost)
        {
            hosts_[i] = newHost;
            cerr << "NetFile::replaceUserHost(..) replace host <"
                 << oldHost
                 << "> with new host <"
                 << newHost << "> " << endl;
            sucess++;
        }
        if (users_[i] == oldUser)
        {
            users_[i] = newUser;
            cerr << "NetFile::replaceUserHost(..) replace user <"
                 << oldUser
                 << "> with new user <"
                 << newUser << "> " << endl;

            sucess++;
        }
    }

    // replace host at all modules
    for (i = 0; i < modules_.size(); ++i)
    {
        if (modules_[i].getHost() == oldHost)
        {
            modules_[i].setHost(newHost);
            sucess++;
        }
    }

    // replace host at all connections
    for (i = 0; i < connections_.size(); ++i)
    {
        if (connections_[i].getFrmHost() == oldHost)
        {
            connections_[i].setFrmHost(newHost);
            sucess++;
        }
        if (connections_[i].getToHost() == oldHost)
        {
            connections_[i].setToHost(newHost);
            sucess++;
        }
    }

    return sucess;
}

void
NetFile::makeLocal()
{
    int i;

    // print something
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "Removing all remote hosts from Map\n");
    fprintf(stderr, "================================================================\n");

    // forget all other hosts in the map
    numHosts_ = 1;

    // replace host/User at all modules
    for (i = 0; i < modules_.size(); ++i)
    {
        modules_[i].setHost("LOCAL");
    }

    // replace host at all connections
    for (i = 0; i < connections_.size(); ++i)
    {
        connections_[i].setFrmHost("LOCAL");
        connections_[i].setToHost("LOCAL");
    }

    return;
}

//
// Destructor
//
NetFile::~NetFile()
{
}

std::string
NetFile::getInputFileName() const
{
    return inputFileName_;
}
