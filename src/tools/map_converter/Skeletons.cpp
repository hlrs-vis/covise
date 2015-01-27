/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class Skeletons                       ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 11.01.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// COVISE includes
#include "Skeletons.h"
#include "ModuleSkel.h"
#include <util/string_util.h>
#include <config/CoviseConfig.h>
#include <QDir>
#include <QFileInfo>

// system includes
#include <string.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#include <dirent.h>
#include <dirent.h>
#endif
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <map>
#include <set>
#include <memory>

#ifdef __APPLE__
#include <libgen.h> // basename
#endif

//
// Constructor
//
Skeletons::Skeletons()
    : isNormalized_(0)
{
}

// copy constructor
Skeletons::Skeletons(const Skeletons &rm)
    : skels_(rm.skels_)
    , isNormalized_(rm.isNormalized_)
{
}

//add a module skeleton to the list of known modules
int
Skeletons::add(const ModuleSkeleton &skel)
{
    skels_.push_back(skel);
    return SUCCESS; // later possible exceptions of the new operator should be catched and an approp. retval
    // has to be returned
}

struct less_than_str
{
    bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) < 0;
    }
};

typedef std::pair<std::string, std::string> Alias;
typedef std::multimap<std::string, std::string> AliasMap;

// obtain module information by executin all module with the "-d" option
int
Skeletons::obtainLocal(const char *bDir)
{
    AliasMap aliasMap;
    covise::coCoviseConfig::ScopeEntries mae = covise::coCoviseConfig::getScopeEntries("System.CRB", "ModuleAlias");
    const char **moduleAliases = mae.getValue();
    for (int i = 0; moduleAliases && moduleAliases[i] != NULL; i = i + 2)
    {
        //fprintf(stderr, "___ %s___%s\n", moduleAliases[i], moduleAliases[i+1]);
        char *line = new char[strlen(moduleAliases[i]) + 1];

        strcpy(line, moduleAliases[i]);
        strtok(line, ":");
        char *p = strtok(NULL, ":");
        std::string newName(p ? p : "");
        size_t pos = newName.find('/');
        if (pos != std::string::npos)
            newName = std::string(newName, pos + 1);
        std::string oldName(moduleAliases[i + 1]);
        aliasMap.insert(Alias(newName, oldName));

        delete[] line;
    }

    char *binName = new char[10];
    sprintf(binName, "%s/bin", getenv("ARCHSUFFIX"));
    int len = strlen(binName);

    char *fullBinDir = new char[strlen(bDir) + len + 2];
    strcpy(fullBinDir, bDir);
    strcat(fullBinDir, "/");
    //strcat(fullBinDir,arch);
    //strcat(fullBinDir,"/");
    strcat(fullBinDir, binName);

    // now all executable module names will be stored in the array execModules
    std::vector<std::string> execModules;
    std::vector<std::string> execModuleNames;
    std::vector<std::string> groupName;

    QDir *dirp = new QDir(fullBinDir);
    if (dirp == NULL || !dirp->exists())
    {
        fprintf(stderr, " Skeletons::obtainLocal(..) could not open directory %s\n", fullBinDir);
    }
    else
    {
        QFileInfoList list = dirp->entryInfoList();
        for (int i = 0; i < list.size(); ++i)
        {
            QFileInfo dirFileInfo = list.at(i);
            if (dirFileInfo.isDir() && dirFileInfo.fileName() != "." && dirFileInfo.fileName() != "..")
            {
                QDir dir(dirFileInfo.absoluteFilePath());
                if (dir.exists())
                {
                    QFileInfoList flist = dir.entryInfoList();
                    for (int i = 0; i < flist.size(); ++i)
                    {
                        QFileInfo fileInfo = flist.at(i);
                        if (!fileInfo.isDir() && fileInfo.isExecutable())
                        {
                            groupName.push_back(dirFileInfo.fileName().toLatin1().data());
                            execModules.push_back(std::string(fileInfo.absoluteFilePath().toLatin1().data()));
                            std::string modNm = fileInfo.fileName().toLatin1().data();
#ifdef _WIN32
                            if (modNm.rfind(".exe") == (modNm.length() - 4))
                                modNm = std::string(modNm, 0, modNm.length() - 4);
#endif
                            execModuleNames.push_back(modNm);
                        }
                    }
                }
            }
        }
    }

    std::string basedir(fullBinDir);
    delete[] fullBinDir;

    // each module is executed with the -d option to obtain the module information
    for (int ii = 0; ii < nameList_.size(); ++ii)
    {
        std::string group;
        std::string mod = nameList_[ii];
        AliasMap::iterator end = aliasMap.upper_bound(mod);
        for (AliasMap::iterator it = aliasMap.lower_bound(mod); it != end; ++it)
        {
            mod = it->second;
            size_t pos = mod.find('/');
            if (pos != std::string::npos)
            {
                group = std::string(mod, 0, pos);
                mod = std::string(mod, pos + 1);
                fprintf(stderr, " Skeletons::obtainLocal() using alias %s/%s for %s\n",
                        group.c_str(), mod.c_str(), nameList_[ii].c_str());
            }
        }

        // check if we have a name list and execute only modules which
        // are contained in the name list
        std::string modNm;
        for (int i = 0; i < execModules.size(); ++i)
        {
            if (mod == execModuleNames[i])
            {
                std::string cmd = execModules[i];
                cmd += " -d";
                //fprintf(stderr, " Skeletons::obtainLocal(..) execute %s\n", cmd.c_str());
                char buf[10000];
                FILE *ptr;
#ifdef WIN32
                if ((ptr = _popen(cmd.c_str(), "r")) != NULL)
#else
                if ((ptr = popen(cmd.c_str(), "r")) != NULL)
#endif
                {
                    // here we parse the module output line by line
                    std::vector<std::string> lines;
                    while (fgets(buf, sizeof(buf), ptr) != NULL)
                    {
                        lines.push_back(std::string(buf));
                    }
                    if (lines.size() > 0)
                    {
                        if (!parseModOutput(groupName[i], lines, nameList_[ii]))
                        {
                            return FAILURE;
                        }
                        else
                        {
                            // the module was sucessfully executed we will remove it
                            // from nameList_
                            remFromNameList(modNm);
                        }
                    }
                    else
                    {
                        fprintf(stderr, " WARNING: <%s> got no output\n", cmd.c_str());
                    }
#ifdef WIN32
                    if (ptr)
                        _pclose(ptr);
#else
                    if (ptr)
                        pclose(ptr);
#endif
                }
                break;
            }
        }
    }

    return SUCCESS;
}

//   Parses output lines from the execution of each module with the option "-d"
//   the lines array has to be allocated outside the method.
//   An object of type ModuleSkeleton is created and will be added to the array of
//   known modules at the end of the method
int
Skeletons::parseModOutput(const std::string &group, const std::vector<std::string> &lines, const std::string &name)
{
    const char QUOTE[] = { '"', '\0' };
    const char SPACE[] = { ' ', '\0' };

    // moduleSkeleton data structure to be added after parsing the module information
    ModuleSkeleton *tmpModule = NULL;

    // go through all lines
    int getParam = 0;
    int getInPorts = 0;
    int getOutPorts = 0;
    const char *modName = NULL;
    char *modDesc = NULL;
    for (int lineCntr = 0; lineCntr < lines.size(); ++lineCntr)
    {
        // we have either a line starting with a key ("Module:", "Desc:", "Parameters:", ..)
        // or a data line
        char delim[64];
        strcpy(delim, SPACE);
        int lineLen = lines[lineCntr].size();
        if (lineLen > 0) // our line is not empty
        {
            //fprintf(stderr, " Skeletons::parseModOutput(..)  line:  %s\n", lines[lineCntr] );
            char *line = new char[1 + lineLen];
            strcpy(line, &lines[lineCntr][0]);
            char *tok;
            // module data
            int numParam = 0;
            int numInPorts = 0;
            int numOutPorts = 0;
            int key = 0;
            tok = strtok(line, delim);
            if (tok != NULL)
            {
                // here we get the module name
                if (!strcmp(tok, "Module:") && (lineCntr == 0))
                {
                    strcpy(delim, QUOTE);
                    tok = strtok(NULL, delim);
                    tok = strtok(NULL, delim);
                    int len = strlen(tok) + 1;
                    char *p = new char[len];
                    strcpy(p, tok);
                    if (len > 4)
                    {
                        if (strcmp(p + strlen(p) - 4, ".exe") == 0)
                        {
                            p[strlen(p) - 4] = '\0';
                        }
                    }
                    modName = p;

                    key = 1;
                    //fprintf(stderr, " Skeletons::parseModOutput(..)  name:  %s\n", modName );
                }
                // here we get the module description
                if (!strcmp(tok, "Desc:") && (lineCntr > 0))
                {
                    strcpy(delim, QUOTE);
                    tok = strtok(NULL, delim);
                    tok = strtok(NULL, delim);
                    int len = strlen(tok) + 1;
                    // the describtion field may be empty
                    modDesc = new char[len];
                    if (!strcmp(tok, "\n"))
                    {
                        delete[] modDesc;
                        modDesc = new char[32];
                        strcpy(modDesc, "description missing");
                    }
                    else
                        strcpy(modDesc, tok);
                    key = 1;
                    //fprintf(stderr, " Skeletons::parseModOutput(..)  desc:  %s\n", modDesc );
                }
                strcpy(delim, SPACE);

                // at this point we can assume that a complete module structure
                // will follow therefore we create an instance of ModuleSkeleton
                if (tmpModule == NULL)
                {
                    if ((modName != NULL) && (modDesc != NULL))
                    {
                        //std::cerr << "will create a new module " << name << "" << group << "  " << modDesc << std::endl;
                        tmpModule = new ModuleSkeleton(name.c_str(), group.c_str(), modDesc);

                        int ii;
                        // alias from translation file
                        for (ii = 0; ii < nameAliases_.cnt(name.c_str()); ++ii)
                        {
                            std::string altName = nameAliases_.get(name.c_str(), ii);
                            tmpModule->addAltName(altName);
                        }
                        for (ii = 0; ii < grpAliases_.cnt(name.c_str()); ++ii)
                        {
                            std::string altGrp = grpAliases_.get(name.c_str(), ii);
                            tmpModule->addAltGroup(altGrp);
                        }
                    }
                }

                // here we get the parameter data
                if (!strcmp(tok, "Parameters:") && (lineCntr > 1))
                {
                    tok = strtok(NULL, delim);
                    int len = strlen(tok) + 1;
                    char *buf = new char[len];
                    strcpy(buf, tok);
                    numParam = atoi(buf);
                    //fprintf(stderr, " Skeletons::parseModOutput(..)  num Param:  %d\n", numParam );
                    delete[] buf;
                    key = 1;
                    getParam = numParam;
                }

                // here we get the inPort data
                if (!strcmp(tok, "InPorts:") && (lineCntr > 1))
                {
                    tok = strtok(NULL, delim);
                    int len = strlen(tok) + 1;
                    char *buf = new char[len];
                    strcpy(buf, tok);
                    numInPorts = atoi(buf);
                    getInPorts = numInPorts;
                    //fprintf(stderr, " Skeletons::parseModOutput(..)  num InPorts:  %d\n", numInPorts );
                    key = 1;
                    delete[] buf;
                }

                // here we get the outPort data
                if (!strcmp(tok, "OutPorts:") && (lineCntr > 1))
                {
                    tok = strtok(NULL, delim);
                    int len = strlen(tok) + 1;
                    char *buf = new char[len];
                    strcpy(buf, tok);
                    numOutPorts = atoi(buf);
                    getOutPorts = numOutPorts;
                    //fprintf(stderr, " Skeletons::parseModOutput(..)  num OutPorts:  %d\n", numOutPorts );
                    key = 1;
                    delete[] buf;
                }

                // here we have a data line ..
                if (key == 0)
                {
                    // ..containing parameter information
                    if ((getParam > 0))
                    {
                        strcpy(delim, QUOTE);
                        const int nP = 5;
                        // lineLen is already known
                        char *dataLine = new char[1 + lineLen];
                        strcpy(dataLine, &lines[lineCntr][0]);
                        const char *paramD[nP];
                        for (int i = 0; i < nP; ++i)
                        {
                            if (i == 0)
                                tok = strtok(dataLine, delim);
                            else
                                tok = strtok(NULL, delim);
                            tok = strtok(NULL, delim);
                            if (tok == NULL)
                            {
                                std::cerr << " got incomplete parameter info in module <"
                                          << modName
                                          << ">" << std::endl;
                                return FAILURE;
                            }
                            paramD[i] = tok;
                            //					  std::cerr << " Skeletons::parseModOutput(..) got paraminfo " << i << " : " << tok << std::endl;
                        }

                        std::string prmNm(paramD[0]);
                        std::string modNm(modName);
                        std::string multNm(prmNm + std::string("@") + modNm);

                        ParamSkel paramS(paramD[0], paramD[1], paramD[2],
                                         paramD[3], paramD[4]);

                        // we have to add all alias names to the parameter
                        int ii;

                        for (ii = 0; ii < multPrmAliases_.cnt(multNm); ++ii)
                        {
                            std::string altName = multPrmAliases_.get(multNm, ii);
                            paramS.addAltName(altName);

                            //std::cerr << " Skeletons::parseModOutput(..) added alias <"
                            //          << altName
                            //          << "> to parameter <"
                            //          << prmNm << ">" << std::endl;
                        }

                        if (tmpModule)
                        {
                            tmpModule->add(paramS);
                        }

                        delete[] dataLine;

                        getParam--;
                    }
                    // ..containing InPort information
                    if ((getInPorts > 0))
                    {
                        strcpy(delim, QUOTE);
                        const int nP = 4;
                        // lineLen is already known
                        char *dataLine = new char[1 + lineLen];
                        strcpy(dataLine, &lines[lineCntr][0]);
                        char *portD[nP];
                        for (int i = 0; i < nP; ++i)
                        {
                            if (i == 0)
                                tok = strtok(dataLine, delim);
                            else
                                tok = strtok(NULL, delim);
                            tok = strtok(NULL, delim);

                            if (tok == NULL)
                            {
                                std::cerr << " got incomplete input port-info in module <"
                                          << modName
                                          << ">" << std::endl;
                                return FAILURE;
                            }

                            portD[i] = tok;
                        }
                        PortSkel portS(portD[0], portD[1], portD[2], portD[3], PIN);

                        // we have to add all alias names to the port
                        int ii;
                        std::string portNm(portD[0]);
                        std::string modNm(modName);
                        std::string multNm(portNm + std::string("@") + modNm);
                        int replPol = multPortAliases_.getPolicy(multNm);
                        portS.setReplacePolicy(replPol);
                        for (ii = 0; ii < multPortAliases_.cnt(multNm); ++ii)
                        {
                            std::string altName = multPortAliases_.get(multNm, ii);
                            portS.addAltName(altName);

                            // 					  std::cerr << "Skeletons::parseModOutput(..) added alias <"
                            // 					       << altName
                            // 					       << "> to port <"
                            // 					       << portNm << ">" << std::endl;
                        }

                        if (tmpModule)
                        {
                            tmpModule->add(portS);
                        }

                        delete[] dataLine;
                        getInPorts--;
                    }

                    // ..containing OutPort information
                    if ((getOutPorts > 0))
                    {
                        strcpy(delim, QUOTE);
                        const int nP = 4;
                        // lineLen is already known
                        char *dataLine = new char[1 + lineLen];
                        strcpy(dataLine, &lines[lineCntr][0]);
                        char *portD[nP];
                        for (int i = 0; i < nP; ++i)
                        {
                            if (i == 0)
                                tok = strtok(dataLine, delim);
                            else
                                tok = strtok(NULL, delim);
                            tok = strtok(NULL, delim);
                            //
                            //					  fprintf(stderr, " Skeletons::parseModOutput(..)  OutPort %d:  %s\n", i,tok );
                            if (tok == NULL)
                            {
                                std::cerr << " got incomplete output port-info in module <"
                                          << modName
                                          << ">" << std::endl;
                                return FAILURE;
                            }

                            portD[i] = tok;
                        }

                        PortSkel portS(portD[0], portD[1], portD[2], portD[3], POUT);

                        // we have to add all alias names to the port
                        int ii;
                        std::string portNm(portD[0]);
                        std::string modNm(modName);
                        std::string multNm(portNm + std::string("@") + modNm);

                        int replPol = multPortAliases_.getPolicy(multNm);
                        portS.setReplacePolicy(replPol);

                        for (ii = 0; ii < multPortAliases_.cnt(multNm); ++ii)
                        {
                            std::string altName = multPortAliases_.get(multNm, ii);
                            portS.addAltName(altName);
                            // 					  std::cerr << " Skeletons::parseModOutput(..) added alias <"
                            // 					       << altName
                            // 					       << "> to port <"
                            // 					       << portNm << ">" << std::endl;
                        }

                        if (tmpModule)
                            tmpModule->add(portS);

                        delete[] dataLine;
                        getOutPorts--;
                    }
                }
                strcpy(delim, SPACE);
            }
        } // if line not empty

    } // for all lines

    if (tmpModule)
    {
        add(*tmpModule);
    }

    delete tmpModule;
    delete[] modName;
    delete[] modDesc;

    return SUCCESS;
}

const ModuleSkeleton &
Skeletons::get(const char *name, const char *grp)
{

    int i;
    for (i = 0; i < skels_.size(); ++i)
    {
        // 	    if (skels_[i].nameValid( std::string(name) ))
        // 		  std::cerr << "Skeletons::get(..) name VALID for module "
        // 		       << skels_[i].getName() << std::endl;
        // 	    if (skels_[i].groupValid( std::string(name) ))
        // 		  std::cerr << "Skeletons::get(..) group VALID for module "
        // 		       << skels_[i].getName() << std::endl;

        if ((skels_[i].nameValid(std::string(name)))
            && (skels_[i].groupValid(std::string(grp))))
            return skels_[i];
    }

    // a 2nd pass w/o group (as COVISE ignores module categories)
    for (i = 0; i < skels_.size(); ++i)
    {
        if (skels_[i].nameValid(std::string(name)))
            return skels_[i];
    }

    return emptySkel_;
}

void
Skeletons::writeToCache()
{
    // to be implemented
}

void
Skeletons::readCache()
{
    // to be implemented
}

int
Skeletons::actCache()
{
    // to be implemented
    return 0;
}

Skeletons::~Skeletons()
{
}

// read translation table form file
int
Skeletons::getTranslations(const char *file)
{
    FILE *fd;

    fd = fopen(file, "r");

    if (!fd)
    {
        return FAILURE;
    }

    char *tok;
    char del[64];

    char buf[10000];

    std::string modName;
    std::string altName;
    std::string altGrp;
    std::string multParamName;
    std::string multPortName;
    std::string portPolicy;

    int iPortPolicy = Translations::NONE;

    while (fgets(buf, sizeof(buf), fd) != NULL)
    {
        std::string bufStr(buf);
        std::string s = strip(bufStr);
        if (bufStr.length() > 0 && bufStr[0] != '\r' && bufStr[0] != '\n')
        {
            // we have no comment
            if (bufStr.find_first_of("#") == std::string::npos)
            {

                strcpy(del, ":");
                tok = strtok(buf, del);

                if (tok)
                {
                    strcpy(del, "\n");
                    if (!strcmp(tok, "module"))
                    {
                        tok = strtok(NULL, del);
                        if (tok)
                        {
                            iPortPolicy = Translations::NONE;
                            modName = strip(tok);
                            //std::cerr << "got module " << modName << std::endl;
                        }
                    }
                    else if (!strcmp(tok, "name"))
                    {

                        tok = strtok(NULL, del);
                        if (tok)
                        {
                            altName = strip(tok);

                            //  			    std::cerr << "got alternative name <"
                            //  				 << altName
                            //  				 << "> for module <"
                            //  				 << modName << ">" << std::endl;

                            nameAliases_.add(modName, altName);
                        }
                    }
                    else if (!strcmp(tok, "group"))
                    {
                        tok = strtok(NULL, del);
                        if (tok)
                        {
                            altGrp = strip(tok);
                            grpAliases_.add(modName, altGrp);
                        }
                    }
                    else if (!strcmp(tok, "portPolicy"))
                    {
                        tok = strtok(NULL, del);
                        if (tok)
                        {
                            std::string portPolicy = strip(tok);

                            if (portPolicy == std::string("translations"))
                            {
                                iPortPolicy = Translations::TRANSLATIONS;
                            }
                        }
                    }
                    else if (!strcmp(tok, "param"))
                    {
                        std::string prmName;
                        std::string altPrmName;
                        tok = strtok(NULL, del);
                        if (tok)
                        {
                            char *x = new char[1 + strlen(tok)];
                            strcpy(x, tok);
                            prmName = strip(tok);
                            strcpy(del, ">");
                            tok = strtok(x, del);
                            if (tok)
                            {
                                prmName = strip(tok);
                            }

                            tok = strtok(NULL, del);
                            if (tok)
                            {
                                altPrmName = strip(tok);
                            }
                            delete[] x;
                        }
                        multParamName = prmName + std::string("@") + modName;
                        multPrmAliases_.add(multParamName, altPrmName);
                    }
                    else if (!strcmp(tok, "port"))
                    {
                        std::string portName;
                        std::string altPortName;
                        tok = strtok(NULL, del);
                        if (tok)
                        {
                            char *x = new char[1 + strlen(tok)];
                            strcpy(x, tok);
                            portName = strip(tok);
                            strcpy(del, ">");
                            tok = strtok(x, del);
                            if (tok)
                            {
                                portName = strip(tok);
                            }

                            tok = strtok(NULL, del);
                            if (tok)
                            {
                                altPortName = strip(tok);
                            }
                            delete[] x;
                        }

                        // 			std::cerr << " Skeletons::getTranslations(..) : got translation for port "
                        // 			     << portName << " to port " << altPortName << std::endl;

                        multPortName = portName + std::string("@") + modName;
                        multPortAliases_.add(multPortName, altPortName, iPortPolicy);
                    }
                }
            }
        }
    }
    return SUCCESS;
}

void
Skeletons::normalizeNameList()
{
    // look at translations and replace an entry of nameList_
    // with its actual module-name
    int i;
    for (i = 0; i < nameList_.size(); ++i)
    {
        std::string nm(nameList_[i]);
        if (!nameAliases_.isName(nm))
        {
            nm = nameAliases_.getNameByAlias(nm);
            if (!nm.empty())
                nameList_[i] = nm;
        }
    }
    isNormalized_ = 1;
}

const Skeletons &
    Skeletons::
    operator=(const Skeletons &rs)
{
    std::cerr << "Skeletons::operator=(const Skeletons& rs) NO ASSIGNMENT " << std::endl;
    return rs;
}

// remove name from nameList_
int
Skeletons::remFromNameList(const std::string &name)
{
    int remCnt = 0;
    for (int i = 0; i < nameList_.size(); ++i)
    {
        if (name == nameList_[i])
        {
            for (int k = i; k < nameList_.size() - 1; ++k)
                nameList_[k] = nameList_[k + 1];
            nameList_.pop_back();
            remCnt++;
        }
    }

    return remCnt;
}

const std::string *Skeletons::findReplacement(std::string name)
{

    return multPrmAliases_.findReplacement(name);
}
