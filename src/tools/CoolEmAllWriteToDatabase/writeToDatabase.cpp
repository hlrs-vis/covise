/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * File:	writeToDatabase
 * Author: 	hpcdrath
 * Created on 4. July 2013
 */

#include "CoolEmAllClient.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>
#include <map>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "no path to file specified!!" << std::endl;
        exit(1);
    }
    else
    {
        CoolEmAllClient *cc = new CoolEmAllClient("recs1.coolemall.eu");
        std::string filePath = argv[1];
        size_t end = filePath.length();
        if (filePath.at(end - 1) == '/')
        {
            filePath = filePath.substr(0, end - 1);
        }
        //std::cerr << "filePath= " << filePath << std::endl;
        std::string patchPath = filePath + "/" + "patchname.txt";
        ifstream pathNames;
        pathNames.open(patchPath.c_str());
        if (pathNames.is_open())
        {
            while (pathNames.good())
            {
                std::string patchNameBuffer;
                std::string pathName;
                std::getline(pathNames, patchNameBuffer);
                std::getline(pathNames, pathName);
                if (!patchNameBuffer.empty())
                {
                    std::istringstream findKeyword;
                    findKeyword.str(patchNameBuffer);
                    std::string keywordBuffer;
                    //std::map<std::string,std::string> patchType;
                    int i = 0;
                    std::string patchName;
                    std::string keyword;
                    while (!findKeyword.eof())
                    {
                        findKeyword >> keywordBuffer;
                        //std::cerr << "keywordBuffer=_" << keywordBuffer << "_" << std::endl;
                        i++;
                        if (i == 1)
                        {
                            if (!keywordBuffer.empty())
                            {
                                keyword = keywordBuffer;
                            }
                        }
                        else if (i == 2)
                        {
                            if (!keywordBuffer.empty() && keywordBuffer.length() > 0)
                            {
                                patchName = keywordBuffer;
                            }
                        }
                    }
                    //patchType[patchName] = keyword;
                    //std::cerr << "keyword=_" << keyword << "_" << std::endl;
                    //std::cerr << "keywordLaenge= " << keyword.length() << std::endl;

                    //std::cerr << "keywordBuffer=_" << keyword << "_" << std::endl;
                    //std::cerr << "keywordBufferLaenge= " << keywordBuffer.length() << std::endl;

                    if (keyword == "outlet")
                    {
                        std::string command;
                        command = "patchAverage T " + patchName + " -latestTime -case " + filePath;
                        //std::cerr << "command patchAverage T= " << command << std::endl;
                        char buf[1000];
#ifdef WIN32
                        FILE *averageresultsT = _popen(command.c_str(), "r");
#else
                        FILE *averageresultsT = popen(command.c_str(), "r");
#endif
                        while (averageresultsT != NULL && !feof(averageresultsT))
                        {
                            if (!fgets(buf, 1000, averageresultsT))
                            {
                                std::cerr << "fgets error 1" << std::endl;
                                break;
                            }
                            if (strstr(buf, "Average of") != NULL)
                            {
                                std::string value;
                                std::string line = buf;
                                size_t posEqual = line.find_last_of("=");
                                value = line.substr(posEqual + 2, (line.length() - (posEqual + 2) - 1));
                                cc->setValue(pathName, "temperature", value);
                                //std::cerr << "temperatureValue=" << value << std::endl;
                                break;
                            }
                        }
                        std::string systemCommand = "foamCalc mag U -latestTime -case " + filePath;
                        int j = system(systemCommand.c_str());
                        //std::cerr << "j= " << j << std::endl;
                        command = "patchAverage magU " + patchName + " -latestTime -case " + filePath;
//std::cerr << "command patchAverage magU= " << command << std::endl;
#ifdef WIN32
                        FILE *averageresultsU = _popen(command.c_str(), "r");
#else
                        FILE *averageresultsU = popen(command.c_str(), "r");
#endif
                        while (averageresultsU != NULL && !feof(averageresultsU))
                        {
                            if (!fgets(buf, 1000, averageresultsU))
                            {
                                std::cerr << "fgets error 2" << std::endl;
                                break;
                            }
                            if (strstr(buf, "Average of") != NULL)
                            {
                                //std::cerr << "Hallo aus der velocityValue if-Schleife" << std::endl;
                                std::string value;
                                std::string line = buf;
                                size_t posEqual = line.find_last_of("=");
                                value = line.substr(posEqual + 2, (line.length() - (posEqual + 2) - 1));
                                cc->setValue(pathName, "velocity", value);
                                //std::cerr << "velocityValue= " << value << std::endl;
                                break;
                            }
                        }
                    }
                    else if (keyword == "inlet")
                    {
                        std::string command;
                        command = "patchAverage p " + patchName + " -latestTime -case " + filePath;
                        //std::cerr << "command patchAverage p " << command << std::endl;
                        char buf[1000];
#ifdef WIN32
                        FILE *averageresultsp = _popen(command.c_str(), "r");
#else
                        FILE *averageresultsp = popen(command.c_str(), "r");
#endif
                        while (averageresultsp != NULL && !feof(averageresultsp))
                        {
                            if (!fgets(buf, 1000, averageresultsp))
                            {
                                std::cerr << "fgets error 3" << std::endl;
                                break;
                            }
                            if (strstr(buf, "Average of") != NULL)
                            {
                                std::string value;
                                std::string line = buf;
                                size_t posEqual = line.find_last_of("=");
                                value = line.substr(posEqual + 2, (line.length() - (posEqual + 2) - 1));
                                cc->setValue(pathName, "pressureDrop", value);
                                //std::cerr << "pressureDropValue= " << value << std::endl;
                                break;
                            }
                        }
                    }
                    else
                    {
                        std::cerr << "could not read file" << std::endl;
                    }
                }
            }
            pathNames.close();
        }
        else
        {
            std::cerr << "could not open input file:" << patchPath << std::endl;
            exit(1);
        }
    }
}
