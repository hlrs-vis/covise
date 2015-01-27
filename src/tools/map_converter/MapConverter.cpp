/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++              Main for map_converter (tool a adopt old style COVISE  ++
// ++              map-files to the current state of the modules          ++
// ++              it currently maps:                                     ++
// ++                -module names                                        ++
// ++                -module groups)                                      ++
// ++                -input ports                                         ++
// ++                -output ports                                        ++
// ++                -parameters                                          ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: $Date: 16-Nov-2001.18:25:53 $                                                  ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "Skeletons.h"
#include "NetFile.h"
#include "PyFile.h"
#include "Environment.h"

#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <util/XGetOpt.h>
#endif
#include <covise/covise_version.h>
#ifndef _WIN32
#include <unistd.h>
#include <dirent.h>
#else
#include <util/unixcompat.h>
#endif

#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace covise;

std::string Version("2.0");

void usage(char *myName);
void version();
#ifndef _WIN32
int max(int a, int b)
{
    return ((a > b) ? a : b);
}
#endif

int main(int argc, char **argv)
{

    //
    // obtain environment variables
    //
    // COVISE_GLOBALINSTDIR
    std::string coGlobInstDir = Environment->get(std::string("COVISE_GLOBALINSTDIR"));
    // COVISEDIR
    std::string coDir = Environment->get(std::string("COVISEDIR"));
    if (coDir.empty() && coGlobInstDir.empty())
    {
        fprintf(stderr, "main: COVISEDIR and COVISE_GLOBALINSTDIR are not set; \n");
        fprintf(stderr, "main: COVISE directory could not be identified exiting..\n");
        exit(EXIT_FAILURE);
    }

    // ARCHSUFFIX
    //      char *arch;
    std::string arch = Environment->get(std::string("ARCHSUFFIX"));
    if (arch.empty())
    {
        cerr << "main: ARCHSUFFIX is not set;" << endl;
        cerr << "main: architecture could not be identified exiting.." << endl;
        exit(EXIT_FAILURE);
    }

    std::string translDefaultGlob;
    std::string translDefaultLoc;
    std::string baseDirLoc;
    std::string baseDirGlob;

    // find location of map_converter_translations.txt
    if (!coGlobInstDir.empty())
    {
        translDefaultGlob = coGlobInstDir + std::string("/share/covise/map_converter_translations.txt");
        baseDirGlob = coGlobInstDir + std::string("/") + arch;
    }

    translDefaultLoc = coDir + std::string("/share/covise/map_converter_translations.txt");
    baseDirLoc = coDir + std::string("/") + arch;

    //
    // handle options and arguments
    //
    int c;
    extern char *optarg;
    extern int optind;

    int errflg = 0;
    int overWrite = 0;
    bool makeLocal = false;
    char *ofile = NULL;
    char *tfile = NULL;

    // insert defaults here;
    std::string outFile;
    std::string translationFile;
    std::string fileToConvert;
    std::string removeMode("NO");
    std::string userHostStr;
    bool createPythonFile(false);

    while ((c = getopt(argc, argv, "lfvPo:t:hr:u:")) != -1)
        switch (c)
        {
        case 'u':
            userHostStr = std::string(optarg);
            break;
        case 'o':
            ofile = optarg;
            outFile = std::string(ofile);
            break;
        case 't':
            tfile = optarg;
            translationFile = std::string(tfile);
            break;
        case 'v':
            version();
            exit(EXIT_SUCCESS);
            break;
        case 'r':
            removeMode = std::string(optarg);
            break;
        case 'l':
            makeLocal = true;
            break;
        case 'f':
            overWrite = 1;
            break;
        case 'P':
            createPythonFile = true;
            break;
        case 'h':
            errflg++;
            break;
        case '?':
            errflg++;
            break;
        }

    if (errflg)
    {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (optind < argc)
        fileToConvert = std::string(argv[optind]);
    else
    {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    // settings for remove mode
    CheckMode checkMode; // type CheckMode defined in NetFile.h
    if (removeMode == std::string("AUTO"))
        checkMode = AUTO_REM;
    else if (removeMode == std::string("NO"))
        checkMode = NO_REM;
    else if (removeMode == std::string("QUERY"))
        checkMode = QUERY_REM;
    else
        checkMode = NO_REM;

    // make a skeletons obj
    Skeletons *skeletons = new Skeletons;

    fprintf(stderr, "================================================================\n");
    fprintf(stderr, " obtaining local module information .. (this may take a while)  \n");
    fprintf(stderr, "================================================================\n");

    // read translatons.txt
    if (translationFile.empty())
    {
        // if no translation file is given in the options - we read the one in $COVISEDIR ..
        if (!skeletons->getTranslations(translDefaultLoc.c_str()))
        {
            cerr << "MapConverter: could not read "
                 << translDefaultLoc << endl;
            // .. it is not present we try to read the one of $COVISE_GLOBALINSTDIR
            if (!skeletons->getTranslations(translDefaultGlob.c_str()))
            {
                cerr << "MapConverter: could not read "
                     << translDefaultGlob << endl;

                cerr << " WARNING:  no translation table will be used " << endl;
            }
        }
    }
    else
    {
        skeletons->getTranslations(translationFile.c_str());
    }

    // if an output file is given by the option -o <output file>
    // we want diagonstic output written to stdout

    NetFile *nFile;
    if (!outFile.empty())
    {
        if (createPythonFile)
        {
            nFile = new PyFile(skeletons, cout);
        }
        else
        {
            nFile = new NetFile(skeletons, cout);
        }
    }
    else
    {
        if (createPythonFile)
        {
            nFile = new PyFile(skeletons);
        }
        else
        {
            nFile = new NetFile(skeletons);
        }
    }

    // we read the file we intend to convert
    if (nFile->read(fileToConvert) == 0)
    {
        cerr << argv[0]
             << ": "
             << "cannot read " << fileToConvert
             << endl;
        exit(EXIT_FAILURE);
    }

    // we obtain an array of module names..
    int numMods = nFile->getNumModules();
    if (numMods == 0)
    {
        // we assume that an error occurred if there are no modules
        cerr << argv[0]
             << ": "
             << fileToConvert << " is probably not a COVISE map file"
             << endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::string> nameList;
    nameList.resize(numMods);
    nFile->getNameList(nameList, numMods);
    // ... and pass it to the Skeletons object
    skeletons->setNameList(nameList);

    // all old names are mapped to new ones due to the translation table
    skeletons->normalizeNameList();

    // we go through all elements of $COVISE_PATH and try to obtain
    // module information

    const std::string covisePath("COVISE_PATH");
#ifdef WIN32
    PathList pl = Environment->scan(covisePath, ";");
#else
    PathList pl = Environment->scan(covisePath);
#endif
    if (pl.empty())
    {
        cerr << argv[0]
             << ": "
             << covisePath
             << " not set or empty exiting.. "
             << endl;
        exit(EXIT_FAILURE);
    }

    int successCnt = 0;
    PathList::iterator it;
    for (it = pl.begin(); it != pl.end(); ++it)
    {
        std::string part = *it;
        int ret = skeletons->obtainLocal(part.c_str());
        if (ret)
            successCnt++;
    }
    // at least in one of the dirs obtainLocal should be successful
    if (successCnt == 0)
    {
        cerr << argv[0]
             << ": cannot convert file "
             << fileToConvert << endl;
        exit(EXIT_FAILURE);
    }

    // compares the incoming net-file with the skeletons obtained
    // in the step above (by directly querying the modules)
    nFile->check(checkMode);

    // in case the option -u is given the user-host info is modified
    if (!userHostStr.empty())
    {
        // userHostStr may be a comma-separated list; split it in parts
        // and call  nFile->replaceUserHost(..) for each part
        std::string pat(",");
        int end = userHostStr.find_first_of(pat.c_str());
        int beg = 0;
        if (end == std::string::npos)
        {
            nFile->replaceUserHost(userHostStr);
        }
        else
        {
            std::string prtUsrHstStr(userHostStr.substr(beg, (end - beg)));
            nFile->replaceUserHost(prtUsrHstStr);
            while (end != std::string::npos)
            {
                beg = end + pat.size();
                end = userHostStr.find_first_of(pat.c_str(), beg);
                std::string prtUsrHstStr;
                if (end != std::string::npos)
                {
                    prtUsrHstStr = userHostStr.substr(beg, (end - beg));
                }
                else
                {
                    prtUsrHstStr = userHostStr.substr(beg);
                }
                nFile->replaceUserHost(prtUsrHstStr);
            }
        }
    }

    // if the -l option is specified, we remove all non-local references
    // from the map
    if (makeLocal)
        nFile->makeLocal();

    //
    // in case the -o option is given we assign the output to the file
    // given and the diagnostic output to stdout if not we use the
    // default: the covverted NetFile will be written to stdout and the
    // diagnostic output will be written to stderr
    //

    if (!outFile.empty())
    {

        // check if file exists
        FILE *fd = fopen(outFile.c_str(), "r");
        if (fd)
        {

            cerr << "Warning: output-file " << outFile << " exists " << endl;

            fclose(fd);

            // file exists - set new name
            // we have to append "--conv"
            if (!overWrite)
            {
                outFile = outFile + std::string("--coNew");
                cerr << "         it will be written to " << outFile << endl;
            }
        }

        ofstream oFileS(outFile.c_str());
        if (!oFileS)
        {
            cerr << "map_converter: could not open file "
                 << outFile
                 << " for write" << endl;
            exit(EXIT_FAILURE);
        }
        if (createPythonFile)
        {
            oFileS << *((PyFile *)nFile);
        }
        else
        {
            oFileS << "#" << NET_FILE_VERERSION << endl;
            oFileS << *nFile;
        }
        oFileS.close();
    }
    else if (createPythonFile)
    {
        cout << *((PyFile *)nFile);
    }
    else
    {
        cout << "#" << NET_FILE_VERERSION << endl;
        cout << *nFile;
    }

    delete nFile;
}

// guess what (-;
void usage(char *myName)
{
    fprintf(stderr, "\n");
    version();
    fprintf(stderr, "usage: %s [-l] [-f] [-h] [-P] [-v] [-o <output file>]  \n", myName);
    fprintf(stderr, "                     [-t <translation file>] [-u <userreplace>] <map file>\n\n");
    //fprintf(stderr,"          [-l] [-f] [-h] [-v] [-r <module remove mode>]  <file to convert>\n\n", myName);
    //fprintf(stderr,"       -r <module remove mode>  *** not supported yet ***\n");
    //fprintf(stderr,"          choices for <module remove mode>: \n");
    //fprintf(stderr,"             AUTO : remove all module unknown to map_converter \n");
    //fprintf(stderr,"               NO : don't remove ANY module take the module description as it is ( !DANGEROUS!) \n");
    //fprintf(stderr,"            QUERY : query user if module should be removed for each module individually \n");
    fprintf(stderr, "       -l : 'make local': move all modules to local host\n");
    fprintf(stderr, "            and remove all other hosts from map file\n\n");
    fprintf(stderr, "       -f : force - overwrite existing output files \n\n");
    fprintf(stderr, "       -u : replace users/hosts by those given in <userreplace>:\n");
    fprintf(stderr, "            \"oldUser@oldHost:newUser@newHost,...\"\n\n");
    fprintf(stderr, "       -P : create Python representation of a covise net-file\n\n");
    fprintf(stderr, "       -o : output file, <stdout> if not given\n\n");
    fprintf(stderr, "       -h : print this message\n");
    fprintf(stderr, "       -v : print version information\n\n");
}

void version()
{
    fprintf(stderr, "COVISE Map-Converter v%s  (C) 2002 VISENSO GmbH\n\n", Version.c_str());
}
