/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CoolEmAll_H
#define CoolEmAll_H

#include <string>
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
#include <osg/Node>
using namespace covise;
class BC_U;
class BC_T;
class BC_alphat;
class BC_nut;
class BC_k;
class BC_epsilon;
class BC_p;
class Blockmesh;
class CoolEmAllClient;
class FileReference;
class SnappyHexmeshDict;

class CoolEmAll
{
public:
    CoolEmAll(std::string PLMXMLFile, std::string pathPrefix, std::string wd, osg::Group *rootNode);
    ~CoolEmAll();
    void parseFile();
    void findTemperatures(osg::Node *currNode, osg::Matrix M, std::string PathName, std::string PathId, FileReference *lastFileId = NULL);
    void findTransform(osg::Node *currNode, osg::Matrix M, std::string PathName, std::string PathId, FileReference *lastFileId = NULL);
    bool transformSTL(std::string stl_file, std::string stl_file_trans, osg::Matrix M);
    CoolEmAllClient *getCoolEmAllClient()
    {
        return cc;
    };
    double getAverageInletTemperature()
    {
        return inletTemperatureSumm / numInlets;
    }
    double getFlaeche()
    {
        return Flaeche;
    }
    std::string getPathPrefix()
    {
        return pathPrefix;
    }
    std::string getPLMXMLFile()
    {
        return PLMXMLFile;
    }

private:
    std::string PLMXMLFile;
    std::string pathPrefix;
    std::string databasePrefix;
    double Flaeche;
    CoolEmAllClient *cc;
    BC_U *bc_U;
    BC_T *bc_T;
    BC_alphat *bc_alphat;
    BC_nut *bc_nut;
    BC_k *bc_k;
    BC_epsilon *bc_epsilon;
    BC_p *bc_p;
    Blockmesh *blockmesh;
    SnappyHexmeshDict *snappyhexmeshdict;
    double inletTemperatureSumm;
    int numInlets;
    osg::ref_ptr<osg::Group> rootNode;
};

#endif
