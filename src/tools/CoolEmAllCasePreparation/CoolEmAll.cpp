/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <CoolEmAll.h>
#include <algorithm>
#include <sstream>
#include <vector>

#include "FileReference.h"
#include "BC_U.h"
#include "BC_T.h"
#include "BC_alphat.h"
#include "BC_nut.h"
#include "BC_k.h"
#include "BC_epsilon.h"
#include "BC_p.h"
#include "Blockmesh.h"
#include "SnappyHexmeshDict.h"
#include "CoolEmAllClient.h"
#include <osg/Vec3d>
#include <osg/Group>
#include <osg/io_utils>
#include <osg/Matrixd>
#include <cstdio>
#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>
#include "PLMXMLParser.h"
#include <osg/Group>
#include <osg/Node>
#include <osg/io_utils>
#include <osg/Matrixd>
#include <fstream>
#include <sstream>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <set>
#include <string>
#include <FileReference.h>
#include <cmath>

using namespace std;

CoolEmAll::CoolEmAll(std::string f, std::string pp, std::string dp, osg::Group *rn)
{
    //workingDirectory = wd+"/";
    //std::cerr <<"workingDirectory: " << workingDirectory << std::endl;
    databasePrefix = dp;
    PLMXMLFile = f;
    int end = pp.length();
    if (pp.at(end - 1) == '/')
    {
        pathPrefix = pp.substr(0, end - 1);
    }
    else
    {
        pathPrefix = pp;
    }
    //std::cout << "pathPrefix:  " << pathPrefix << std::endl;
    bc_U = new BC_U(this);
    bc_T = new BC_T(this);
    bc_alphat = new BC_alphat(this);
    bc_nut = new BC_nut(this);
    bc_k = new BC_k(this);
    bc_epsilon = new BC_epsilon(this);
    bc_p = new BC_p(this);
    blockmesh = new Blockmesh(this);
    snappyhexmeshdict = new SnappyHexmeshDict(this);
    cc = new CoolEmAllClient("recs1.coolemall.eu");
    rootNode = rn;
    inletTemperatureSumm = 0;
    numInlets = 0;
}
CoolEmAll::~CoolEmAll()
{
    delete bc_U;
    delete bc_T;
    delete bc_alphat;
}
void CoolEmAll::parseFile()
{
    osg::Matrix M;
    M.makeIdentity();
    snappyhexmeshdict->writeHeader();
    bc_U->writeHeader();
    bc_T->writeHeader();
    bc_alphat->writeHeader();
    bc_nut->writeHeader();
    bc_k->writeHeader();
    bc_epsilon->writeHeader();
    bc_p->writeHeader();
    blockmesh->writeHeader();
    std::string rootId;
    findTemperatures(rootNode.get(), M, databasePrefix, rootId);
    M.makeIdentity();
    rootId = "";
    findTransform(rootNode.get(), M, databasePrefix, rootId);
    snappyhexmeshdict->writeFooter();
    bc_U->writeFooter();
    bc_T->writeFooter();
    bc_alphat->writeFooter();
    bc_nut->writeFooter();
    bc_k->writeFooter();
    bc_epsilon->writeFooter();
    bc_p->writeFooter();
    blockmesh->writeFooter();
}

void CoolEmAll::findTemperatures(osg::Node *currNode, osg::Matrix M, std::string DatabasePrefix, std::string PathId, FileReference *lastFileId)
{
    osg::Group *currGroup;
    osg::MatrixTransform *transformMatrix = dynamic_cast<osg::MatrixTransform *>(currNode);
    std::string myName;
    std::string myId;

    if (transformMatrix != NULL)
    {
        osg::Matrix M1 = transformMatrix->getMatrix();
        M = M * M1;
    }
    if (currNode == NULL)
    {
        std::cerr << "no node submitted" << std::endl;
    }
    else
    {
        osg::Referenced *name_tag = currNode->getUserData();
        FileReference *FileReference_Tag = dynamic_cast<FileReference *>(name_tag);
        if (FileReference_Tag != NULL)
        {
            myName = FileReference_Tag->getName();
            myId = FileReference_Tag->getId();

            if (lastFileId == NULL)
            {
                lastFileId = FileReference_Tag;
            }
        }
        osg::Group *currGroup = dynamic_cast<osg::Group *>(currNode);
        if (currGroup != NULL)
        {
            int num_children = currGroup->getNumChildren();
            for (int i = 0; i < num_children; i++)
            {
                FileReference *currentFileId = FileReference_Tag;
                std::string newName = DatabasePrefix;
                std::string newId = PathId;

                if (myId.length() > 0)
                    newId = myId;
                if (myName.length() > 0)
                {
                    if (myName.at(0) != '/')
                    {
                        newName = DatabasePrefix + "/" + myName;
                    }
                    else
                    {
                        newName = DatabasePrefix + myName;
                    }
                }
                if (currentFileId == NULL)
                {
                    currentFileId = lastFileId;
                }
                findTemperatures(currGroup->getChild(i), M, newName, newId, currentFileId);
            }
        }
        else
        {
            std::cerr << "Knoten ist keine Gruppe und damit ein Leaf-Node" << std::endl;
        }
        osg::Referenced *FILE = currNode->getUserData();
        FileReference *UserData = dynamic_cast<FileReference *>(FILE);
        if (UserData != NULL)
        {
            string stl_file = UserData->getFilename("stl");
            std::string DataBase_Path = DatabasePrefix;
            if (myName.length() > 0)
            {
                DataBase_Path = DatabasePrefix + "/" + myName;
            }
            //std::cout << "My Database PATH :" << DataBase_Path << std::endl;
            int laenge = stl_file.length();
            if (laenge > 3 && stl_file[laenge - 3] == 's' && stl_file[laenge - 2] == 't' && stl_file[laenge - 1] == 'l')
            { // this is a ProductRevisionView with an stl file

                std::string DEBBLevel;

                DEBBLevel = UserData->getUserValue("DEBBLevel");
                std::transform(DEBBLevel.begin(), DEBBLevel.end(), DEBBLevel.begin(), ::tolower);
                if (DEBBLevel == "inlet")
                {
                    double temperature = cc->getValue(DataBase_Path, lastFileId->getUserValue("temperature-sensor"));
                    if (temperature != -1)
                    {
                        inletTemperatureSumm += temperature;
                    }
                    else
                        inletTemperatureSumm += 20;
                    numInlets++;
                }
            }
            else
            {
                //this is a product instance
            }
        }
    }
}
void CoolEmAll::findTransform(osg::Node *currNode, osg::Matrix M, std::string DatabasePrefix, std::string PathId, FileReference *lastFileId)
{
    osg::Group *currGroup;
    osg::MatrixTransform *transformMatrix = dynamic_cast<osg::MatrixTransform *>(currNode);
    std::string myName;
    std::string myId;

    if (transformMatrix != NULL)
    {
        osg::Matrix M1 = transformMatrix->getMatrix();
        M = M * M1;
    }
    if (currNode == NULL)
    {
        std::cerr << "no node submitted" << std::endl;
    }
    else
    {
        osg::Referenced *name_tag = currNode->getUserData();
        FileReference *FileReference_Tag = dynamic_cast<FileReference *>(name_tag);
        if (FileReference_Tag != NULL)
        {
            myName = FileReference_Tag->getName();
            myId = FileReference_Tag->getId();

            if (lastFileId == NULL)
            {
                lastFileId = FileReference_Tag;
            }
        }
        osg::Group *currGroup = dynamic_cast<osg::Group *>(currNode);
        if (currGroup != NULL)
        {
            int num_children = currGroup->getNumChildren();
            for (int i = 0; i < num_children; i++)
            {
                FileReference *currentFileId = FileReference_Tag;
                std::string newName = DatabasePrefix;
                std::string newId = PathId;

                if (myId.length() > 0)
                    newId = myId;
                if (myName.length() > 0)
                {
                    if (myName.at(0) != '/')
                    {
                        newName = DatabasePrefix + "/" + myName;
                    }
                    else
                    {
                        newName = DatabasePrefix + myName;
                    }
                }
                if (currentFileId == NULL)
                {
                    currentFileId = lastFileId;
                }
                findTransform(currGroup->getChild(i), M, newName, newId, currentFileId);
            }
        }
        else
        {
            std::cerr << "Knoten ist keine Gruppe und damit ein Leaf-Node" << std::endl;
        }
        osg::Referenced *FILE = currNode->getUserData();
        FileReference *UserData = dynamic_cast<FileReference *>(FILE);
        if (UserData != NULL)
        {
            string stl_file = UserData->getFilename("stl");
            std::string DataBase_Path = DatabasePrefix;
            if (myName.length() > 0)
            {
                DataBase_Path = DatabasePrefix + "/" + myName;
            }
            //std::cout << "My Database PATH :" << DataBase_Path << std::endl;
            int laenge = stl_file.length();
            if (laenge > 3 && stl_file[laenge - 3] == 's' && stl_file[laenge - 2] == 't' && stl_file[laenge - 1] == 'l')
            { // this is a ProductRevisionView with an stl file

                std::string stl_file_trans;
                std::string stl_file_trans_buffer = stl_file.substr(0, stl_file.length() - 4);
                int pos_slash = stl_file_trans_buffer.find_last_of("/");
                stl_file_trans = stl_file_trans_buffer.substr(pos_slash + 1) + "@" + lastFileId->getId() + ".stl";
                //stl_file_trans = "constant/triSurface/"+stl_file_trans_buffer.substr(pos_slash + 1) + "@" + lastFileId->getId() + ".stl";
                transformSTL(stl_file, stl_file_trans, M);
                snappyhexmeshdict->writeSTL(DataBase_Path, UserData, lastFileId, stl_file_trans); //_buffer.substr(pos_slash + 1) + "@" + lastFileId->getId() + ".stl");
                bc_U->writeSTL(DataBase_Path, UserData, lastFileId, stl_file_trans);
                bc_T->writeSTL(DataBase_Path, UserData, lastFileId, stl_file_trans);
                bc_alphat->writeSTL(DataBase_Path, UserData, lastFileId, stl_file_trans);
                bc_nut->writeSTL(DataBase_Path, UserData, lastFileId, stl_file_trans);
                bc_k->writeSTL(DataBase_Path, UserData, lastFileId, stl_file_trans);
                bc_epsilon->writeSTL(DataBase_Path, UserData, lastFileId, stl_file_trans);
                bc_p->writeSTL(DataBase_Path, UserData, lastFileId, stl_file_trans);
                blockmesh->writeBound(UserData);
            }
            //else
            //hier inlet_temperature abfragen
            //this is a product instance
        }
    }
}

bool CoolEmAll::transformSTL(std::string stl_file, std::string stl_file_trans, osg::Matrix M)
{
    Flaeche = 0;
    //std::cout << "from:"<< stl_file << std::endl;
    //int end = pathPrefix.length();
    stl_file_trans = pathPrefix + "/constant/triSurface/" + stl_file_trans;
    //std::cout << "to:"<< stl_file_trans << std::endl;

    std::ifstream file1;
    file1.open(stl_file.c_str());
    if (file1.is_open())
    {
        std::ofstream file2;
        file2.open(stl_file_trans.c_str());
        if (file2.is_open())
        {
            std::string s1, s2, s3;
            file1 >> s1 >> s2;
            file2 << s1 << " " << s2 << std::endl;
            while (file1.good() && !file1.eof())
            {
                s1 = "";
                s2 = "";
                file1 >> s1 >> s2;
                if (s1 == std::string("endsolid"))
                {
                    file2 << s1 << " ";
                    file2 << s2;
                }
                else
                {
                    file2 << s1 << " " << s2 << " ";
                    if (s1 == "facet" && s2 == "normal")
                    {
                        double nx = 0, ny = 0, nz = 0;
                        file1 >> nx >> ny >> nz;
                        osg::Vec3d N(nx, ny, nz);

                        osg::Matrix IM = M;
                        bool test1 = IM.invert(M);
                        if (test1 == true)
                        {
                            //N=IM.preMult(N); 	//geaendert am 7.3.2013 8:30
                            N = osg::Matrix::transform3x3(IM, N);
                            N.normalize();
                        }
                        file2 << scientific << setprecision(6) << N << std::endl;
                    }
                    file1 >> s1 >> s2;
                    file2 << s1 << " " << s2 << std::endl;
                    if (s1 == "outer" && s2 == "loop")
                    {
                        file1 >> s1;
                        file2 << s1 << " ";
                        double P1x = 0, P1y = 0, P1z = 0;
                        file1 >> P1x >> P1y >> P1z;
                        osg::Vec3d P1(P1x, P1y, P1z);
                        P1 = M.preMult(P1);
                        file2 << scientific << setprecision(6) << P1 << std::endl;

                        file1 >> s1;
                        file2 << s1 << " ";
                        double P2x = 0, P2y = 0, P2z = 0;
                        file1 >> P2x >> P2y >> P2z;
                        osg::Vec3d P2(P2x, P2y, P2z);
                        P2 = M.preMult(P2);
                        file2 << scientific << setprecision(6) << P2 << std::endl;

                        file1 >> s1;
                        file2 << s1 << " ";
                        double P3x = 0, P3y = 0, P3z = 0;
                        file1 >> P3x >> P3y >> P3z;
                        osg::Vec3d P3(P3x, P3y, P3z);
                        P3 = M.preMult(P3);
                        file2 << scientific << setprecision(6) << P3 << std::endl;

                        file1 >> s1;
                        file2 << s1 << std::endl;

                        file1 >> s1;
                        file2 << s1 << std::endl;

                        osg::Vec3d L1 = P2 - P1;
                        osg::Vec3d L2 = P3 - P2;
                        osg::Vec3d L3 = P3 - P1;
                        double a = L1.length();
                        double b = L2.length();
                        double c = L3.length();
                        double s = (a + b + c) / 2;
                        double f = sqrt(s * (s - a) * (s - b) * (s - c));
                        Flaeche = Flaeche + f;
                    }
                }
            }
            //std::cerr << "F=" << Flaeche << std::endl;
        }
        else
        {
            std::cerr << "could not open file to store transformed .stl-file" << std::endl;
            return false;
        }
        file2.close();
    }
    else
    {
        std::cerr << "could not open input .stl file, specified in PLMXML-file" << std::endl;
        return false;
    }
    file1.close();

    return true;
}
