/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "functions.h"
#include "FileReference.h"

void
findTransform(osg::Node *currNode, osg::Matrix M,
              std::map<std::string, int> &fileReferenceMap, std::string PathName,
              NameId *lastNameId)
{
    osg::Group *currGroup;
    osg::MatrixTransform *transformMatrix = dynamic_cast<osg::MatrixTransform *>(currNode);
    std::string myName;
    std::string id_for_name;

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
        osg::Referenced *name_tag = currNode->getUserData(); //hpcdrath: hinzugefügt 26.2.2013

        NameId *NameId_Tag = dynamic_cast<NameId *>(name_tag);

        if (NameId_Tag != NULL)
        {
            std::cout << "**name_is " << NameId_Tag->getName() << std::endl;
            std::cout << "**id is " << NameId_Tag->getId() << std::endl;
            std::string id_buffer = NameId_Tag->getId();

            myName = NameId_Tag->getName();
        }
        std::cout << "PathName " << PathName << std::endl;
        std::cout << "myName " << myName << std::endl;
        osg::Group *currGroup = dynamic_cast<osg::Group *>(currNode);

        if (currGroup != NULL)
        {
            int num_children = currGroup->getNumChildren();

            for (int i = 0; i < num_children; i++)
            {

                std::cout << "Process Child  " << i << std::endl;
                NameId *currentNameID = NameId_Tag;
                if (currentNameID == NULL)
                    currentNameID = lastNameId;
                std::string newName = PathName;
                if (myName.length() > 0)
                {
                    if (myName.at(0) != '/')
                    {
                        newName = PathName + "/" + myName;
                    }
                    else
                    {
                        newName = PathName + myName;
                    }
                }

                findTransform(currGroup->getChild(i), M, fileReferenceMap,
                              newName, currentNameID);
            }
        }
        else
        {
            std::cerr << "Knoten ist keine Gruppe und damit ein Leaf-Node"
                      << std::endl;
        }

        osg::Referenced *FILE = currNode->getUserData();

        FileReference *FILE_1 = dynamic_cast<FileReference *>(FILE);
        if (FILE_1 != NULL)
        {

            string stl_file = FILE_1->getFilename("stl");
            std::cout << "My Database PATH :" << PathName + "/" + myName
                      << std::endl;

            std::cout << "1_stl_file_1: " << stl_file << std::endl;

            int laenge = stl_file.length();

            if (laenge > 3 && stl_file[laenge - 3] == 's'
                && stl_file[laenge - 2] == 't' && stl_file[laenge - 1] == 'l')
            {
                int file_counter = fileReferenceMap[stl_file];

                std::stringstream ss;

                std::string path;
                //std::string stl_file_trans;

                path = stl_file;

                int l = path.length();
#ifdef _WIN32
                int pos = path.find_last_of("/\\");
#else
                int pos = path.find_last_of("/");
#endif
                pos = pos + 1;

                path = path.substr(0, pos);

                //std::cout << "path: " << path << std::endl;

                //ss << stl_file.substr(0, laenge-4) << "_trans_" << file_counter << stl_file.substr(laenge-4, laenge);

                std::string stl_file_trans; //= ss.str();

                fileReferenceMap[stl_file] = ++file_counter;

                std::string stl_file_trans_buffer = PathName;

                int pos_slash = stl_file_trans_buffer.find_last_of("/");

                stl_file_trans_buffer = stl_file_trans_buffer.substr(
                    pos_slash + 1);

                stl_file_trans = stl_file_trans_buffer + "@" + lastNameId->getId()
                                 + ".stl";

                std::cout << "stl_file_trans_ft: " << stl_file_trans << std::endl;

                transformSTL(stl_file, stl_file_trans, M);
            }
            else
            {
                std::cerr << "no stl-file specified" << endl;
            }
        }
    }
}
