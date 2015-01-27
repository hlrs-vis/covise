/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * File:	main.cpp zur PLMXML_Parser.cpp und PLMXML_Parser.h
 * Author: 	hpcdrath
 * Created on 24. Januar 2013
 */

#include "functions.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "no PLMXML-file submitted" << endl;
        exit(1);
    }
    else
    {
        osg::ref_ptr<osg::Group> wurzel(new osg::Group());

        string filename(argv[1]);

        std::string ProductInstanceName;
        std::string ProductInstanceID;
        //cout << filename << endl;
        PLMXMLParser myParser;

        myParser.parse(argv[1], wurzel);

        unsigned int num_child = wurzel->getNumChildren();

        osg::Matrix M;
        M.makeIdentity();

        std::map<std::string, int> fileReferenceMap;
        std::string rootName;
        if (argc == 3)
            rootName = argv[2];
        findTransform(wurzel, M, fileReferenceMap, rootName);
    }
}
