/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * File:	main.cpp zur PLMXML_Parser.cpp und PLMXML_Parser.h
 * Author: 	hpcdrath
 * Created on 24. Januar 2013
 */

#include "BC_U.h"
#include "BC_T.h"
#include "CoolEmAllClient.h"
#include "CoolEmAll.h"
#include <stdio.h>
#include <stdlib.h>
#include "PLMXMLParser.h"

#include <osg/Group>
#include <osg/Node>
#include <osg/MatrixTransform>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        cerr << "no PLMXML-file and/or database prefix submitted" << endl;
        exit(1);
    }
    else
    {
        osg::ref_ptr<osg::Group> wurzel(new osg::Group());

        // argv1 = plmxml
        // argv2 = PATHPREFIX
        // argv3 = DATABASEPREFIX
        // argv4 = object to simulate in PLMXML (TODO)

        string filename(argv[1]);

        PLMXMLParser myParser;

        myParser.parse(argv[1], wurzel);

        unsigned int num_child = wurzel->getNumChildren();

        CoolEmAll cool(filename, argv[2], argv[3], wurzel.get());
        cool.parseFile();
    }
}
