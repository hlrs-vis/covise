/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef IOUTIL_H
#define IOUTIL_H

// #include <PLMXMLParser>
#include <osg/Vec3d>
#include <osg/Group>
#include <osg/Node>
#include <osg/io_utils>
#include <osg/Matrixd>
#include <cstdio>
#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
//#include "PLMXMLParser.h"
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <net/covise_socket.h>
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
#include <sys/stat.h>

#define MAX_CUBES 64

#define USE_DATABASE 0

bool
readPLM(const char *xml_file, double (*positionen)[MAX_CUBES][3],
        double (*dimensionen)[MAX_CUBES][3], double (*bounds)[3],
        double (*power)[MAX_CUBES], std::string *dir);

bool
writePLM(char *xml_file);

//	private:
void
    addNode(osg::Node);

void
    delNode(osg::Node);

double getValue(std::string path, std::string var, covise::Host *dbHost);
#endif
