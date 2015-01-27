/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
#include <NameId.h>
#include <string>

bool transformSTL(std::string stl_file, std::string stl_file_trans, osg::Matrix M);
void findTransform(osg::Node *currNode, osg::Matrix M, std::map<std::string, int> &fileReferenceMap, std::string pathName, NameId *lastNameId = NULL);
