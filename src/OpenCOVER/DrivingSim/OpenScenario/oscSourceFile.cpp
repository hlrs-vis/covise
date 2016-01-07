/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscSourceFile.h"

#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/util/XMLString.hpp>


using namespace OpenScenario;


oscSourceFile::oscSourceFile()
{
    xmlDoc = NULL;
}

oscSourceFile::~oscSourceFile()
{

}



/*****
 * public functions
 *****/

void oscSourceFile::setSrcFileName(const std::string &sf)
{
    srcFileName = sf;
}

void oscSourceFile::setSrcFileName(const XMLCh *sf)
{
    srcFileName = xercesc::XMLString::transcode(sf);
}

void oscSourceFile::setRootElementName(const std::string &ren)
{
    rootElementName = ren;
}

void oscSourceFile::setRootElementName(const XMLCh *ren)
{
    rootElementName = xercesc::XMLString::transcode(ren);
}

void oscSourceFile::setXmlDoc(xercesc::DOMDocument *xD)
{
    xmlDoc = xD;
}


//
std::string oscSourceFile::getSrcFileNameAsStr() const
{
    return srcFileName;
}

const XMLCh *oscSourceFile::getSrcFileNameAsXmlCh() const
{
    return xercesc::XMLString::transcode(srcFileName.c_str());
}

std::string oscSourceFile::getRootElementNameAsStr() const
{
    return rootElementName;
}

const XMLCh *oscSourceFile::getRootElementNameAsXmlCh() const
{
    return xercesc::XMLString::transcode(rootElementName.c_str());
}

xercesc::DOMDocument *oscSourceFile::getXmlDoc() const
{
    return xmlDoc;
}
