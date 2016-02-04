/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscSourceFile.h"
#include "utilities.h"

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

void oscSourceFile::setSrcFileHref(const std::string &sfhr)
{
    srcFileHref = sfhr;
}

void oscSourceFile::setSrcFileHref(const XMLCh *sfhr)
{
    srcFileHref = xercesc::XMLString::transcode(sfhr);
}

void oscSourceFile::setSrcFileName(const std::string &sfn)
{
    srcFileName = sfn;
}

void oscSourceFile::setMainDocPath(const std::string &mdp)
{
    mainDocPath = mdp;
}

void oscSourceFile::setRelPathFromMainDoc(const std::string &rpfmd)
{
    relPathFromMainDoc = rpfmd;
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
std::string oscSourceFile::getSrcFileHrefAsStr() const
{
    return srcFileHref;
}

const XMLCh *oscSourceFile::getSrcFileHrefAsXmlCh() const
{
    return xercesc::XMLString::transcode(srcFileHref.c_str());
}

std::string oscSourceFile::getSrcFileName() const
{
    return srcFileName;
}

std::string oscSourceFile::getMainDocPath() const
{
    return mainDocPath;
}

std::string oscSourceFile::getRelPathFromMainDoc() const
{
    return relPathFromMainDoc;
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


//
fileNamePath *oscSourceFile::getFileNamePath(const std::string &fnp)
{
    fileNamePath *fileNP = new fileNamePath();
    const std::string FILE = "file://";
    std::string fnpToUse;

    //find possible 'file://' in string and ignore it
    size_t proto = fnp.find(FILE);
    if (proto != std::string::npos)
    {
        fnpToUse = fnp.substr(FILE.length());
    }
    else
    {
        fnpToUse = fnp;
    }

    //find last slash as delimiter (between path and filename)
    size_t delimiter = fnpToUse.find_last_of("/");
    if (delimiter == std::string::npos)
    {
        fileNP->fileName = fnpToUse;
        fileNP->path = "";
    }
    else
    {
        fileNP->fileName = fnpToUse.substr(delimiter + 1);
        //set path with slash as delimiter at the end
        fileNP->path = fnpToUse.substr(0, delimiter + 1);
    }

    return fileNP;
}
