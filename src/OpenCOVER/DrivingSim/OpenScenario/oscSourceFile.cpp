/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscSourceFile.h"
#include "oscUtilities.h"

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

void oscSourceFile::setSrcFileHref(const bf::path &sfhr)
{
    srcFileHref = convertToGenericFormat(sfhr);
}

void oscSourceFile::setSrcFileHref(const std::string &sfhr)
{
    srcFileHref = convertToGenericFormat(sfhr);
}

void oscSourceFile::setSrcFileHref(const XMLCh *sfhr)
{
    std::string tmp_sfhr = xercesc::XMLString::transcode(sfhr);
    srcFileHref = convertToGenericFormat(tmp_sfhr);
}

void oscSourceFile::setSrcFileName(const bf::path &sfn)
{
    srcFileName = convertToGenericFormat(sfn);
}

void oscSourceFile::setSrcFileName(const std::string &sfn)
{
    srcFileName = convertToGenericFormat(sfn);
}

void oscSourceFile::setMainDocPath(const bf::path &mdp)
{
    mainDocPath = convertToGenericFormat(mdp);
}

void oscSourceFile::setMainDocPath(const std::string &mdp)
{
    mainDocPath = convertToGenericFormat(mdp);
}

void oscSourceFile::setRelPathFromMainDoc(const bf::path &rpfmd)
{
    relPathFromMainDoc = convertToGenericFormat(rpfmd);
}

void oscSourceFile::setRelPathFromMainDoc(const std::string &rpfmd)
{
    relPathFromMainDoc = convertToGenericFormat(rpfmd);
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
    return srcFileHref.generic_string();
}

const XMLCh *oscSourceFile::getSrcFileHrefAsXmlCh() const
{
    return xercesc::XMLString::transcode(srcFileHref.c_str());
}

bf::path oscSourceFile::getSrcFileName() const
{
    return srcFileName;
}

bf::path oscSourceFile::getMainDocPath() const
{
    return mainDocPath;
}

bf::path oscSourceFile::getRelPathFromMainDoc() const
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
bf::path oscSourceFile::getFileNamePath(const std::string &fnp)
{
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

    return convertToGenericFormat(fnpToUse);
}



/*****
 * private functions
 *****/

bf::path oscSourceFile::convertToGenericFormat(const bf::path &boostPath)
{
    return boostPath.generic_string();
}

bf::path oscSourceFile::convertToGenericFormat(const std::string &strPath)
{
    bf::path pathName = strPath;
    return pathName.generic_string();
}
