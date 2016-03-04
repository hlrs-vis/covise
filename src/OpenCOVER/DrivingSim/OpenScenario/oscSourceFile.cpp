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
    m_xmlDoc = NULL;
}

oscSourceFile::~oscSourceFile()
{

}



/*****
 * public functions
 *****/

void oscSourceFile::setSrcFileHref(const bf::path &srcFileHref)
{
    m_srcFileHref = convertToGenericFormat(srcFileHref);
}

void oscSourceFile::setSrcFileHref(const std::string &srcFileHref)
{
    m_srcFileHref = convertToGenericFormat(srcFileHref);
}

void oscSourceFile::setSrcFileHref(const XMLCh *srcFileHref)
{
    std::string tmpSrcFileHref = xercesc::XMLString::transcode(srcFileHref);
    m_srcFileHref = convertToGenericFormat(tmpSrcFileHref);
}

void oscSourceFile::setSrcFileName(const bf::path &srcFileName)
{
    m_srcFileName = convertToGenericFormat(srcFileName);
}

void oscSourceFile::setSrcFileName(const std::string &srcFileName)
{
    m_srcFileName = convertToGenericFormat(srcFileName);
}

void oscSourceFile::setMainDocPath(const bf::path &mainDocPath)
{
    m_mainDocPath = convertToGenericFormat(mainDocPath);
}

void oscSourceFile::setMainDocPath(const std::string &mainDocPath)
{
    m_mainDocPath = convertToGenericFormat(mainDocPath);
}

void oscSourceFile::setRelPathFromMainDoc(const bf::path &rpfmd)
{
    m_relPathFromMainDoc = convertToGenericFormat(rpfmd);
}

void oscSourceFile::setRelPathFromMainDoc(const std::string &relPathFromMainDoc)
{
    m_relPathFromMainDoc = convertToGenericFormat(relPathFromMainDoc);
}

void oscSourceFile::setRootElementName(const std::string &rootElementName)
{
    m_rootElementName = rootElementName;
}

void oscSourceFile::setRootElementName(const XMLCh *rootElementName)
{
    m_rootElementName = xercesc::XMLString::transcode(rootElementName);
}

void oscSourceFile::setXmlDoc(xercesc::DOMDocument *xmlDoc)
{
    m_xmlDoc = xmlDoc;
}


//
std::string oscSourceFile::getSrcFileHrefAsStr() const
{
    return m_srcFileHref.generic_string();
}

const XMLCh *oscSourceFile::getSrcFileHrefAsXmlCh() const
{
    return xercesc::XMLString::transcode(m_srcFileHref.c_str());
}

bf::path oscSourceFile::getSrcFileName() const
{
    return m_srcFileName;
}

bf::path oscSourceFile::getMainDocPath() const
{
    return m_mainDocPath;
}

bf::path oscSourceFile::getRelPathFromMainDoc() const
{
    return m_relPathFromMainDoc;
}

std::string oscSourceFile::getRootElementNameAsStr() const
{
    return m_rootElementName;
}

const XMLCh *oscSourceFile::getRootElementNameAsXmlCh() const
{
    return xercesc::XMLString::transcode(m_rootElementName.c_str());
}

xercesc::DOMDocument *oscSourceFile::getXmlDoc() const
{
    return m_xmlDoc;
}


//
bf::path oscSourceFile::getFileNamePath(const std::string &fileNamePath)
{
    const std::string FILE = "file://";
    std::string fileNamePathToUse;

    //find possible 'file://' in string and ignore it
    size_t proto = fileNamePath.find(FILE);
    if (proto != std::string::npos)
    {
        fileNamePathToUse = fileNamePath.substr(FILE.length());
    }
    else
    {
        fileNamePathToUse = fileNamePath;
    }

    return convertToGenericFormat(fileNamePathToUse);
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
