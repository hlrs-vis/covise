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

void oscSourceFile::setPathFromCurrentDirToMainDir (const bf::path &pathFromExeToMainDir)
{
    m_pathFromCurrentDirToMainDir = pathFromExeToMainDir;
}

void oscSourceFile::setPathFromCurrentDirToMainDir (const std::string &pathFromExeToMainDir)
{
    m_pathFromCurrentDirToMainDir = convertToGenericFormat(pathFromExeToMainDir);
}

void oscSourceFile::setAbsPathToMainDir(const bf::path &mainDirPath)
{
    m_absPathToMainDir = convertToGenericFormat(mainDirPath);
}

void oscSourceFile::setAbsPathToMainDir(const std::string &mainDirPath)
{
    m_absPathToMainDir = convertToGenericFormat(mainDirPath);
}

void oscSourceFile::setRelPathFromMainDir(const bf::path &rpfmd)
{
    m_relPathFromMainDir = convertToGenericFormat(rpfmd);
}

void oscSourceFile::setRelPathFromMainDir(const std::string &relPathFromMainDir)
{
    m_relPathFromMainDir = convertToGenericFormat(relPathFromMainDir);
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

bf::path oscSourceFile::getSrcFileHref() const
{
    return m_srcFileHref;
}

bf::path oscSourceFile::getSrcFileName() const
{
    return m_srcFileName;
}

bf::path oscSourceFile::getPathFromCurrentDirToMainDir() const
{
    return m_pathFromCurrentDirToMainDir;
}

bf::path oscSourceFile::getAbsPathToMainDir() const
{
    return m_absPathToMainDir;
}

bf::path oscSourceFile::getRelPathFromMainDir() const
{
    return m_relPathFromMainDir;
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
