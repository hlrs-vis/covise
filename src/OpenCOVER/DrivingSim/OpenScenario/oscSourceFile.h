/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SOURCE_FILE_H
#define OSC_SOURCE_FILE_H

#include "oscExport.h"

#include <string>

#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
XERCES_CPP_NAMESPACE_END

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>


namespace bf = boost::filesystem;


namespace OpenScenario
{

class OPENSCENARIOEXPORT oscSourceFile
{
protected:
    bf::path m_srcFileHref; ///< reference to the file that is imported (relative path from parent and filename)
    bf::path m_srcFileName; ///< filename of the imported file
    bf::path m_mainDocPath; ///< absolute path to the main document
    bf::path m_relPathFromMainDoc; ///< path from the location of main xosc document to the imported file
    std::string m_rootElementName; ///< of the file that is read in
    xercesc::DOMDocument *m_xmlDoc;

public:
    oscSourceFile(); ///< constructor
    ~oscSourceFile(); ///< destructor

    void setSrcFileHref(const bf::path &srcFileHref);
    void setSrcFileHref(const std::string &srcFileHref);
    void setSrcFileHref(const XMLCh *srcFileHref);
    void setSrcFileName(const bf::path &srcFileName);
    void setSrcFileName(const std::string &srcFileName);
    void setMainDocPath(const bf::path &mainDocPath);
    void setMainDocPath(const std::string &mainDocPath);
    void setRelPathFromMainDoc(const bf::path &relPathFromMainDoc);
    void setRelPathFromMainDoc(const std::string &relPathFromMainDoc);
    void setRootElementName(const std::string &rootElementName);
    void setRootElementName(const XMLCh *rootElementName);
    void setXmlDoc(xercesc::DOMDocument *xmlDoc);

    std::string getSrcFileHrefAsStr() const;
    const XMLCh *getSrcFileHrefAsXmlCh() const;
    bf::path getSrcFileName() const;
    bf::path getMainDocPath() const;
    bf::path getRelPathFromMainDoc() const;
    std::string getRootElementNameAsStr() const;
    const XMLCh *getRootElementNameAsXmlCh() const;
    xercesc::DOMDocument *getXmlDoc() const;

    bf::path getFileNamePath(const std::string &fnp); ///< return filename and path without file:// (if present in parameter) at beginning

private:
    bf::path convertToGenericFormat(const bf::path &boostPath); ///< convert a path (bf::path) into generic format with / as delimiter
    bf::path convertToGenericFormat(const std::string &strPath); ///< convert a path (std::string) into generic format with / as delimiter
};

}

#endif /* OSC_SOURCE_FILE_H */
