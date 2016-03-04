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
    bf::path srcFileHref; ///< reference to the file that is imported (relative path from parent and filename)
    bf::path srcFileName; ///< filename of the imported file
    bf::path mainDocPath; ///< absolute path to the main document
    bf::path relPathFromMainDoc; ///< path from the location of main xosc document to the imported file
    std::string rootElementName; ///< of the file that is read in
    xercesc::DOMDocument *xmlDoc;

public:
    oscSourceFile(); ///< constructor
    ~oscSourceFile(); ///< destructor

    void setSrcFileHref(const bf::path &sfhr);
    void setSrcFileHref(const std::string &sfhr);
    void setSrcFileHref(const XMLCh *sfhr);
    void setSrcFileName(const bf::path &sfn);
    void setSrcFileName(const std::string &sfn);
    void setMainDocPath(const bf::path &mdp);
    void setMainDocPath(const std::string &mdp);
    void setRelPathFromMainDoc(const bf::path &rpfmd);
    void setRelPathFromMainDoc(const std::string &rpfmd);
    void setRootElementName(const std::string &ren);
    void setRootElementName(const XMLCh *ren);
    void setXmlDoc(xercesc::DOMDocument *xD);

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
