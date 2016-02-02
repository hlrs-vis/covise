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


namespace OpenScenario
{

struct fileNamePath
{
    fileNamePath() { };
    ~fileNamePath() { };

    std::string fileName;
    std::string path;
};

class OPENSCENARIOEXPORT oscSourceFile
{
protected:
    std::string srcFileHref; ///< reference to the file that is imported (relative path from parent and filename)
    std::string srcFileName; ///< filename of the imported file
    std::string mainDocPath; ///< absolute path to the main document
    std::string relPathFromMainDoc; ///< path from the location of main xosc document to the imported file
    std::string rootElementName; ///< of the file that is read in
    xercesc::DOMDocument *xmlDoc;

public:
    oscSourceFile(); ///< constructor
    ~oscSourceFile(); ///< destructor

    void setSrcFileHref(const std::string &sfhr);
    void setSrcFileHref(const XMLCh *sfhr);
    void setSrcFileName(const std::string &sfn);
    void setMainDocPath(const std::string &mdp);
    void setRelPathFromMainDoc(const std::string &rpfmd);
    void setRootElementName(const std::string &ren);
    void setRootElementName(const XMLCh *ren);
    void setXmlDoc(xercesc::DOMDocument *xD);

    std::string getSrcFileHrefAsStr() const;
    const XMLCh *getSrcFileHrefAsXmlCh() const;
    std::string getSrcFileName() const;
    std::string getMainDocPath() const;
    std::string getRelPathFromMainDoc() const;
    std::string getRootElementNameAsStr() const;
    const XMLCh *getRootElementNameAsXmlCh() const;
    xercesc::DOMDocument *getXmlDoc() const;

    fileNamePath *getFileNamePath(const std::string &fnp); ///< return filename and path with slash as delimiter at the end
};

}

#endif /* OSC_SOURCE_FILE_H */
