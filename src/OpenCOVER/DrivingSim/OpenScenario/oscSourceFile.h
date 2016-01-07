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


namespace OpenScenario {

class OPENSCENARIOEXPORT oscSourceFile
{
protected:
    std::string srcFileName; ///< file name of the file that is imported
    std::string rootElementName; ///< of the file that is read in
    xercesc::DOMDocument *xmlDoc;

public:
    oscSourceFile(); ///< constructor
    ~oscSourceFile(); ///< destructor

    void setSrcFileName(const std::string &sf);
    void setSrcFileName(const XMLCh *sf);
    void setRootElementName(const std::string &ren);
    void setRootElementName(const XMLCh *ren);
    void setXmlDoc(xercesc::DOMDocument *xD);

    std::string getSrcFileNameAsStr() const;
    const XMLCh *getSrcFileNameAsXmlCh() const;
    std::string getRootElementNameAsStr() const;
    const XMLCh *getRootElementNameAsXmlCh() const;
    xercesc::DOMDocument *getXmlDoc() const;
};

}

#endif /* OSC_SOURCE_FILE_H */
