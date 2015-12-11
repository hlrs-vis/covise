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
class DOMElement;
XERCES_CPP_NAMESPACE_END


namespace OpenScenario {

class OpenScenarioBase;
class oscObjectBase;

class OPENSCENARIOEXPORT oscSourceFile
{
public:
    oscSourceFile(); ///< constructor
    ~oscSourceFile(); ///< destructor

    void initialize(OpenScenarioBase *b);

    void setVariables(std::string ren, std::string sf = ""); ///< set srcFile and rootElementName when object is used the first time
    void setXmlDoc(xercesc::DOMDocument *xD);
    void setIncludeParentElem(xercesc::DOMElement *inclParentElem);

    std::string getSrcFileName() const;
    std::string getRootElementName() const;
    xercesc::DOMDocument *getXmlDoc() const;
    xercesc::DOMElement *getIncludeParentElem() const;


protected:
    OpenScenarioBase *base;
    std::string srcFileName; ///< file name from which we read
    std::string rootElementName; ///< name of the root element of the file from which is read
    xercesc::DOMDocument *xmlDoc;
    xercesc::DOMElement *includeParentElem;

};

}


#endif /* OSC_SOURCE_FILE_H */
