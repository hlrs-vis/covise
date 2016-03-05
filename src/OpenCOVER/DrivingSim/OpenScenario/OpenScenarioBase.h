/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OPEN_SCENARIO_BASE_H
#define OPEN_SCENARIO_BASE_H

#include "oscExport.h"
#include "oscObjectBase.h"

#include "oscFileHeader.h"
#include "oscCatalogs.h"
#include "oscRoadNetwork.h"
#include "oscEnvironmentReference.h"
#include "oscEntities.h"
#include "oscStoryboard.h"
#include "oscScenarioEnd.h"

#include "oscTest.h"

#include <string>
#include <vector>
#if __cplusplus >= 201103L || defined WIN32
#include <unordered_map>
using std::unordered_map;
#else
#include <tr1/unordered_map>
using std::tr1::unordered_map;
#endif

#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
class XercesDOMParser;
class MemBufFormatTarget;
XERCES_CPP_NAMESPACE_END


namespace OpenScenario
{

class oscSourceFile;

/// \class This class represents an OpenScenario database
class OPENSCENARIOEXPORT OpenScenarioBase: public oscObjectBase
{
protected: 
    xercesc::DOMElement *rootElement; ///< DOM Root element
    xercesc::XercesDOMParser *parser; ///< validating parser
    xercesc::DOMDocument *xmlDoc; ///< main xml document
    std::vector<oscSourceFile *> srcFileVec; ///< store oscSourceFile of all included and read files
    static bool m_validate;
    static const unordered_map<std::string /*XmlFileType*/, std::string /*XsdFileName*/> m_fileTypeToXsdFileName; ///< XSD Schema file for file type (OpenSCENARIO or catalog objects)

public:
    oscFileHeaderMember fileHeader;
    oscCatalogsMember catalogs;
    oscRoadNetworkMember roadNetwork;
    oscEnvironmentReferenceMember environment;
    oscEntitiesMember entities;
    oscStoryboardMember storyboard;
    oscScenarioEndMember scenarioEnd;
	oscTestMember test;

    OpenScenarioBase(); /// constructor, initializes the class and sets a default factory
    ~OpenScenarioBase(); /// destructor, terminate xerces-c

    void setValidation(const bool validate); ///< turn on/off validation

    bool loadFile(const std::string &fileName, const std::string &fileType); /*!< load an OpenScenario databas file in xml format
                                                                                 \param fileName file to load.
                                                                                 \return false if loading the file failed.*/
    bool saveFile(const std::string &fileName, bool overwrite = false);/*!< store an OpenScenario databas to a file in xml format
                                                                      \param fileName file to save to.
                                                                      \param overwrite if set to true, an existing file with the same name is overwritten, otherwise false is returned if a file with that name already exists.
                                                                      \return false if writing to the file failed.*/

    bool writeFileToDisk(xercesc::DOMDocument *xmlDocToWrite, const char *filenameToWrite);
    xercesc::MemBufFormatTarget *writeFileToMemory(xercesc::DOMDocument *xmlDocToWrite);
    xercesc::DOMElement *getRootElement(const std::string &fileName, const std::string &fileType, const bool validate = m_validate); ///< init xerces, create validating parser and parse an OpenSCENARIO or catalog object file with XInclude and validation to a DOM hierarchy
    
    bool parseFromXML(xercesc::DOMElement *currentElement); ///< parses the document, returns true if successful

    xercesc::DOMDocument *getDocument() const;

    void addToSrcFileVec(oscSourceFile *src);
    std::vector<oscSourceFile *> getSrcFileVec() const;
};

}

#endif //OPEN_SCENARIO_BASE_H
