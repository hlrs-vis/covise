/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OPEN_SCENARIO_BASE
#define OPEN_SCENARIO_BASE

#include "oscExport.h"
#include "oscObjectBase.h"

#include "oscFileHeader.h"
#include "oscCatalogs.h"
#include "oscRoadNetwork.h"
#include "oscEnvironmentRef.h"
#include "oscEntities.h"
#include "oscStoryboard.h"
#include "oscScenarioEnd.h"

#include "oscTest.h"

#include <string>
#include <vector>

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

public:
    oscFileHeaderMember fileHeader;
    oscCatalogsMember catalogs;
    oscRoadNetworkMember roadNetwork;
    oscEnvironmentRefMember environment; // temp only, should be a reference
    oscEntitiesMember entities;
    oscStoryboardMember storyboard;
    oscScenarioEndMember scenarioEnd;
	oscTestMember test;

    OpenScenarioBase(); /// constructor, initializes the class and sets a default factory
    ~OpenScenarioBase(); /// destructor, terminate xerces-c

    bool loadFile(const std::string &fileName, const bool validate = true); /*!< load an OpenScenario databas file in xml format
                                                                             \param fileName file to load.
                                                                             \param validate - turn on/off validation
                                                                             \return false if loading the file failed.*/
    bool saveFile(const std::string &fileName, bool overwrite=false);/*!< store an OpenScenario databas to a file in xml format
                                                                      \param fileName file to save to.
                                                                      \param overwrite if set to true, an existing file with the same name is overwritten, otherwise false is retured if a file with that name already exists.
                                                                      \return false if writing to the file failed.*/

    bool writeFileToDisk(xercesc::DOMDocument *xmlDocToWrite, const char *filenameToWrite);
    xercesc::MemBufFormatTarget *writeFileToMemory(xercesc::DOMDocument *xmlDocToWrite);
    xercesc::DOMElement *getRootElement(const std::string &filename, const bool validate = true); ///< init xerces, create validating parser and parse the OpenSCENARIO file with XInclude and validation to a DOM hierarchy
    
    bool parseFromXML(xercesc::DOMElement *currentElement); ///< parses the document, returns true if successfull

    xercesc::DOMDocument *getDocument() const;

    void addToSrcFileVec(oscSourceFile *src);
    std::vector<oscSourceFile *> getSrcFileVec() const;
};

}

#endif //OPEN_SCENARIO_BASE
