/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OPEN_SCENARIO_BASE
#define OPEN_SCENARIO_BASE

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscRoadNetwork.h>
#include <oscHeader.h>
#include <oscEnvironment.h>
#include <string>
#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class DOMDocument;
class DOMElement;
class XercesDOMParser; 
XERCES_CPP_NAMESPACE_END

namespace OpenScenario {

class oscHeader;

/// \class This class represents an OpenScenario database
class OPENSCENARIOEXPORT OpenScenarioBase: public oscObjectBase
{
protected: 
    xercesc::DOMElement *rootElement; ///< DOM Root element
    xercesc::XercesDOMParser *parser; ///< validating parser
    xercesc::DOMDocument *xmlDoc; ///< main xml document
    
public:
    oscHeaderMember header;
    oscHeaderMember database;
    oscRoadNetworkMember roadNetwork;
    oscEnvironmentMember environment; // temp only, should be a reference
    oscHeaderMember entities;
    oscHeaderMember storyboard;
    oscHeaderMember scenarioEnd;


    OpenScenarioBase(); /// constructor, initializes the class and sets a default factory
    ~OpenScenarioBase(); /// destructor, terminate xerces-c

    int loadFile(std::string &fileName); /*!< load an OpenScenario databas file in xml format
                                         \param fileName file to load.
                                         \return false if loading the file failed.*/
    int saveFile(std::string &fileName, bool overwrite=false);/*!< store an OpenScenario databas to a file in xml format
                                                              \param fileName file to save to.
                                                              \param overwrite if set to true, an existing file with the same name is overwritten, otherwise false is retured if a file with that name already exists.
                                                              \return false if writing to the file failed.*/

    xercesc::DOMElement *getRootElement(std::string filename); ///< init xerces, create validating parser and parse the OpenScenario file to a DOM hierarchy

    
    bool parseFromXML(xercesc::DOMElement *currentElement); ///< parses the document, returns true if successfull
    xercesc::DOMDocument *getDocument(){return xmlDoc;};

};

}

#endif //OPEN_SCENARIO_BASE
