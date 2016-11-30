/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OPEN_SCENARIO_BASE_H
#define OPEN_SCENARIO_BASE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscUtilities.h"

#include "schema/oscFileHeader.h"
#include "schema/oscCatalogs.h"
#include "schema/oscRoadNetwork.h"
#include "schema/oscEntities.h"
#include "schema/oscStoryboard.h"
#include "oscSourceFile.h"

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
public:
    typedef unordered_map<std::string /*OscFileType*/, bf::path /*XsdFileName*/> FileTypeXsdFileNameMap;
	typedef unordered_map<std::string /*OscFileType*/, bf::path /*DefaultFileName*/> DefaultFileTypeNameMap;

protected: 
    static const FileTypeXsdFileNameMap s_fileTypeToXsdFileName; ///< XSD Schema file for file type (OpenSCENARIO or catalog objects)
	static const DefaultFileTypeNameMap s_defaultFileTypeMap;  ///< Default values for file type
    xercesc::XercesDOMParser *parser; ///< validating parser
    ParserErrorHandler *parserErrorHandler; ///< error handler for parser
    xercesc::DOMDocument *xmlDoc; ///< main xml document
    std::vector<oscSourceFile *> srcFileVec; ///< store oscSourceFile of all included and read files
    bool m_validate; ///< turn on/off validation of imported files (OpenSCENARIO and catalog files)
    bool m_fullReadCatalogs; ///< turn on/off full read of catalog objects (in oscObjectBase::parseFromXML()) so that they can be written back into files later
    bf::path m_pathFromCurrentDirToDoc; ///< path from current directory to the file with OpenSCENARIO root element (given by the executable (e.g. `oscTest path/to/mainDir/tescScenario.xosc` or by oddlot))
    bf::path m_xsdPathFileName; ///< store the actual loaded xsd schema grammar file

public:
	oscFileHeaderMember FileHeader;
	oscCatalogsMember Catalogs;
	oscRoadNetworkMember RoadNetwork;
	oscEntitiesMember Entities;
	oscStoryboardMember Storyboard;

    OpenScenarioBase(); /// constructor, initializes the class and sets a default factory, initialize xerces-c, create parser and error handler, generic parser settings
    ~OpenScenarioBase(); /// destructor, delete parser and error handler, terminate xerces-c

    //
    xercesc::DOMDocument *getDocument() const;

    //
    void addToSrcFileVec(oscSourceFile *src);
    std::vector<oscSourceFile *> getSrcFileVec() const;

    //
    void setValidation(const bool validate); ///< turn on/off validation
    bool getValidation() const;
    void setFullReadCatalogs(const bool fullReadCatalogs); ///<  turn on/off full read of catalog objects
    bool getFullReadCatalogs() const;

    //
    void setPathFromCurrentDirToDoc(const bf::path &path);
    void setPathFromCurrentDirToDoc(const std::string &path);
    bf::path getPathFromCurrentDirToDoc() const;

    //
    bool loadFile(const std::string &fileName, const std::string &nodeName, const std::string &fileType); /*!< load an OpenScenario database file in xml format
                                                                                 \param fileName file to load.
                                                                                 \param fileType type of the imported file.
                                                                                 \return false if loading the file failed.*/
    bool saveFile(const std::string &fileName, bool overwrite = false);/*!< store an OpenScenario database to a file in xml format
                                                                      \param fileName file to save to.
                                                                      \param overwrite if set to true, an existing file with the same name is overwritten, otherwise false is returned if a file with that name already exists.
                                                                      \return false if writing to the file failed.*/

	void clearDOM();

    //
    xercesc::MemBufFormatTarget *writeFileToMemory(xercesc::DOMDocument *xmlDocToWrite);
    xercesc::DOMElement *getRootElement(const std::string &fileName, const std::string &nodeName, const std::string &fileType, const bool validate, std::string *errorMessage = NULL); ///< parse an OpenSCENARIO or catalog object file with XInclude and validation to a DOM hierarchy

    //
    bool parseFromXML(xercesc::DOMElement *currentElement); ///< parses the document, returns true if successful
	xercesc::DOMElement *getDefaultXML(const std::string &fileType);
	oscSourceFile *createSource(const std::string &fileName, const std::string &fileType); // create source 
};

}

#endif //OPEN_SCENARIO_BASE_H
