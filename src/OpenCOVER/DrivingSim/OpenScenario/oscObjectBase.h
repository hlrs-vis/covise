/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_BASE_H
#define OSC_OBJECT_BASE_H

#include "oscExport.h"
#include "oscMember.h"

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
XERCES_CPP_NAMESPACE_END

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>


namespace bf = boost::filesystem;


//varName is of type oscMember
//set the name of varName as string of varName (oscMember[.cpp,.h])
//register varName in MemberMap members ([oscMember,oscObjectBase][.cpp,.h])
//set the type of variable varName ([oscMember,oscMemberValue,oscVariables][.cpp,.h])
//set the type name of varName (element name in xml file) (oscMember[.cpp,.h])
//register varName in MemberChoice choice ([oscMember,oscObjectBase][.cpp,.h])
//register varName in MemberOptional optional ([oscMember,oscObjectBase][.cpp,.h])
#define                          OSC_ADD_MEMBER(varName, choiceNumber) varName.setName(#varName); varName.registerWith(this, choiceNumber); varName.setType(varName.getValueType()) /*OSC_ADD_MEMBER(varName)*/
#define                 OSC_ADD_MEMBER_OPTIONAL(varName, choiceNumber) varName.setName(#varName); varName.registerWith(this, choiceNumber); varName.setType(varName.getValueType()); varName.registerOptionalWith(this) /*OSC_ADD_MEMBER_OPTIONAL(varName)*/
#define          OSC_OBJECT_ADD_MEMBER(varName,typeName, choiceNumber) varName.setName(#varName); varName.registerWith(this, choiceNumber); varName.setType(varName.getValueType()); varName.setTypeName(typeName) /*OSC_OBJECT_ADD_MEMBER(varName,typeName)*/
#define OSC_OBJECT_ADD_MEMBER_OPTIONAL(varName,typeName, choiceNumber) varName.setName(#varName); varName.registerWith(this, choiceNumber); varName.setType(varName.getValueType()); varName.setTypeName(typeName); varName.registerOptionalWith(this) /*OSC_OBJECT_ADD_MEMBER_OPTIONAL(varName,typeName)*/


namespace OpenScenario
{

class OpenScenarioBase;
class oscSourceFile;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObjectBase
{
public:
	 enum ValidationResult
    {
        VAL_invalid = 0x0,
        VAL_optional = 0x1,		// optional and no value set
		VAL_valid = 0x2
    };

	struct MemberElement {
		std::string name;
		oscMember *member;
	};

    typedef std::list<MemberElement> MemberMap;
    typedef std::vector<oscMember *> MemberOptional;

protected:
    MemberMap members; ///< list of all member variables
    OpenScenarioBase *base;
    oscSourceFile *source;
    oscObjectBase *parentObject; ///< the parent of this objectBase
    oscMember *ownMember; ///< the member which store this objectBase as a valueT in oscObjectVariable, oscObjectVariableArray or oscObjectVariableCatalog
    MemberOptional optional;

public:
    oscObjectBase(); ///< constructor
    virtual ~oscObjectBase(); ///< destructor
	virtual const char *getScope() { return ""; }; ///< return parent hierarchie in order to uniquely identify chrildren by name

	// Clone for members //
	//
	void cloneMembers(oscObjectBase *objectBase, oscObjectBase *parentObj, oscMember *ownMember);

    //
    virtual void initialize(OpenScenarioBase *b, oscObjectBase *parentObject, oscMember *ownMember, oscSourceFile *s); ///< params: base, parentObj, ownMem, source
    void addMember(oscMember *m);
    void setBase(OpenScenarioBase *b);
    void setSource(oscSourceFile *s);
    MemberMap getMembers() const;
	oscMember *getMember(const std::string &s) const;

    OpenScenarioBase *getBase() const;
    oscSourceFile *getSource() const;

    //
    void setParentObj(OpenScenarioBase *pObj);
    void setOwnMember(oscMember *om);
    oscObjectBase *getParentObj() const;
    oscMember *getOwnMember() const;

    //
    bool hasChoice() const;
    std::list<oscMember *> getChoice() const;
	void setChosenMember(oscMember *chosenMember);
	oscMember *getChosenMember(unsigned short choiceNumber);

    //
    void addMemberToOptional(oscMember *m);
    bool hasOptional() const;
    MemberOptional getOptional() const;
	bool isMemberOptional(oscMember *m);

	//
	oscObjectBase *getObjectByName(const std::string &name);

    //
    virtual bool parseFromXML(xercesc::DOMElement *currentElement, oscSourceFile *src, bool saveInclude = true);
    virtual bool writeToDOM(xercesc::DOMElement *currentElement, xercesc::DOMDocument *document, bool writeInclude = true);
	virtual void finishedParsing() {}; /// derived classes can implement this to initialize their state after all children have been parsed
	bool writeToDisk();


	oscObjectBase *readDefaultXMLObject(bf::path destFilePath, const std::string &memberName, const std::string &typeName, oscSourceFile *src = NULL);  ///< read default object with specified type and generate an object with source destFilePath

	//void validate(std::string *errorMessage = NULL);   // generate a temporary file and validate the object
	unsigned char validate();

private:
    void addXInclude(xercesc::DOMElement *currElem, xercesc::DOMDocument *doc, const XMLCh *fileHref); ///< during write adds the include node
    oscSourceFile *determineSrcFile(xercesc::DOMElement *memElem, oscSourceFile *srcF); ///< determine which source file to use
};

}

#endif //OSC_OBJECT_BASE_H
