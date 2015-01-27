//
// Definitions for schema: http://www.w3.org/2004/08/xop/include
//  http://www.w3.org/2004/08/xop/include
//
//
// Constructor for XML Schema item {http://www.w3.org/2004/08/xop/include}Include
//
function XOP_Include () {
    this.typeMarker = 'XOP_Include';
    this._any = [];
    this._href = '';
}

//
// accessor is XOP_Include.prototype.getAny
// element get for any
// - xs:any
// - required element
// - array
//
// element set for any
// setter function is is XOP_Include.prototype.setAny
//
function XOP_Include_getAny() { return this._any;}

XOP_Include.prototype.getAny = XOP_Include_getAny;

function XOP_Include_setAny(value) { this._any = value;}

XOP_Include.prototype.setAny = XOP_Include_setAny;
//
// accessor is XOP_Include.prototype.getHref
// element get for href
// - element type is {http://www.w3.org/2001/XMLSchema}anyURI
// - required element
//
// element set for href
// setter function is is XOP_Include.prototype.setHref
//
function XOP_Include_getHref() { return this._href;}

XOP_Include.prototype.getHref = XOP_Include_getHref;

function XOP_Include_setHref(value) { this._href = value;}

XOP_Include.prototype.setHref = XOP_Include_setHref;
//
// Serialize {http://www.w3.org/2004/08/xop/include}Include
//
function XOP_Include_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    var anyHolder = this._any;
    var anySerializer = null;
    var anyXmlTag = null;
    var anyXmlNsDef = null;
    var anyData = null;
    var anyStartTag;
    if (anyHolder != null && !anyHolder.raw) {
     anySerializer = cxfjsutils.interfaceObject.globalElementSerializers[anyHolder.qname];
     anyXmlTag = 'cxfjsany1:' + anyHolder.localName;
     anyXmlNsDef = 'xmlns:cxfjsany1=\'' + anyHolder.namespaceURI + '\'';
     anyStartTag = '<' + anyXmlTag + ' ' + anyXmlNsDef + '>';
     anyEndTag = '</' + anyXmlTag + '>';
     anyEmptyTag = '<' + anyXmlTag + ' ' + anyXmlNsDef + '/>';
     anyData = anyHolder.object;
    }
    if (anyHolder != null && anyHolder.raw) {
     xml = xml + anyHolder.xml;
    } else {
     if (anyHolder == null || anyData == null) {
      throw 'null value for required any item';
     }
     for (var ax = 0;ax < anyData.length;ax ++) {
      if (anyData[ax] == null) {
       xml = xml + anyEmptyTag;
      } else {
       if (anySerializer) {
        xml = xml + anySerializer.call(anyData[ax], cxfjsutils, anyXmlTag, anyXmlNsDef);
       } else {
        xml = xml + anyStartTag;
        xml = xml + cxfjsutils.escapeXmlEntities(anyData[ax]);
        xml = xml + anyEndTag;
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

XOP_Include.prototype.serialize = XOP_Include_serialize;

function XOP_Include_deserialize (cxfjsutils, element) {
    var newobject = new XOP_Include();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    var anyObject = [];
    var matcher = new org_apache_cxf_any_ns_matcher(org_apache_cxf_any_ns_matcher.OTHER, 'http://www.w3.org/2004/08/xop/include', [], null);
    var anyNeeded = 0;
    var anyAllowed = 9223372036854775807;
    while (anyNeeded > 0 || anyAllowed > 0) {
     var anyURI;
     var anyLocalPart;
     var anyMatched = false;
     if (curElement) {
      anyURI = cxfjsutils.getElementNamespaceURI(curElement);
      anyLocalPart = cxfjsutils.getNodeLocalName(curElement);
      var anyQName = '{' + anyURI + '}' + anyLocalPart;
      cxfjsutils.trace('any match: ' + anyQName);
      anyMatched = matcher.match(anyURI, anyLocalPart)
      cxfjsutils.trace(' --> ' + anyMatched);
     }
     if (anyMatched) {
      anyDeserializer = cxfjsutils.interfaceObject.globalElementDeserializers[anyQName];
      cxfjsutils.trace(' deserializer: ' + anyDeserializer);
      if (anyDeserializer) {
       var anyValue = anyDeserializer(cxfjsutils, curElement);
      } else {
       var anyValue = curElement.nodeValue;
      }
      anyObject.push(anyValue);
      anyNeeded--;
      anyAllowed--;
      curElement = cxfjsutils.getNextElementSibling(curElement);
     } else {
      if (anyNeeded > 0) {
       throw 'not enough ws:any elements';
      }
     }
    }
    var anyHolder = new org_apache_cxf_any_holder(anyURI, anyLocalPart, anyValue);
    newobject.setAny(anyHolder);
    return newobject;
}

//
// Definitions for schema: http://www.hlrs.de/organization/vis/covise
//  file:/home/hpcaiyin/vis/wsdl/COVISE.wsdl#types1
//
//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}OpenNetDoneEvent
//
function COVISE_OpenNetDoneEvent () {
    this.typeMarker = 'COVISE_OpenNetDoneEvent';
    this._type = '';
    this._mapname = '';
}

//
// accessor is COVISE_OpenNetDoneEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_OpenNetDoneEvent.prototype.setType
//
function COVISE_OpenNetDoneEvent_getType() { return this._type;}

COVISE_OpenNetDoneEvent.prototype.getType = COVISE_OpenNetDoneEvent_getType;

function COVISE_OpenNetDoneEvent_setType(value) { this._type = value;}

COVISE_OpenNetDoneEvent.prototype.setType = COVISE_OpenNetDoneEvent_setType;
//
// accessor is COVISE_OpenNetDoneEvent.prototype.getMapname
// element get for mapname
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for mapname
// setter function is is COVISE_OpenNetDoneEvent.prototype.setMapname
//
function COVISE_OpenNetDoneEvent_getMapname() { return this._mapname;}

COVISE_OpenNetDoneEvent.prototype.getMapname = COVISE_OpenNetDoneEvent_getMapname;

function COVISE_OpenNetDoneEvent_setMapname(value) { this._mapname = value;}

COVISE_OpenNetDoneEvent.prototype.setMapname = COVISE_OpenNetDoneEvent_setMapname;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}OpenNetDoneEvent
//
function COVISE_OpenNetDoneEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<mapname>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapname);
     xml = xml + '</mapname>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_OpenNetDoneEvent.prototype.serialize = COVISE_OpenNetDoneEvent_serialize;

function COVISE_OpenNetDoneEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_OpenNetDoneEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapname');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setMapname(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}StringPair
//
function COVISE_StringPair () {
    this.typeMarker = 'COVISE_StringPair';
    this._first = '';
    this._second = '';
}

//
// accessor is COVISE_StringPair.prototype.getFirst
// element get for first
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for first
// setter function is is COVISE_StringPair.prototype.setFirst
//
function COVISE_StringPair_getFirst() { return this._first;}

COVISE_StringPair.prototype.getFirst = COVISE_StringPair_getFirst;

function COVISE_StringPair_setFirst(value) { this._first = value;}

COVISE_StringPair.prototype.setFirst = COVISE_StringPair_setFirst;
//
// accessor is COVISE_StringPair.prototype.getSecond
// element get for second
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for second
// setter function is is COVISE_StringPair.prototype.setSecond
//
function COVISE_StringPair_getSecond() { return this._second;}

COVISE_StringPair.prototype.getSecond = COVISE_StringPair_getSecond;

function COVISE_StringPair_setSecond(value) { this._second = value;}

COVISE_StringPair.prototype.setSecond = COVISE_StringPair_setSecond;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}StringPair
//
function COVISE_StringPair_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<first>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._first);
     xml = xml + '</first>';
    }
    // block for local variables
    {
     xml = xml + '<second>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._second);
     xml = xml + '</second>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_StringPair.prototype.serialize = COVISE_StringPair_serialize;

function COVISE_StringPair_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_StringPair();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing first');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFirst(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing second');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setSecond(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}Event
//
function COVISE_Event () {
    this.typeMarker = 'COVISE_Event';
    this._type = '';
}

//
// accessor is COVISE_Event.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_Event.prototype.setType
//
function COVISE_Event_getType() { return this._type;}

COVISE_Event.prototype.getType = COVISE_Event_getType;

function COVISE_Event_setType(value) { this._type = value;}

COVISE_Event.prototype.setType = COVISE_Event_setType;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}Event
//
function COVISE_Event_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_Event.prototype.serialize = COVISE_Event_serialize;

function COVISE_Event_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_Event();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}FileData
//
function COVISE_FileData () {
    this.typeMarker = 'COVISE_FileData';
    this._Include = null;
    this._contentType = null;
}

//
// accessor is COVISE_FileData.prototype.getInclude
// element get for Include
// - element type is {http://www.w3.org/2004/08/xop/include}Include
// - required element
//
// element set for Include
// setter function is is COVISE_FileData.prototype.setInclude
//
function COVISE_FileData_getInclude() { return this._Include;}

COVISE_FileData.prototype.getInclude = COVISE_FileData_getInclude;

function COVISE_FileData_setInclude(value) { this._Include = value;}

COVISE_FileData.prototype.setInclude = COVISE_FileData_setInclude;
//
// accessor is COVISE_FileData.prototype.getContentType
// element get for contentType
// - element type is null
// - optional element
//
// element set for contentType
// setter function is is COVISE_FileData.prototype.setContentType
//
function COVISE_FileData_getContentType() { return this._contentType;}

COVISE_FileData.prototype.getContentType = COVISE_FileData_getContentType;

function COVISE_FileData_setContentType(value) { this._contentType = value;}

COVISE_FileData.prototype.setContentType = COVISE_FileData_setContentType;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}FileData
//
function COVISE_FileData_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + this._Include.serialize(cxfjsutils, 'jns0:Include', null);
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_FileData.prototype.serialize = COVISE_FileData_serialize;

function COVISE_FileData_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_FileData();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing Include');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = XOP_Include_deserialize(cxfjsutils, curElement);
    }
    newobject.setInclude(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}setParameterResponse
//
function COVISE_setParameterResponse () {
    this.typeMarker = 'COVISE_setParameterResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}setParameterResponse
//
function COVISE_setParameterResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_setParameterResponse.prototype.serialize = COVISE_setParameterResponse_serialize;

function COVISE_setParameterResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_setParameterResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ModuleExecuteStartEvent
//
function COVISE_ModuleExecuteStartEvent () {
    this.typeMarker = 'COVISE_ModuleExecuteStartEvent';
    this._type = '';
    this._moduleID = '';
}

//
// accessor is COVISE_ModuleExecuteStartEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ModuleExecuteStartEvent.prototype.setType
//
function COVISE_ModuleExecuteStartEvent_getType() { return this._type;}

COVISE_ModuleExecuteStartEvent.prototype.getType = COVISE_ModuleExecuteStartEvent_getType;

function COVISE_ModuleExecuteStartEvent_setType(value) { this._type = value;}

COVISE_ModuleExecuteStartEvent.prototype.setType = COVISE_ModuleExecuteStartEvent_setType;
//
// accessor is COVISE_ModuleExecuteStartEvent.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_ModuleExecuteStartEvent.prototype.setModuleID
//
function COVISE_ModuleExecuteStartEvent_getModuleID() { return this._moduleID;}

COVISE_ModuleExecuteStartEvent.prototype.getModuleID = COVISE_ModuleExecuteStartEvent_getModuleID;

function COVISE_ModuleExecuteStartEvent_setModuleID(value) { this._moduleID = value;}

COVISE_ModuleExecuteStartEvent.prototype.setModuleID = COVISE_ModuleExecuteStartEvent_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ModuleExecuteStartEvent
//
function COVISE_ModuleExecuteStartEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ModuleExecuteStartEvent.prototype.serialize = COVISE_ModuleExecuteStartEvent_serialize;

function COVISE_ModuleExecuteStartEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ModuleExecuteStartEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ModuleAddEvent
//
function COVISE_ModuleAddEvent () {
    this.typeMarker = 'COVISE_ModuleAddEvent';
    this._type = '';
    this._module = null;
}

//
// accessor is COVISE_ModuleAddEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ModuleAddEvent.prototype.setType
//
function COVISE_ModuleAddEvent_getType() { return this._type;}

COVISE_ModuleAddEvent.prototype.getType = COVISE_ModuleAddEvent_getType;

function COVISE_ModuleAddEvent_setType(value) { this._type = value;}

COVISE_ModuleAddEvent.prototype.setType = COVISE_ModuleAddEvent_setType;
//
// accessor is COVISE_ModuleAddEvent.prototype.getModule
// element get for module
// - element type is {http://www.hlrs.de/organization/vis/covise}Module
// - required element
//
// element set for module
// setter function is is COVISE_ModuleAddEvent.prototype.setModule
//
function COVISE_ModuleAddEvent_getModule() { return this._module;}

COVISE_ModuleAddEvent.prototype.getModule = COVISE_ModuleAddEvent_getModule;

function COVISE_ModuleAddEvent_setModule(value) { this._module = value;}

COVISE_ModuleAddEvent.prototype.setModule = COVISE_ModuleAddEvent_setModule;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ModuleAddEvent
//
function COVISE_ModuleAddEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + this._module.serialize(cxfjsutils, 'module', null);
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ModuleAddEvent.prototype.serialize = COVISE_ModuleAddEvent_serialize;

function COVISE_ModuleAddEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ModuleAddEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing module');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Module_deserialize(cxfjsutils, curElement);
    }
    newobject.setModule(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ModuleChangeEvent
//
function COVISE_ModuleChangeEvent () {
    this.typeMarker = 'COVISE_ModuleChangeEvent';
    this._type = '';
    this._module = null;
}

//
// accessor is COVISE_ModuleChangeEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ModuleChangeEvent.prototype.setType
//
function COVISE_ModuleChangeEvent_getType() { return this._type;}

COVISE_ModuleChangeEvent.prototype.getType = COVISE_ModuleChangeEvent_getType;

function COVISE_ModuleChangeEvent_setType(value) { this._type = value;}

COVISE_ModuleChangeEvent.prototype.setType = COVISE_ModuleChangeEvent_setType;
//
// accessor is COVISE_ModuleChangeEvent.prototype.getModule
// element get for module
// - element type is {http://www.hlrs.de/organization/vis/covise}Module
// - required element
//
// element set for module
// setter function is is COVISE_ModuleChangeEvent.prototype.setModule
//
function COVISE_ModuleChangeEvent_getModule() { return this._module;}

COVISE_ModuleChangeEvent.prototype.getModule = COVISE_ModuleChangeEvent_getModule;

function COVISE_ModuleChangeEvent_setModule(value) { this._module = value;}

COVISE_ModuleChangeEvent.prototype.setModule = COVISE_ModuleChangeEvent_setModule;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ModuleChangeEvent
//
function COVISE_ModuleChangeEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + this._module.serialize(cxfjsutils, 'module', null);
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ModuleChangeEvent.prototype.serialize = COVISE_ModuleChangeEvent_serialize;

function COVISE_ModuleChangeEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ModuleChangeEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing module');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Module_deserialize(cxfjsutils, curElement);
    }
    newobject.setModule(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}QuitEvent
//
function COVISE_QuitEvent () {
    this.typeMarker = 'COVISE_QuitEvent';
    this._type = '';
}

//
// accessor is COVISE_QuitEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_QuitEvent.prototype.setType
//
function COVISE_QuitEvent_getType() { return this._type;}

COVISE_QuitEvent.prototype.getType = COVISE_QuitEvent_getType;

function COVISE_QuitEvent_setType(value) { this._type = value;}

COVISE_QuitEvent.prototype.setType = COVISE_QuitEvent_setType;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}QuitEvent
//
function COVISE_QuitEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_QuitEvent.prototype.serialize = COVISE_QuitEvent_serialize;

function COVISE_QuitEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_QuitEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getEventResponse
//
function COVISE_getEventResponse () {
    this.typeMarker = 'COVISE_getEventResponse';
    this._event = null;
    this._uuid = '';
}

//
// accessor is COVISE_getEventResponse.prototype.getEvent
// element get for event
// - element type is {http://www.hlrs.de/organization/vis/covise}Event
// - optional element
// - nillable
//
// element set for event
// setter function is is COVISE_getEventResponse.prototype.setEvent
//
function COVISE_getEventResponse_getEvent() { return this._event;}

COVISE_getEventResponse.prototype.getEvent = COVISE_getEventResponse_getEvent;

function COVISE_getEventResponse_setEvent(value) { this._event = value;}

COVISE_getEventResponse.prototype.setEvent = COVISE_getEventResponse_setEvent;
//
// accessor is COVISE_getEventResponse.prototype.getUuid
// element get for uuid
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for uuid
// setter function is is COVISE_getEventResponse.prototype.setUuid
//
function COVISE_getEventResponse_getUuid() { return this._uuid;}

COVISE_getEventResponse.prototype.getUuid = COVISE_getEventResponse_getUuid;

function COVISE_getEventResponse_setUuid(value) { this._uuid = value;}

COVISE_getEventResponse.prototype.setUuid = COVISE_getEventResponse_setUuid;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getEventResponse
//
function COVISE_getEventResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     if (this._event != null) {
      if (this._event == null) {
       xml = xml + '<event xsi:nil=\'true\'/>';
      } else {
       xml = xml + this._event.serialize(cxfjsutils, 'event', null);
      }
     }
    }
    // block for local variables
    {
     xml = xml + '<uuid>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._uuid);
     xml = xml + '</uuid>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getEventResponse.prototype.serialize = COVISE_getEventResponse_serialize;

function COVISE_getEventResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getEventResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing event');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'event')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      item = COVISE_Event_deserialize(cxfjsutils, curElement);
     }
     newobject.setEvent(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing uuid');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setUuid(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}createNewDir
//
function COVISE_createNewDir () {
    this.typeMarker = 'COVISE_createNewDir';
    this._path = '';
    this._newDir = '';
}

//
// accessor is COVISE_createNewDir.prototype.getPath
// element get for path
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for path
// setter function is is COVISE_createNewDir.prototype.setPath
//
function COVISE_createNewDir_getPath() { return this._path;}

COVISE_createNewDir.prototype.getPath = COVISE_createNewDir_getPath;

function COVISE_createNewDir_setPath(value) { this._path = value;}

COVISE_createNewDir.prototype.setPath = COVISE_createNewDir_setPath;
//
// accessor is COVISE_createNewDir.prototype.getNewDir
// element get for newDir
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for newDir
// setter function is is COVISE_createNewDir.prototype.setNewDir
//
function COVISE_createNewDir_getNewDir() { return this._newDir;}

COVISE_createNewDir.prototype.getNewDir = COVISE_createNewDir_getNewDir;

function COVISE_createNewDir_setNewDir(value) { this._newDir = value;}

COVISE_createNewDir.prototype.setNewDir = COVISE_createNewDir_setNewDir;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}createNewDir
//
function COVISE_createNewDir_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<path>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._path);
     xml = xml + '</path>';
    }
    // block for local variables
    {
     xml = xml + '<newDir>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._newDir);
     xml = xml + '</newDir>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_createNewDir.prototype.serialize = COVISE_createNewDir_serialize;

function COVISE_createNewDir_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_createNewDir();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing path');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setPath(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing newDir');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setNewDir(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}LinkAddEvent
//
function COVISE_LinkAddEvent () {
    this.typeMarker = 'COVISE_LinkAddEvent';
    this._type = '';
    this._link = null;
}

//
// accessor is COVISE_LinkAddEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_LinkAddEvent.prototype.setType
//
function COVISE_LinkAddEvent_getType() { return this._type;}

COVISE_LinkAddEvent.prototype.getType = COVISE_LinkAddEvent_getType;

function COVISE_LinkAddEvent_setType(value) { this._type = value;}

COVISE_LinkAddEvent.prototype.setType = COVISE_LinkAddEvent_setType;
//
// accessor is COVISE_LinkAddEvent.prototype.getLink
// element get for link
// - element type is {http://www.hlrs.de/organization/vis/covise}Link
// - required element
//
// element set for link
// setter function is is COVISE_LinkAddEvent.prototype.setLink
//
function COVISE_LinkAddEvent_getLink() { return this._link;}

COVISE_LinkAddEvent.prototype.getLink = COVISE_LinkAddEvent_getLink;

function COVISE_LinkAddEvent_setLink(value) { this._link = value;}

COVISE_LinkAddEvent.prototype.setLink = COVISE_LinkAddEvent_setLink;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}LinkAddEvent
//
function COVISE_LinkAddEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + this._link.serialize(cxfjsutils, 'link', null);
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_LinkAddEvent.prototype.serialize = COVISE_LinkAddEvent_serialize;

function COVISE_LinkAddEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_LinkAddEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing link');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Link_deserialize(cxfjsutils, curElement);
    }
    newobject.setLink(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getParameterAsString
//
function COVISE_getParameterAsString () {
    this.typeMarker = 'COVISE_getParameterAsString';
    this._moduleID = '';
    this._parameter = '';
}

//
// accessor is COVISE_getParameterAsString.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_getParameterAsString.prototype.setModuleID
//
function COVISE_getParameterAsString_getModuleID() { return this._moduleID;}

COVISE_getParameterAsString.prototype.getModuleID = COVISE_getParameterAsString_getModuleID;

function COVISE_getParameterAsString_setModuleID(value) { this._moduleID = value;}

COVISE_getParameterAsString.prototype.setModuleID = COVISE_getParameterAsString_setModuleID;
//
// accessor is COVISE_getParameterAsString.prototype.getParameter
// element get for parameter
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for parameter
// setter function is is COVISE_getParameterAsString.prototype.setParameter
//
function COVISE_getParameterAsString_getParameter() { return this._parameter;}

COVISE_getParameterAsString.prototype.getParameter = COVISE_getParameterAsString_getParameter;

function COVISE_getParameterAsString_setParameter(value) { this._parameter = value;}

COVISE_getParameterAsString.prototype.setParameter = COVISE_getParameterAsString_setParameter;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getParameterAsString
//
function COVISE_getParameterAsString_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    // block for local variables
    {
     xml = xml + '<parameter>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._parameter);
     xml = xml + '</parameter>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getParameterAsString.prototype.serialize = COVISE_getParameterAsString_serialize;

function COVISE_getParameterAsString_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getParameterAsString();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing parameter');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setParameter(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getModuleID
//
function COVISE_getModuleID () {
    this.typeMarker = 'COVISE_getModuleID';
    this._module = '';
    this._instance = '';
    this._host = '';
}

//
// accessor is COVISE_getModuleID.prototype.getModule
// element get for module
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for module
// setter function is is COVISE_getModuleID.prototype.setModule
//
function COVISE_getModuleID_getModule() { return this._module;}

COVISE_getModuleID.prototype.getModule = COVISE_getModuleID_getModule;

function COVISE_getModuleID_setModule(value) { this._module = value;}

COVISE_getModuleID.prototype.setModule = COVISE_getModuleID_setModule;
//
// accessor is COVISE_getModuleID.prototype.getInstance
// element get for instance
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for instance
// setter function is is COVISE_getModuleID.prototype.setInstance
//
function COVISE_getModuleID_getInstance() { return this._instance;}

COVISE_getModuleID.prototype.getInstance = COVISE_getModuleID_getInstance;

function COVISE_getModuleID_setInstance(value) { this._instance = value;}

COVISE_getModuleID.prototype.setInstance = COVISE_getModuleID_setInstance;
//
// accessor is COVISE_getModuleID.prototype.getHost
// element get for host
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for host
// setter function is is COVISE_getModuleID.prototype.setHost
//
function COVISE_getModuleID_getHost() { return this._host;}

COVISE_getModuleID.prototype.getHost = COVISE_getModuleID_getHost;

function COVISE_getModuleID_setHost(value) { this._host = value;}

COVISE_getModuleID.prototype.setHost = COVISE_getModuleID_setHost;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getModuleID
//
function COVISE_getModuleID_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<module>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._module);
     xml = xml + '</module>';
    }
    // block for local variables
    {
     xml = xml + '<instance>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._instance);
     xml = xml + '</instance>';
    }
    // block for local variables
    {
     xml = xml + '<host>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._host);
     xml = xml + '</host>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getModuleID.prototype.serialize = COVISE_getModuleID_serialize;

function COVISE_getModuleID_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getModuleID();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing module');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModule(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing instance');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setInstance(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing host');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setHost(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}isFileExistResponse
//
function COVISE_isFileExistResponse () {
    this.typeMarker = 'COVISE_isFileExistResponse';
    this._result = '';
    this._isFileExist = '';
}

//
// accessor is COVISE_isFileExistResponse.prototype.getResult
// element get for result
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for result
// setter function is is COVISE_isFileExistResponse.prototype.setResult
//
function COVISE_isFileExistResponse_getResult() { return this._result;}

COVISE_isFileExistResponse.prototype.getResult = COVISE_isFileExistResponse_getResult;

function COVISE_isFileExistResponse_setResult(value) { this._result = value;}

COVISE_isFileExistResponse.prototype.setResult = COVISE_isFileExistResponse_setResult;
//
// accessor is COVISE_isFileExistResponse.prototype.getIsFileExist
// element get for isFileExist
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for isFileExist
// setter function is is COVISE_isFileExistResponse.prototype.setIsFileExist
//
function COVISE_isFileExistResponse_getIsFileExist() { return this._isFileExist;}

COVISE_isFileExistResponse.prototype.getIsFileExist = COVISE_isFileExistResponse_getIsFileExist;

function COVISE_isFileExistResponse_setIsFileExist(value) { this._isFileExist = value;}

COVISE_isFileExistResponse.prototype.setIsFileExist = COVISE_isFileExistResponse_setIsFileExist;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}isFileExistResponse
//
function COVISE_isFileExistResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<result>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._result);
     xml = xml + '</result>';
    }
    // block for local variables
    {
     xml = xml + '<isFileExist>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._isFileExist);
     xml = xml + '</isFileExist>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_isFileExistResponse.prototype.serialize = COVISE_isFileExistResponse_serialize;

function COVISE_isFileExistResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_isFileExistResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing result');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setResult(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing isFileExist');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setIsFileExist(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}instantiateModuleResponse
//
function COVISE_instantiateModuleResponse () {
    this.typeMarker = 'COVISE_instantiateModuleResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}instantiateModuleResponse
//
function COVISE_instantiateModuleResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_instantiateModuleResponse.prototype.serialize = COVISE_instantiateModuleResponse_serialize;

function COVISE_instantiateModuleResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_instantiateModuleResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}FloatSliderParameter
//
function COVISE_FloatSliderParameter () {
    this.typeMarker = 'COVISE_FloatSliderParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = 0.0;
    this._min = 0.0;
    this._max = 0.0;
}

//
// accessor is COVISE_FloatSliderParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_FloatSliderParameter.prototype.setName
//
function COVISE_FloatSliderParameter_getName() { return this._name;}

COVISE_FloatSliderParameter.prototype.getName = COVISE_FloatSliderParameter_getName;

function COVISE_FloatSliderParameter_setName(value) { this._name = value;}

COVISE_FloatSliderParameter.prototype.setName = COVISE_FloatSliderParameter_setName;
//
// accessor is COVISE_FloatSliderParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_FloatSliderParameter.prototype.setType
//
function COVISE_FloatSliderParameter_getType() { return this._type;}

COVISE_FloatSliderParameter.prototype.getType = COVISE_FloatSliderParameter_getType;

function COVISE_FloatSliderParameter_setType(value) { this._type = value;}

COVISE_FloatSliderParameter.prototype.setType = COVISE_FloatSliderParameter_setType;
//
// accessor is COVISE_FloatSliderParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_FloatSliderParameter.prototype.setDescription
//
function COVISE_FloatSliderParameter_getDescription() { return this._description;}

COVISE_FloatSliderParameter.prototype.getDescription = COVISE_FloatSliderParameter_getDescription;

function COVISE_FloatSliderParameter_setDescription(value) { this._description = value;}

COVISE_FloatSliderParameter.prototype.setDescription = COVISE_FloatSliderParameter_setDescription;
//
// accessor is COVISE_FloatSliderParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_FloatSliderParameter.prototype.setMapped
//
function COVISE_FloatSliderParameter_getMapped() { return this._mapped;}

COVISE_FloatSliderParameter.prototype.getMapped = COVISE_FloatSliderParameter_getMapped;

function COVISE_FloatSliderParameter_setMapped(value) { this._mapped = value;}

COVISE_FloatSliderParameter.prototype.setMapped = COVISE_FloatSliderParameter_setMapped;
//
// accessor is COVISE_FloatSliderParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for value
// setter function is is COVISE_FloatSliderParameter.prototype.setValue
//
function COVISE_FloatSliderParameter_getValue() { return this._value;}

COVISE_FloatSliderParameter.prototype.getValue = COVISE_FloatSliderParameter_getValue;

function COVISE_FloatSliderParameter_setValue(value) { this._value = value;}

COVISE_FloatSliderParameter.prototype.setValue = COVISE_FloatSliderParameter_setValue;
//
// accessor is COVISE_FloatSliderParameter.prototype.getMin
// element get for min
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for min
// setter function is is COVISE_FloatSliderParameter.prototype.setMin
//
function COVISE_FloatSliderParameter_getMin() { return this._min;}

COVISE_FloatSliderParameter.prototype.getMin = COVISE_FloatSliderParameter_getMin;

function COVISE_FloatSliderParameter_setMin(value) { this._min = value;}

COVISE_FloatSliderParameter.prototype.setMin = COVISE_FloatSliderParameter_setMin;
//
// accessor is COVISE_FloatSliderParameter.prototype.getMax
// element get for max
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for max
// setter function is is COVISE_FloatSliderParameter.prototype.setMax
//
function COVISE_FloatSliderParameter_getMax() { return this._max;}

COVISE_FloatSliderParameter.prototype.getMax = COVISE_FloatSliderParameter_getMax;

function COVISE_FloatSliderParameter_setMax(value) { this._max = value;}

COVISE_FloatSliderParameter.prototype.setMax = COVISE_FloatSliderParameter_setMax;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}FloatSliderParameter
//
function COVISE_FloatSliderParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    // block for local variables
    {
     xml = xml + '<min>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._min);
     xml = xml + '</min>';
    }
    // block for local variables
    {
     xml = xml + '<max>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._max);
     xml = xml + '</max>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_FloatSliderParameter.prototype.serialize = COVISE_FloatSliderParameter_serialize;

function COVISE_FloatSliderParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_FloatSliderParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing min');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setMin(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing max');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setMax(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ExecuteStartEvent
//
function COVISE_ExecuteStartEvent () {
    this.typeMarker = 'COVISE_ExecuteStartEvent';
    this._type = '';
}

//
// accessor is COVISE_ExecuteStartEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ExecuteStartEvent.prototype.setType
//
function COVISE_ExecuteStartEvent_getType() { return this._type;}

COVISE_ExecuteStartEvent.prototype.getType = COVISE_ExecuteStartEvent_getType;

function COVISE_ExecuteStartEvent_setType(value) { this._type = value;}

COVISE_ExecuteStartEvent.prototype.setType = COVISE_ExecuteStartEvent_setType;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ExecuteStartEvent
//
function COVISE_ExecuteStartEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ExecuteStartEvent.prototype.serialize = COVISE_ExecuteStartEvent_serialize;

function COVISE_ExecuteStartEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ExecuteStartEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ParameterChangeEvent
//
function COVISE_ParameterChangeEvent () {
    this.typeMarker = 'COVISE_ParameterChangeEvent';
    this._type = '';
    this._moduleID = '';
    this._parameter = null;
}

//
// accessor is COVISE_ParameterChangeEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ParameterChangeEvent.prototype.setType
//
function COVISE_ParameterChangeEvent_getType() { return this._type;}

COVISE_ParameterChangeEvent.prototype.getType = COVISE_ParameterChangeEvent_getType;

function COVISE_ParameterChangeEvent_setType(value) { this._type = value;}

COVISE_ParameterChangeEvent.prototype.setType = COVISE_ParameterChangeEvent_setType;
//
// accessor is COVISE_ParameterChangeEvent.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_ParameterChangeEvent.prototype.setModuleID
//
function COVISE_ParameterChangeEvent_getModuleID() { return this._moduleID;}

COVISE_ParameterChangeEvent.prototype.getModuleID = COVISE_ParameterChangeEvent_getModuleID;

function COVISE_ParameterChangeEvent_setModuleID(value) { this._moduleID = value;}

COVISE_ParameterChangeEvent.prototype.setModuleID = COVISE_ParameterChangeEvent_setModuleID;
//
// accessor is COVISE_ParameterChangeEvent.prototype.getParameter
// element get for parameter
// - element type is {http://www.hlrs.de/organization/vis/covise}Parameter
// - required element
// - nillable
//
// element set for parameter
// setter function is is COVISE_ParameterChangeEvent.prototype.setParameter
//
function COVISE_ParameterChangeEvent_getParameter() { return this._parameter;}

COVISE_ParameterChangeEvent.prototype.getParameter = COVISE_ParameterChangeEvent_getParameter;

function COVISE_ParameterChangeEvent_setParameter(value) { this._parameter = value;}

COVISE_ParameterChangeEvent.prototype.setParameter = COVISE_ParameterChangeEvent_setParameter;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ParameterChangeEvent
//
function COVISE_ParameterChangeEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    // block for local variables
    {
     if (this._parameter == null) {
      xml = xml + '<parameter xsi:nil=\'true\'/>';
     } else {
      xml = xml + this._parameter.serialize(cxfjsutils, 'parameter', null);
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ParameterChangeEvent.prototype.serialize = COVISE_ParameterChangeEvent_serialize;

function COVISE_ParameterChangeEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ParameterChangeEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing parameter');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Parameter_deserialize(cxfjsutils, curElement);
    }
    newobject.setParameter(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getParameterAsStringResponse
//
function COVISE_getParameterAsStringResponse () {
    this.typeMarker = 'COVISE_getParameterAsStringResponse';
    this._value = '';
}

//
// accessor is COVISE_getParameterAsStringResponse.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for value
// setter function is is COVISE_getParameterAsStringResponse.prototype.setValue
//
function COVISE_getParameterAsStringResponse_getValue() { return this._value;}

COVISE_getParameterAsStringResponse.prototype.getValue = COVISE_getParameterAsStringResponse_getValue;

function COVISE_getParameterAsStringResponse_setValue(value) { this._value = value;}

COVISE_getParameterAsStringResponse.prototype.setValue = COVISE_getParameterAsStringResponse_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getParameterAsStringResponse
//
function COVISE_getParameterAsStringResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getParameterAsStringResponse.prototype.serialize = COVISE_getParameterAsStringResponse_serialize;

function COVISE_getParameterAsStringResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getParameterAsStringResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getFileInfoList
//
function COVISE_getFileInfoList () {
    this.typeMarker = 'COVISE_getFileInfoList';
    this._path = '';
}

//
// accessor is COVISE_getFileInfoList.prototype.getPath
// element get for path
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for path
// setter function is is COVISE_getFileInfoList.prototype.setPath
//
function COVISE_getFileInfoList_getPath() { return this._path;}

COVISE_getFileInfoList.prototype.getPath = COVISE_getFileInfoList_getPath;

function COVISE_getFileInfoList_setPath(value) { this._path = value;}

COVISE_getFileInfoList.prototype.setPath = COVISE_getFileInfoList_setPath;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getFileInfoList
//
function COVISE_getFileInfoList_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<path>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._path);
     xml = xml + '</path>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getFileInfoList.prototype.serialize = COVISE_getFileInfoList_serialize;

function COVISE_getFileInfoList_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getFileInfoList();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing path');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setPath(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}executeModuleResponse
//
function COVISE_executeModuleResponse () {
    this.typeMarker = 'COVISE_executeModuleResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}executeModuleResponse
//
function COVISE_executeModuleResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_executeModuleResponse.prototype.serialize = COVISE_executeModuleResponse_serialize;

function COVISE_executeModuleResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_executeModuleResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getRunningModules
//
function COVISE_getRunningModules () {
    this.typeMarker = 'COVISE_getRunningModules';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}getRunningModules
//
function COVISE_getRunningModules_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getRunningModules.prototype.serialize = COVISE_getRunningModules_serialize;

function COVISE_getRunningModules_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getRunningModules();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getRunningModulesResponse
//
function COVISE_getRunningModulesResponse () {
    this.typeMarker = 'COVISE_getRunningModulesResponse';
    this._modules = [];
    this._networkFile = '';
}

//
// accessor is COVISE_getRunningModulesResponse.prototype.getModules
// element get for modules
// - element type is {http://www.hlrs.de/organization/vis/covise}Module
// - required element
// - array
//
// element set for modules
// setter function is is COVISE_getRunningModulesResponse.prototype.setModules
//
function COVISE_getRunningModulesResponse_getModules() { return this._modules;}

COVISE_getRunningModulesResponse.prototype.getModules = COVISE_getRunningModulesResponse_getModules;

function COVISE_getRunningModulesResponse_setModules(value) { this._modules = value;}

COVISE_getRunningModulesResponse.prototype.setModules = COVISE_getRunningModulesResponse_setModules;
//
// accessor is COVISE_getRunningModulesResponse.prototype.getNetworkFile
// element get for networkFile
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for networkFile
// setter function is is COVISE_getRunningModulesResponse.prototype.setNetworkFile
//
function COVISE_getRunningModulesResponse_getNetworkFile() { return this._networkFile;}

COVISE_getRunningModulesResponse.prototype.getNetworkFile = COVISE_getRunningModulesResponse_getNetworkFile;

function COVISE_getRunningModulesResponse_setNetworkFile(value) { this._networkFile = value;}

COVISE_getRunningModulesResponse.prototype.setNetworkFile = COVISE_getRunningModulesResponse_setNetworkFile;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getRunningModulesResponse
//
function COVISE_getRunningModulesResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     if (this._modules != null) {
      for (var ax = 0;ax < this._modules.length;ax ++) {
       if (this._modules[ax] == null) {
        xml = xml + '<modules/>';
       } else {
        xml = xml + this._modules[ax].serialize(cxfjsutils, 'modules', null);
       }
      }
     }
    }
    // block for local variables
    {
     xml = xml + '<networkFile>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._networkFile);
     xml = xml + '</networkFile>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getRunningModulesResponse.prototype.serialize = COVISE_getRunningModulesResponse_serialize;

function COVISE_getRunningModulesResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getRunningModulesResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing modules');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'modules')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_Module_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'modules'));
     newobject.setModules(item);
     var item = null;
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing networkFile');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setNetworkFile(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getModuleIDResponse
//
function COVISE_getModuleIDResponse () {
    this.typeMarker = 'COVISE_getModuleIDResponse';
    this._moduleID = '';
}

//
// accessor is COVISE_getModuleIDResponse.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_getModuleIDResponse.prototype.setModuleID
//
function COVISE_getModuleIDResponse_getModuleID() { return this._moduleID;}

COVISE_getModuleIDResponse.prototype.getModuleID = COVISE_getModuleIDResponse_getModuleID;

function COVISE_getModuleIDResponse_setModuleID(value) { this._moduleID = value;}

COVISE_getModuleIDResponse.prototype.setModuleID = COVISE_getModuleIDResponse_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getModuleIDResponse
//
function COVISE_getModuleIDResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getModuleIDResponse.prototype.serialize = COVISE_getModuleIDResponse_serialize;

function COVISE_getModuleIDResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getModuleIDResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}isFileExist
//
function COVISE_isFileExist () {
    this.typeMarker = 'COVISE_isFileExist';
    this._path = '';
    this._fileName = '';
}

//
// accessor is COVISE_isFileExist.prototype.getPath
// element get for path
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for path
// setter function is is COVISE_isFileExist.prototype.setPath
//
function COVISE_isFileExist_getPath() { return this._path;}

COVISE_isFileExist.prototype.getPath = COVISE_isFileExist_getPath;

function COVISE_isFileExist_setPath(value) { this._path = value;}

COVISE_isFileExist.prototype.setPath = COVISE_isFileExist_setPath;
//
// accessor is COVISE_isFileExist.prototype.getFileName
// element get for fileName
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for fileName
// setter function is is COVISE_isFileExist.prototype.setFileName
//
function COVISE_isFileExist_getFileName() { return this._fileName;}

COVISE_isFileExist.prototype.getFileName = COVISE_isFileExist_getFileName;

function COVISE_isFileExist_setFileName(value) { this._fileName = value;}

COVISE_isFileExist.prototype.setFileName = COVISE_isFileExist_setFileName;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}isFileExist
//
function COVISE_isFileExist_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<path>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._path);
     xml = xml + '</path>';
    }
    // block for local variables
    {
     xml = xml + '<fileName>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileName);
     xml = xml + '</fileName>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_isFileExist.prototype.serialize = COVISE_isFileExist_serialize;

function COVISE_isFileExist_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_isFileExist();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing path');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setPath(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileName');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFileName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getLinksResponse
//
function COVISE_getLinksResponse () {
    this.typeMarker = 'COVISE_getLinksResponse';
    this._links = [];
}

//
// accessor is COVISE_getLinksResponse.prototype.getLinks
// element get for links
// - element type is {http://www.hlrs.de/organization/vis/covise}Link
// - required element
// - array
//
// element set for links
// setter function is is COVISE_getLinksResponse.prototype.setLinks
//
function COVISE_getLinksResponse_getLinks() { return this._links;}

COVISE_getLinksResponse.prototype.getLinks = COVISE_getLinksResponse_getLinks;

function COVISE_getLinksResponse_setLinks(value) { this._links = value;}

COVISE_getLinksResponse.prototype.setLinks = COVISE_getLinksResponse_setLinks;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getLinksResponse
//
function COVISE_getLinksResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     if (this._links != null) {
      for (var ax = 0;ax < this._links.length;ax ++) {
       if (this._links[ax] == null) {
        xml = xml + '<links/>';
       } else {
        xml = xml + this._links[ax].serialize(cxfjsutils, 'links', null);
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getLinksResponse.prototype.serialize = COVISE_getLinksResponse_serialize;

function COVISE_getLinksResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getLinksResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing links');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'links')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_Link_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'links'));
     newobject.setLinks(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}quitResponse
//
function COVISE_quitResponse () {
    this.typeMarker = 'COVISE_quitResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}quitResponse
//
function COVISE_quitResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_quitResponse.prototype.serialize = COVISE_quitResponse_serialize;

function COVISE_quitResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_quitResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}uploadFile
//
function COVISE_uploadFile () {
    this.typeMarker = 'COVISE_uploadFile';
    this._path = '';
    this._fileName = '';
    this._resource = null;
    this._chunkIndex = 0;
    this._chunkNr = 0;
    this._chunkSize = 0;
    this._fileSize = 0;
    this._fileTruncated = '';
}

//
// accessor is COVISE_uploadFile.prototype.getPath
// element get for path
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for path
// setter function is is COVISE_uploadFile.prototype.setPath
//
function COVISE_uploadFile_getPath() { return this._path;}

COVISE_uploadFile.prototype.getPath = COVISE_uploadFile_getPath;

function COVISE_uploadFile_setPath(value) { this._path = value;}

COVISE_uploadFile.prototype.setPath = COVISE_uploadFile_setPath;
//
// accessor is COVISE_uploadFile.prototype.getFileName
// element get for fileName
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for fileName
// setter function is is COVISE_uploadFile.prototype.setFileName
//
function COVISE_uploadFile_getFileName() { return this._fileName;}

COVISE_uploadFile.prototype.getFileName = COVISE_uploadFile_getFileName;

function COVISE_uploadFile_setFileName(value) { this._fileName = value;}

COVISE_uploadFile.prototype.setFileName = COVISE_uploadFile_setFileName;
//
// accessor is COVISE_uploadFile.prototype.getResource
// element get for resource
// - element type is {http://www.w3.org/2001/XMLSchema}base64Binary
// - required element
// - nillable
//
// element set for resource
// setter function is is COVISE_uploadFile.prototype.setResource
//
function COVISE_uploadFile_getResource() { return this._resource;}

COVISE_uploadFile.prototype.getResource = COVISE_uploadFile_getResource;

function COVISE_uploadFile_setResource(value) { this._resource = value;}

COVISE_uploadFile.prototype.setResource = COVISE_uploadFile_setResource;
//
// accessor is COVISE_uploadFile.prototype.getChunkIndex
// element get for chunkIndex
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for chunkIndex
// setter function is is COVISE_uploadFile.prototype.setChunkIndex
//
function COVISE_uploadFile_getChunkIndex() { return this._chunkIndex;}

COVISE_uploadFile.prototype.getChunkIndex = COVISE_uploadFile_getChunkIndex;

function COVISE_uploadFile_setChunkIndex(value) { this._chunkIndex = value;}

COVISE_uploadFile.prototype.setChunkIndex = COVISE_uploadFile_setChunkIndex;
//
// accessor is COVISE_uploadFile.prototype.getChunkNr
// element get for chunkNr
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for chunkNr
// setter function is is COVISE_uploadFile.prototype.setChunkNr
//
function COVISE_uploadFile_getChunkNr() { return this._chunkNr;}

COVISE_uploadFile.prototype.getChunkNr = COVISE_uploadFile_getChunkNr;

function COVISE_uploadFile_setChunkNr(value) { this._chunkNr = value;}

COVISE_uploadFile.prototype.setChunkNr = COVISE_uploadFile_setChunkNr;
//
// accessor is COVISE_uploadFile.prototype.getChunkSize
// element get for chunkSize
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for chunkSize
// setter function is is COVISE_uploadFile.prototype.setChunkSize
//
function COVISE_uploadFile_getChunkSize() { return this._chunkSize;}

COVISE_uploadFile.prototype.getChunkSize = COVISE_uploadFile_getChunkSize;

function COVISE_uploadFile_setChunkSize(value) { this._chunkSize = value;}

COVISE_uploadFile.prototype.setChunkSize = COVISE_uploadFile_setChunkSize;
//
// accessor is COVISE_uploadFile.prototype.getFileSize
// element get for fileSize
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for fileSize
// setter function is is COVISE_uploadFile.prototype.setFileSize
//
function COVISE_uploadFile_getFileSize() { return this._fileSize;}

COVISE_uploadFile.prototype.getFileSize = COVISE_uploadFile_getFileSize;

function COVISE_uploadFile_setFileSize(value) { this._fileSize = value;}

COVISE_uploadFile.prototype.setFileSize = COVISE_uploadFile_setFileSize;
//
// accessor is COVISE_uploadFile.prototype.getFileTruncated
// element get for fileTruncated
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for fileTruncated
// setter function is is COVISE_uploadFile.prototype.setFileTruncated
//
function COVISE_uploadFile_getFileTruncated() { return this._fileTruncated;}

COVISE_uploadFile.prototype.getFileTruncated = COVISE_uploadFile_getFileTruncated;

function COVISE_uploadFile_setFileTruncated(value) { this._fileTruncated = value;}

COVISE_uploadFile.prototype.setFileTruncated = COVISE_uploadFile_setFileTruncated;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}uploadFile
//
function COVISE_uploadFile_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<path>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._path);
     xml = xml + '</path>';
    }
    // block for local variables
    {
     xml = xml + '<fileName>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileName);
     xml = xml + '</fileName>';
    }
    // block for local variables
    {
     if (this._resource == null) {
      xml = xml + '<resource xsi:nil=\'true\'/>';
     } else {
      xml = xml + '<resource>';
      xml = xml + cxfjsutils.escapeXmlEntities(this._resource);
      xml = xml + '</resource>';
     }
    }
    // block for local variables
    {
     xml = xml + '<chunkIndex>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._chunkIndex);
     xml = xml + '</chunkIndex>';
    }
    // block for local variables
    {
     xml = xml + '<chunkNr>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._chunkNr);
     xml = xml + '</chunkNr>';
    }
    // block for local variables
    {
     xml = xml + '<chunkSize>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._chunkSize);
     xml = xml + '</chunkSize>';
    }
    // block for local variables
    {
     xml = xml + '<fileSize>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileSize);
     xml = xml + '</fileSize>';
    }
    // block for local variables
    {
     xml = xml + '<fileTruncated>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileTruncated);
     xml = xml + '</fileTruncated>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_uploadFile.prototype.serialize = COVISE_uploadFile_serialize;

function COVISE_uploadFile_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_uploadFile();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing path');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setPath(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileName');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFileName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing resource');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = cxfjsutils.deserializeBase64orMom(curElement);
    }
    newobject.setResource(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing chunkIndex');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setChunkIndex(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing chunkNr');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setChunkNr(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing chunkSize');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setChunkSize(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileSize');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setFileSize(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileTruncated');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setFileTruncated(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}uploadFileResponse
//
function COVISE_uploadFileResponse () {
    this.typeMarker = 'COVISE_uploadFileResponse';
    this._result = '';
    this._lastChunk = '';
}

//
// accessor is COVISE_uploadFileResponse.prototype.getResult
// element get for result
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for result
// setter function is is COVISE_uploadFileResponse.prototype.setResult
//
function COVISE_uploadFileResponse_getResult() { return this._result;}

COVISE_uploadFileResponse.prototype.getResult = COVISE_uploadFileResponse_getResult;

function COVISE_uploadFileResponse_setResult(value) { this._result = value;}

COVISE_uploadFileResponse.prototype.setResult = COVISE_uploadFileResponse_setResult;
//
// accessor is COVISE_uploadFileResponse.prototype.getLastChunk
// element get for lastChunk
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for lastChunk
// setter function is is COVISE_uploadFileResponse.prototype.setLastChunk
//
function COVISE_uploadFileResponse_getLastChunk() { return this._lastChunk;}

COVISE_uploadFileResponse.prototype.getLastChunk = COVISE_uploadFileResponse_getLastChunk;

function COVISE_uploadFileResponse_setLastChunk(value) { this._lastChunk = value;}

COVISE_uploadFileResponse.prototype.setLastChunk = COVISE_uploadFileResponse_setLastChunk;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}uploadFileResponse
//
function COVISE_uploadFileResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<result>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._result);
     xml = xml + '</result>';
    }
    // block for local variables
    {
     xml = xml + '<lastChunk>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._lastChunk);
     xml = xml + '</lastChunk>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_uploadFileResponse.prototype.serialize = COVISE_uploadFileResponse_serialize;

function COVISE_uploadFileResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_uploadFileResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing result');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setResult(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing lastChunk');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setLastChunk(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getLinks
//
function COVISE_getLinks () {
    this.typeMarker = 'COVISE_getLinks';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}getLinks
//
function COVISE_getLinks_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getLinks.prototype.serialize = COVISE_getLinks_serialize;

function COVISE_getLinks_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getLinks();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}createNewDirResponse
//
function COVISE_createNewDirResponse () {
    this.typeMarker = 'COVISE_createNewDirResponse';
    this._result = '';
}

//
// accessor is COVISE_createNewDirResponse.prototype.getResult
// element get for result
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for result
// setter function is is COVISE_createNewDirResponse.prototype.setResult
//
function COVISE_createNewDirResponse_getResult() { return this._result;}

COVISE_createNewDirResponse.prototype.getResult = COVISE_createNewDirResponse_getResult;

function COVISE_createNewDirResponse_setResult(value) { this._result = value;}

COVISE_createNewDirResponse.prototype.setResult = COVISE_createNewDirResponse_setResult;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}createNewDirResponse
//
function COVISE_createNewDirResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<result>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._result);
     xml = xml + '</result>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_createNewDirResponse.prototype.serialize = COVISE_createNewDirResponse_serialize;

function COVISE_createNewDirResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_createNewDirResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing result');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setResult(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}listModules
//
function COVISE_listModules () {
    this.typeMarker = 'COVISE_listModules';
    this._ipaddr = '';
}

//
// accessor is COVISE_listModules.prototype.getIpaddr
// element get for ipaddr
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for ipaddr
// setter function is is COVISE_listModules.prototype.setIpaddr
//
function COVISE_listModules_getIpaddr() { return this._ipaddr;}

COVISE_listModules.prototype.getIpaddr = COVISE_listModules_getIpaddr;

function COVISE_listModules_setIpaddr(value) { this._ipaddr = value;}

COVISE_listModules.prototype.setIpaddr = COVISE_listModules_setIpaddr;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}listModules
//
function COVISE_listModules_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<ipaddr>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._ipaddr);
     xml = xml + '</ipaddr>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_listModules.prototype.serialize = COVISE_listModules_serialize;

function COVISE_listModules_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_listModules();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing ipaddr');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setIpaddr(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ModuleDiedEvent
//
function COVISE_ModuleDiedEvent () {
    this.typeMarker = 'COVISE_ModuleDiedEvent';
    this._type = '';
    this._moduleID = '';
}

//
// accessor is COVISE_ModuleDiedEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ModuleDiedEvent.prototype.setType
//
function COVISE_ModuleDiedEvent_getType() { return this._type;}

COVISE_ModuleDiedEvent.prototype.getType = COVISE_ModuleDiedEvent_getType;

function COVISE_ModuleDiedEvent_setType(value) { this._type = value;}

COVISE_ModuleDiedEvent.prototype.setType = COVISE_ModuleDiedEvent_setType;
//
// accessor is COVISE_ModuleDiedEvent.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_ModuleDiedEvent.prototype.setModuleID
//
function COVISE_ModuleDiedEvent_getModuleID() { return this._moduleID;}

COVISE_ModuleDiedEvent.prototype.getModuleID = COVISE_ModuleDiedEvent_getModuleID;

function COVISE_ModuleDiedEvent_setModuleID(value) { this._moduleID = value;}

COVISE_ModuleDiedEvent.prototype.setModuleID = COVISE_ModuleDiedEvent_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ModuleDiedEvent
//
function COVISE_ModuleDiedEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ModuleDiedEvent.prototype.serialize = COVISE_ModuleDiedEvent_serialize;

function COVISE_ModuleDiedEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ModuleDiedEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}deleteDir
//
function COVISE_deleteDir () {
    this.typeMarker = 'COVISE_deleteDir';
    this._path = '';
}

//
// accessor is COVISE_deleteDir.prototype.getPath
// element get for path
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for path
// setter function is is COVISE_deleteDir.prototype.setPath
//
function COVISE_deleteDir_getPath() { return this._path;}

COVISE_deleteDir.prototype.getPath = COVISE_deleteDir_getPath;

function COVISE_deleteDir_setPath(value) { this._path = value;}

COVISE_deleteDir.prototype.setPath = COVISE_deleteDir_setPath;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}deleteDir
//
function COVISE_deleteDir_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<path>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._path);
     xml = xml + '</path>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_deleteDir.prototype.serialize = COVISE_deleteDir_serialize;

function COVISE_deleteDir_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_deleteDir();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing path');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setPath(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}addPartnerResponse
//
function COVISE_addPartnerResponse () {
    this.typeMarker = 'COVISE_addPartnerResponse';
    this._success = null;
}

//
// accessor is COVISE_addPartnerResponse.prototype.getSuccess
// element get for success
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - optional element
// - nillable
//
// element set for success
// setter function is is COVISE_addPartnerResponse.prototype.setSuccess
//
function COVISE_addPartnerResponse_getSuccess() { return this._success;}

COVISE_addPartnerResponse.prototype.getSuccess = COVISE_addPartnerResponse_getSuccess;

function COVISE_addPartnerResponse_setSuccess(value) { this._success = value;}

COVISE_addPartnerResponse.prototype.setSuccess = COVISE_addPartnerResponse_setSuccess;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}addPartnerResponse
//
function COVISE_addPartnerResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     if (this._success != null) {
      if (this._success == null) {
       xml = xml + '<success xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<success>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._success);
       xml = xml + '</success>';
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_addPartnerResponse.prototype.serialize = COVISE_addPartnerResponse_serialize;

function COVISE_addPartnerResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_addPartnerResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing success');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'success')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = (value == 'true');
     }
     newobject.setSuccess(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}setParameter
//
function COVISE_setParameter () {
    this.typeMarker = 'COVISE_setParameter';
    this._moduleID = '';
    this._parameter = null;
}

//
// accessor is COVISE_setParameter.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_setParameter.prototype.setModuleID
//
function COVISE_setParameter_getModuleID() { return this._moduleID;}

COVISE_setParameter.prototype.getModuleID = COVISE_setParameter_getModuleID;

function COVISE_setParameter_setModuleID(value) { this._moduleID = value;}

COVISE_setParameter.prototype.setModuleID = COVISE_setParameter_setModuleID;
//
// accessor is COVISE_setParameter.prototype.getParameter
// element get for parameter
// - element type is {http://www.hlrs.de/organization/vis/covise}Parameter
// - required element
// - nillable
//
// element set for parameter
// setter function is is COVISE_setParameter.prototype.setParameter
//
function COVISE_setParameter_getParameter() { return this._parameter;}

COVISE_setParameter.prototype.getParameter = COVISE_setParameter_getParameter;

function COVISE_setParameter_setParameter(value) { this._parameter = value;}

COVISE_setParameter.prototype.setParameter = COVISE_setParameter_setParameter;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}setParameter
//
function COVISE_setParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    // block for local variables
    {
     if (this._parameter == null) {
      xml = xml + '<parameter xsi:nil=\'true\'/>';
     } else {
      xml = xml + this._parameter.serialize(cxfjsutils, 'parameter', null);
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_setParameter.prototype.serialize = COVISE_setParameter_serialize;

function COVISE_setParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_setParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing parameter');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Parameter_deserialize(cxfjsutils, curElement);
    }
    newobject.setParameter(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}FloatVectorParameter
//
function COVISE_FloatVectorParameter () {
    this.typeMarker = 'COVISE_FloatVectorParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = [];
}

//
// accessor is COVISE_FloatVectorParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_FloatVectorParameter.prototype.setName
//
function COVISE_FloatVectorParameter_getName() { return this._name;}

COVISE_FloatVectorParameter.prototype.getName = COVISE_FloatVectorParameter_getName;

function COVISE_FloatVectorParameter_setName(value) { this._name = value;}

COVISE_FloatVectorParameter.prototype.setName = COVISE_FloatVectorParameter_setName;
//
// accessor is COVISE_FloatVectorParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_FloatVectorParameter.prototype.setType
//
function COVISE_FloatVectorParameter_getType() { return this._type;}

COVISE_FloatVectorParameter.prototype.getType = COVISE_FloatVectorParameter_getType;

function COVISE_FloatVectorParameter_setType(value) { this._type = value;}

COVISE_FloatVectorParameter.prototype.setType = COVISE_FloatVectorParameter_setType;
//
// accessor is COVISE_FloatVectorParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_FloatVectorParameter.prototype.setDescription
//
function COVISE_FloatVectorParameter_getDescription() { return this._description;}

COVISE_FloatVectorParameter.prototype.getDescription = COVISE_FloatVectorParameter_getDescription;

function COVISE_FloatVectorParameter_setDescription(value) { this._description = value;}

COVISE_FloatVectorParameter.prototype.setDescription = COVISE_FloatVectorParameter_setDescription;
//
// accessor is COVISE_FloatVectorParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_FloatVectorParameter.prototype.setMapped
//
function COVISE_FloatVectorParameter_getMapped() { return this._mapped;}

COVISE_FloatVectorParameter.prototype.getMapped = COVISE_FloatVectorParameter_getMapped;

function COVISE_FloatVectorParameter_setMapped(value) { this._mapped = value;}

COVISE_FloatVectorParameter.prototype.setMapped = COVISE_FloatVectorParameter_setMapped;
//
// accessor is COVISE_FloatVectorParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
// - array
//
// element set for value
// setter function is is COVISE_FloatVectorParameter.prototype.setValue
//
function COVISE_FloatVectorParameter_getValue() { return this._value;}

COVISE_FloatVectorParameter.prototype.getValue = COVISE_FloatVectorParameter_getValue;

function COVISE_FloatVectorParameter_setValue(value) { this._value = value;}

COVISE_FloatVectorParameter.prototype.setValue = COVISE_FloatVectorParameter_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}FloatVectorParameter
//
function COVISE_FloatVectorParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     if (this._value != null) {
      for (var ax = 0;ax < this._value.length;ax ++) {
       if (this._value[ax] == null) {
        xml = xml + '<value/>';
       } else {
        xml = xml + '<value>';
        xml = xml + cxfjsutils.escapeXmlEntities(this._value[ax]);
        xml = xml + '</value>';
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_FloatVectorParameter.prototype.serialize = COVISE_FloatVectorParameter_serialize;

function COVISE_FloatVectorParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_FloatVectorParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'value')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       value = cxfjsutils.getNodeText(curElement);
       arrayItem = parseFloat(value);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'value'));
     newobject.setValue(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}IntSliderParameter
//
function COVISE_IntSliderParameter () {
    this.typeMarker = 'COVISE_IntSliderParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = 0;
    this._min = 0;
    this._max = 0;
}

//
// accessor is COVISE_IntSliderParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_IntSliderParameter.prototype.setName
//
function COVISE_IntSliderParameter_getName() { return this._name;}

COVISE_IntSliderParameter.prototype.getName = COVISE_IntSliderParameter_getName;

function COVISE_IntSliderParameter_setName(value) { this._name = value;}

COVISE_IntSliderParameter.prototype.setName = COVISE_IntSliderParameter_setName;
//
// accessor is COVISE_IntSliderParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_IntSliderParameter.prototype.setType
//
function COVISE_IntSliderParameter_getType() { return this._type;}

COVISE_IntSliderParameter.prototype.getType = COVISE_IntSliderParameter_getType;

function COVISE_IntSliderParameter_setType(value) { this._type = value;}

COVISE_IntSliderParameter.prototype.setType = COVISE_IntSliderParameter_setType;
//
// accessor is COVISE_IntSliderParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_IntSliderParameter.prototype.setDescription
//
function COVISE_IntSliderParameter_getDescription() { return this._description;}

COVISE_IntSliderParameter.prototype.getDescription = COVISE_IntSliderParameter_getDescription;

function COVISE_IntSliderParameter_setDescription(value) { this._description = value;}

COVISE_IntSliderParameter.prototype.setDescription = COVISE_IntSliderParameter_setDescription;
//
// accessor is COVISE_IntSliderParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_IntSliderParameter.prototype.setMapped
//
function COVISE_IntSliderParameter_getMapped() { return this._mapped;}

COVISE_IntSliderParameter.prototype.getMapped = COVISE_IntSliderParameter_getMapped;

function COVISE_IntSliderParameter_setMapped(value) { this._mapped = value;}

COVISE_IntSliderParameter.prototype.setMapped = COVISE_IntSliderParameter_setMapped;
//
// accessor is COVISE_IntSliderParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for value
// setter function is is COVISE_IntSliderParameter.prototype.setValue
//
function COVISE_IntSliderParameter_getValue() { return this._value;}

COVISE_IntSliderParameter.prototype.getValue = COVISE_IntSliderParameter_getValue;

function COVISE_IntSliderParameter_setValue(value) { this._value = value;}

COVISE_IntSliderParameter.prototype.setValue = COVISE_IntSliderParameter_setValue;
//
// accessor is COVISE_IntSliderParameter.prototype.getMin
// element get for min
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for min
// setter function is is COVISE_IntSliderParameter.prototype.setMin
//
function COVISE_IntSliderParameter_getMin() { return this._min;}

COVISE_IntSliderParameter.prototype.getMin = COVISE_IntSliderParameter_getMin;

function COVISE_IntSliderParameter_setMin(value) { this._min = value;}

COVISE_IntSliderParameter.prototype.setMin = COVISE_IntSliderParameter_setMin;
//
// accessor is COVISE_IntSliderParameter.prototype.getMax
// element get for max
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for max
// setter function is is COVISE_IntSliderParameter.prototype.setMax
//
function COVISE_IntSliderParameter_getMax() { return this._max;}

COVISE_IntSliderParameter.prototype.getMax = COVISE_IntSliderParameter_getMax;

function COVISE_IntSliderParameter_setMax(value) { this._max = value;}

COVISE_IntSliderParameter.prototype.setMax = COVISE_IntSliderParameter_setMax;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}IntSliderParameter
//
function COVISE_IntSliderParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    // block for local variables
    {
     xml = xml + '<min>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._min);
     xml = xml + '</min>';
    }
    // block for local variables
    {
     xml = xml + '<max>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._max);
     xml = xml + '</max>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_IntSliderParameter.prototype.serialize = COVISE_IntSliderParameter_serialize;

function COVISE_IntSliderParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_IntSliderParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing min');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setMin(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing max');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setMax(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}addEventListenerResponse
//
function COVISE_addEventListenerResponse () {
    this.typeMarker = 'COVISE_addEventListenerResponse';
    this._uuid = '';
}

//
// accessor is COVISE_addEventListenerResponse.prototype.getUuid
// element get for uuid
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for uuid
// setter function is is COVISE_addEventListenerResponse.prototype.setUuid
//
function COVISE_addEventListenerResponse_getUuid() { return this._uuid;}

COVISE_addEventListenerResponse.prototype.getUuid = COVISE_addEventListenerResponse_getUuid;

function COVISE_addEventListenerResponse_setUuid(value) { this._uuid = value;}

COVISE_addEventListenerResponse.prototype.setUuid = COVISE_addEventListenerResponse_setUuid;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}addEventListenerResponse
//
function COVISE_addEventListenerResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<uuid>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._uuid);
     xml = xml + '</uuid>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_addEventListenerResponse.prototype.serialize = COVISE_addEventListenerResponse_serialize;

function COVISE_addEventListenerResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_addEventListenerResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing uuid');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setUuid(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getEvent
//
function COVISE_getEvent () {
    this.typeMarker = 'COVISE_getEvent';
    this._uuid = '';
    this._timeout = null;
}

//
// accessor is COVISE_getEvent.prototype.getUuid
// element get for uuid
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for uuid
// setter function is is COVISE_getEvent.prototype.setUuid
//
function COVISE_getEvent_getUuid() { return this._uuid;}

COVISE_getEvent.prototype.getUuid = COVISE_getEvent_getUuid;

function COVISE_getEvent_setUuid(value) { this._uuid = value;}

COVISE_getEvent.prototype.setUuid = COVISE_getEvent_setUuid;
//
// accessor is COVISE_getEvent.prototype.getTimeout
// element get for timeout
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - optional element
// - nillable
//
// element set for timeout
// setter function is is COVISE_getEvent.prototype.setTimeout
//
function COVISE_getEvent_getTimeout() { return this._timeout;}

COVISE_getEvent.prototype.getTimeout = COVISE_getEvent_getTimeout;

function COVISE_getEvent_setTimeout(value) { this._timeout = value;}

COVISE_getEvent.prototype.setTimeout = COVISE_getEvent_setTimeout;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getEvent
//
function COVISE_getEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<uuid>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._uuid);
     xml = xml + '</uuid>';
    }
    // block for local variables
    {
     if (this._timeout != null) {
      if (this._timeout == null) {
       xml = xml + '<timeout xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<timeout>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._timeout);
       xml = xml + '</timeout>';
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getEvent.prototype.serialize = COVISE_getEvent_serialize;

function COVISE_getEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing uuid');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setUuid(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing timeout');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'timeout')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = parseInt(value);
     }
     newobject.setTimeout(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}instantiateModule
//
function COVISE_instantiateModule () {
    this.typeMarker = 'COVISE_instantiateModule';
    this._module = '';
    this._host = '';
    this._x = null;
    this._y = null;
}

//
// accessor is COVISE_instantiateModule.prototype.getModule
// element get for module
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for module
// setter function is is COVISE_instantiateModule.prototype.setModule
//
function COVISE_instantiateModule_getModule() { return this._module;}

COVISE_instantiateModule.prototype.getModule = COVISE_instantiateModule_getModule;

function COVISE_instantiateModule_setModule(value) { this._module = value;}

COVISE_instantiateModule.prototype.setModule = COVISE_instantiateModule_setModule;
//
// accessor is COVISE_instantiateModule.prototype.getHost
// element get for host
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for host
// setter function is is COVISE_instantiateModule.prototype.setHost
//
function COVISE_instantiateModule_getHost() { return this._host;}

COVISE_instantiateModule.prototype.getHost = COVISE_instantiateModule_getHost;

function COVISE_instantiateModule_setHost(value) { this._host = value;}

COVISE_instantiateModule.prototype.setHost = COVISE_instantiateModule_setHost;
//
// accessor is COVISE_instantiateModule.prototype.getX
// element get for x
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - optional element
// - nillable
//
// element set for x
// setter function is is COVISE_instantiateModule.prototype.setX
//
function COVISE_instantiateModule_getX() { return this._x;}

COVISE_instantiateModule.prototype.getX = COVISE_instantiateModule_getX;

function COVISE_instantiateModule_setX(value) { this._x = value;}

COVISE_instantiateModule.prototype.setX = COVISE_instantiateModule_setX;
//
// accessor is COVISE_instantiateModule.prototype.getY
// element get for y
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - optional element
// - nillable
//
// element set for y
// setter function is is COVISE_instantiateModule.prototype.setY
//
function COVISE_instantiateModule_getY() { return this._y;}

COVISE_instantiateModule.prototype.getY = COVISE_instantiateModule_getY;

function COVISE_instantiateModule_setY(value) { this._y = value;}

COVISE_instantiateModule.prototype.setY = COVISE_instantiateModule_setY;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}instantiateModule
//
function COVISE_instantiateModule_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<module>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._module);
     xml = xml + '</module>';
    }
    // block for local variables
    {
     xml = xml + '<host>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._host);
     xml = xml + '</host>';
    }
    // block for local variables
    {
     if (this._x != null) {
      if (this._x == null) {
       xml = xml + '<x xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<x>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._x);
       xml = xml + '</x>';
      }
     }
    }
    // block for local variables
    {
     if (this._y != null) {
      if (this._y == null) {
       xml = xml + '<y xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<y>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._y);
       xml = xml + '</y>';
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_instantiateModule.prototype.serialize = COVISE_instantiateModule_serialize;

function COVISE_instantiateModule_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_instantiateModule();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing module');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModule(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing host');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setHost(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing x');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'x')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = parseInt(value);
     }
     newobject.setX(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing y');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'y')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = parseInt(value);
     }
     newobject.setY(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}listHosts
//
function COVISE_listHosts () {
    this.typeMarker = 'COVISE_listHosts';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}listHosts
//
function COVISE_listHosts_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_listHosts.prototype.serialize = COVISE_listHosts_serialize;

function COVISE_listHosts_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_listHosts();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getFileInfoListResponse
//
function COVISE_getFileInfoListResponse () {
    this.typeMarker = 'COVISE_getFileInfoListResponse';
    this._fileInfoList = [];
}

//
// accessor is COVISE_getFileInfoListResponse.prototype.getFileInfoList
// element get for fileInfoList
// - element type is {http://www.hlrs.de/organization/vis/covise}FileInfo
// - required element
// - array
//
// element set for fileInfoList
// setter function is is COVISE_getFileInfoListResponse.prototype.setFileInfoList
//
function COVISE_getFileInfoListResponse_getFileInfoList() { return this._fileInfoList;}

COVISE_getFileInfoListResponse.prototype.getFileInfoList = COVISE_getFileInfoListResponse_getFileInfoList;

function COVISE_getFileInfoListResponse_setFileInfoList(value) { this._fileInfoList = value;}

COVISE_getFileInfoListResponse.prototype.setFileInfoList = COVISE_getFileInfoListResponse_setFileInfoList;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getFileInfoListResponse
//
function COVISE_getFileInfoListResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     if (this._fileInfoList != null) {
      for (var ax = 0;ax < this._fileInfoList.length;ax ++) {
       if (this._fileInfoList[ax] == null) {
        xml = xml + '<fileInfoList/>';
       } else {
        xml = xml + this._fileInfoList[ax].serialize(cxfjsutils, 'fileInfoList', null);
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getFileInfoListResponse.prototype.serialize = COVISE_getFileInfoListResponse_serialize;

function COVISE_getFileInfoListResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getFileInfoListResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileInfoList');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'fileInfoList')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_FileInfo_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'fileInfoList'));
     newobject.setFileInfoList(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}LinkDelEvent
//
function COVISE_LinkDelEvent () {
    this.typeMarker = 'COVISE_LinkDelEvent';
    this._type = '';
    this._linkID = '';
}

//
// accessor is COVISE_LinkDelEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_LinkDelEvent.prototype.setType
//
function COVISE_LinkDelEvent_getType() { return this._type;}

COVISE_LinkDelEvent.prototype.getType = COVISE_LinkDelEvent_getType;

function COVISE_LinkDelEvent_setType(value) { this._type = value;}

COVISE_LinkDelEvent.prototype.setType = COVISE_LinkDelEvent_setType;
//
// accessor is COVISE_LinkDelEvent.prototype.getLinkID
// element get for linkID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for linkID
// setter function is is COVISE_LinkDelEvent.prototype.setLinkID
//
function COVISE_LinkDelEvent_getLinkID() { return this._linkID;}

COVISE_LinkDelEvent.prototype.getLinkID = COVISE_LinkDelEvent_getLinkID;

function COVISE_LinkDelEvent_setLinkID(value) { this._linkID = value;}

COVISE_LinkDelEvent.prototype.setLinkID = COVISE_LinkDelEvent_setLinkID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}LinkDelEvent
//
function COVISE_LinkDelEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<linkID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._linkID);
     xml = xml + '</linkID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_LinkDelEvent.prototype.serialize = COVISE_LinkDelEvent_serialize;

function COVISE_LinkDelEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_LinkDelEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing linkID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setLinkID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}Parameter
//
function COVISE_Parameter () {
    this.typeMarker = 'COVISE_Parameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
}

//
// accessor is COVISE_Parameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_Parameter.prototype.setName
//
function COVISE_Parameter_getName() { return this._name;}

COVISE_Parameter.prototype.getName = COVISE_Parameter_getName;

function COVISE_Parameter_setName(value) { this._name = value;}

COVISE_Parameter.prototype.setName = COVISE_Parameter_setName;
//
// accessor is COVISE_Parameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_Parameter.prototype.setType
//
function COVISE_Parameter_getType() { return this._type;}

COVISE_Parameter.prototype.getType = COVISE_Parameter_getType;

function COVISE_Parameter_setType(value) { this._type = value;}

COVISE_Parameter.prototype.setType = COVISE_Parameter_setType;
//
// accessor is COVISE_Parameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_Parameter.prototype.setDescription
//
function COVISE_Parameter_getDescription() { return this._description;}

COVISE_Parameter.prototype.getDescription = COVISE_Parameter_getDescription;

function COVISE_Parameter_setDescription(value) { this._description = value;}

COVISE_Parameter.prototype.setDescription = COVISE_Parameter_setDescription;
//
// accessor is COVISE_Parameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_Parameter.prototype.setMapped
//
function COVISE_Parameter_getMapped() { return this._mapped;}

COVISE_Parameter.prototype.getMapped = COVISE_Parameter_getMapped;

function COVISE_Parameter_setMapped(value) { this._mapped = value;}

COVISE_Parameter.prototype.setMapped = COVISE_Parameter_setMapped;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}Parameter
//
function COVISE_Parameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_Parameter.prototype.serialize = COVISE_Parameter_serialize;

function COVISE_Parameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_Parameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getRunningModuleResponse
//
function COVISE_getRunningModuleResponse () {
    this.typeMarker = 'COVISE_getRunningModuleResponse';
    this._module = null;
}

//
// accessor is COVISE_getRunningModuleResponse.prototype.getModule
// element get for module
// - element type is {http://www.hlrs.de/organization/vis/covise}Module
// - required element
//
// element set for module
// setter function is is COVISE_getRunningModuleResponse.prototype.setModule
//
function COVISE_getRunningModuleResponse_getModule() { return this._module;}

COVISE_getRunningModuleResponse.prototype.getModule = COVISE_getRunningModuleResponse_getModule;

function COVISE_getRunningModuleResponse_setModule(value) { this._module = value;}

COVISE_getRunningModuleResponse.prototype.setModule = COVISE_getRunningModuleResponse_setModule;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getRunningModuleResponse
//
function COVISE_getRunningModuleResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + this._module.serialize(cxfjsutils, 'module', null);
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getRunningModuleResponse.prototype.serialize = COVISE_getRunningModuleResponse_serialize;

function COVISE_getRunningModuleResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getRunningModuleResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing module');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Module_deserialize(cxfjsutils, curElement);
    }
    newobject.setModule(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Simple type (enumeration) {http://www.hlrs.de/organization/vis/covise}AddPartnerMethod
//
//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFileResponse
//
function COVISE_setParameterFromUploadedFileResponse () {
    this.typeMarker = 'COVISE_setParameterFromUploadedFileResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFileResponse
//
function COVISE_setParameterFromUploadedFileResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_setParameterFromUploadedFileResponse.prototype.serialize = COVISE_setParameterFromUploadedFileResponse_serialize;

function COVISE_setParameterFromUploadedFileResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_setParameterFromUploadedFileResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Simple type (enumeration) {http://www.hlrs.de/organization/vis/covise}covise-AddPartnerMethod
//
// - RExec
// - RSH
// - SSH
// - NQS
// - Manual
// - RemoteDaemon
//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}Colormap
//
function COVISE_Colormap () {
    this.typeMarker = 'COVISE_Colormap';
    this._name = '';
    this._pins = [];
}

//
// accessor is COVISE_Colormap.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_Colormap.prototype.setName
//
function COVISE_Colormap_getName() { return this._name;}

COVISE_Colormap.prototype.getName = COVISE_Colormap_getName;

function COVISE_Colormap_setName(value) { this._name = value;}

COVISE_Colormap.prototype.setName = COVISE_Colormap_setName;
//
// accessor is COVISE_Colormap.prototype.getPins
// element get for pins
// - element type is {http://www.hlrs.de/organization/vis/covise}ColormapPin
// - required element
// - array
//
// element set for pins
// setter function is is COVISE_Colormap.prototype.setPins
//
function COVISE_Colormap_getPins() { return this._pins;}

COVISE_Colormap.prototype.getPins = COVISE_Colormap_getPins;

function COVISE_Colormap_setPins(value) { this._pins = value;}

COVISE_Colormap.prototype.setPins = COVISE_Colormap_setPins;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}Colormap
//
function COVISE_Colormap_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     if (this._pins != null) {
      for (var ax = 0;ax < this._pins.length;ax ++) {
       if (this._pins[ax] == null) {
        xml = xml + '<pins/>';
       } else {
        xml = xml + this._pins[ax].serialize(cxfjsutils, 'pins', null);
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_Colormap.prototype.serialize = COVISE_Colormap_serialize;

function COVISE_Colormap_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_Colormap();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing pins');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'pins')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_ColormapPin_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'pins'));
     newobject.setPins(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}FileBrowserParameter
//
function COVISE_FileBrowserParameter () {
    this.typeMarker = 'COVISE_FileBrowserParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = '';
}

//
// accessor is COVISE_FileBrowserParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_FileBrowserParameter.prototype.setName
//
function COVISE_FileBrowserParameter_getName() { return this._name;}

COVISE_FileBrowserParameter.prototype.getName = COVISE_FileBrowserParameter_getName;

function COVISE_FileBrowserParameter_setName(value) { this._name = value;}

COVISE_FileBrowserParameter.prototype.setName = COVISE_FileBrowserParameter_setName;
//
// accessor is COVISE_FileBrowserParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_FileBrowserParameter.prototype.setType
//
function COVISE_FileBrowserParameter_getType() { return this._type;}

COVISE_FileBrowserParameter.prototype.getType = COVISE_FileBrowserParameter_getType;

function COVISE_FileBrowserParameter_setType(value) { this._type = value;}

COVISE_FileBrowserParameter.prototype.setType = COVISE_FileBrowserParameter_setType;
//
// accessor is COVISE_FileBrowserParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_FileBrowserParameter.prototype.setDescription
//
function COVISE_FileBrowserParameter_getDescription() { return this._description;}

COVISE_FileBrowserParameter.prototype.getDescription = COVISE_FileBrowserParameter_getDescription;

function COVISE_FileBrowserParameter_setDescription(value) { this._description = value;}

COVISE_FileBrowserParameter.prototype.setDescription = COVISE_FileBrowserParameter_setDescription;
//
// accessor is COVISE_FileBrowserParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_FileBrowserParameter.prototype.setMapped
//
function COVISE_FileBrowserParameter_getMapped() { return this._mapped;}

COVISE_FileBrowserParameter.prototype.getMapped = COVISE_FileBrowserParameter_getMapped;

function COVISE_FileBrowserParameter_setMapped(value) { this._mapped = value;}

COVISE_FileBrowserParameter.prototype.setMapped = COVISE_FileBrowserParameter_setMapped;
//
// accessor is COVISE_FileBrowserParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for value
// setter function is is COVISE_FileBrowserParameter.prototype.setValue
//
function COVISE_FileBrowserParameter_getValue() { return this._value;}

COVISE_FileBrowserParameter.prototype.getValue = COVISE_FileBrowserParameter_getValue;

function COVISE_FileBrowserParameter_setValue(value) { this._value = value;}

COVISE_FileBrowserParameter.prototype.setValue = COVISE_FileBrowserParameter_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}FileBrowserParameter
//
function COVISE_FileBrowserParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_FileBrowserParameter.prototype.serialize = COVISE_FileBrowserParameter_serialize;

function COVISE_FileBrowserParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_FileBrowserParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}addPartner
//
function COVISE_addPartner () {
    this.typeMarker = 'COVISE_addPartner';
    this._method = null;
    this._ip = null;
    this._user = null;
    this._password = null;
    this._timeout = null;
    this._display = null;
}

//
// accessor is COVISE_addPartner.prototype.getMethod
// element get for method
// - element type is {http://www.hlrs.de/organization/vis/covise}AddPartnerMethod
// - required element
// - nillable
//
// element set for method
// setter function is is COVISE_addPartner.prototype.setMethod
//
function COVISE_addPartner_getMethod() { return this._method;}

COVISE_addPartner.prototype.getMethod = COVISE_addPartner_getMethod;

function COVISE_addPartner_setMethod(value) { this._method = value;}

COVISE_addPartner.prototype.setMethod = COVISE_addPartner_setMethod;
//
// accessor is COVISE_addPartner.prototype.getIp
// element get for ip
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - optional element
// - nillable
//
// element set for ip
// setter function is is COVISE_addPartner.prototype.setIp
//
function COVISE_addPartner_getIp() { return this._ip;}

COVISE_addPartner.prototype.getIp = COVISE_addPartner_getIp;

function COVISE_addPartner_setIp(value) { this._ip = value;}

COVISE_addPartner.prototype.setIp = COVISE_addPartner_setIp;
//
// accessor is COVISE_addPartner.prototype.getUser
// element get for user
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - optional element
// - nillable
//
// element set for user
// setter function is is COVISE_addPartner.prototype.setUser
//
function COVISE_addPartner_getUser() { return this._user;}

COVISE_addPartner.prototype.getUser = COVISE_addPartner_getUser;

function COVISE_addPartner_setUser(value) { this._user = value;}

COVISE_addPartner.prototype.setUser = COVISE_addPartner_setUser;
//
// accessor is COVISE_addPartner.prototype.getPassword
// element get for password
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - optional element
// - nillable
//
// element set for password
// setter function is is COVISE_addPartner.prototype.setPassword
//
function COVISE_addPartner_getPassword() { return this._password;}

COVISE_addPartner.prototype.getPassword = COVISE_addPartner_getPassword;

function COVISE_addPartner_setPassword(value) { this._password = value;}

COVISE_addPartner.prototype.setPassword = COVISE_addPartner_setPassword;
//
// accessor is COVISE_addPartner.prototype.getTimeout
// element get for timeout
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - optional element
// - nillable
//
// element set for timeout
// setter function is is COVISE_addPartner.prototype.setTimeout
//
function COVISE_addPartner_getTimeout() { return this._timeout;}

COVISE_addPartner.prototype.getTimeout = COVISE_addPartner_getTimeout;

function COVISE_addPartner_setTimeout(value) { this._timeout = value;}

COVISE_addPartner.prototype.setTimeout = COVISE_addPartner_setTimeout;
//
// accessor is COVISE_addPartner.prototype.getDisplay
// element get for display
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - optional element
// - nillable
//
// element set for display
// setter function is is COVISE_addPartner.prototype.setDisplay
//
function COVISE_addPartner_getDisplay() { return this._display;}

COVISE_addPartner.prototype.getDisplay = COVISE_addPartner_getDisplay;

function COVISE_addPartner_setDisplay(value) { this._display = value;}

COVISE_addPartner.prototype.setDisplay = COVISE_addPartner_setDisplay;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}addPartner
//
function COVISE_addPartner_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     if (this._method == null) {
      xml = xml + '<method xsi:nil=\'true\'/>';
     } else {
      xml = xml + '<method>';
      xml = xml + cxfjsutils.escapeXmlEntities(this._method);
      xml = xml + '</method>';
     }
    }
    // block for local variables
    {
     if (this._ip != null) {
      if (this._ip == null) {
       xml = xml + '<ip xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<ip>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._ip);
       xml = xml + '</ip>';
      }
     }
    }
    // block for local variables
    {
     if (this._user != null) {
      if (this._user == null) {
       xml = xml + '<user xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<user>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._user);
       xml = xml + '</user>';
      }
     }
    }
    // block for local variables
    {
     if (this._password != null) {
      if (this._password == null) {
       xml = xml + '<password xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<password>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._password);
       xml = xml + '</password>';
      }
     }
    }
    // block for local variables
    {
     if (this._timeout != null) {
      if (this._timeout == null) {
       xml = xml + '<timeout xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<timeout>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._timeout);
       xml = xml + '</timeout>';
      }
     }
    }
    // block for local variables
    {
     if (this._display != null) {
      if (this._display == null) {
       xml = xml + '<display xsi:nil=\'true\'/>';
      } else {
       xml = xml + '<display>';
       xml = xml + cxfjsutils.escapeXmlEntities(this._display);
       xml = xml + '</display>';
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_addPartner.prototype.serialize = COVISE_addPartner_serialize;

function COVISE_addPartner_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_addPartner();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing method');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setMethod(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing ip');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'ip')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = value;
     }
     newobject.setIp(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing user');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'user')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = value;
     }
     newobject.setUser(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing password');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'password')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = value;
     }
     newobject.setPassword(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing timeout');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'timeout')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = parseInt(value);
     }
     newobject.setTimeout(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing display');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'display')) {
     var value = null;
     if (!cxfjsutils.isElementNil(curElement)) {
      value = cxfjsutils.getNodeText(curElement);
      item = value;
     }
     newobject.setDisplay(item);
     var item = null;
     if (curElement != null) {
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}executeModule
//
function COVISE_executeModule () {
    this.typeMarker = 'COVISE_executeModule';
    this._moduleID = '';
}

//
// accessor is COVISE_executeModule.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_executeModule.prototype.setModuleID
//
function COVISE_executeModule_getModuleID() { return this._moduleID;}

COVISE_executeModule.prototype.getModuleID = COVISE_executeModule_getModuleID;

function COVISE_executeModule_setModuleID(value) { this._moduleID = value;}

COVISE_executeModule.prototype.setModuleID = COVISE_executeModule_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}executeModule
//
function COVISE_executeModule_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_executeModule.prototype.serialize = COVISE_executeModule_serialize;

function COVISE_executeModule_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_executeModule();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}isDirExist
//
function COVISE_isDirExist () {
    this.typeMarker = 'COVISE_isDirExist';
    this._path = '';
    this._newDir = '';
}

//
// accessor is COVISE_isDirExist.prototype.getPath
// element get for path
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for path
// setter function is is COVISE_isDirExist.prototype.setPath
//
function COVISE_isDirExist_getPath() { return this._path;}

COVISE_isDirExist.prototype.getPath = COVISE_isDirExist_getPath;

function COVISE_isDirExist_setPath(value) { this._path = value;}

COVISE_isDirExist.prototype.setPath = COVISE_isDirExist_setPath;
//
// accessor is COVISE_isDirExist.prototype.getNewDir
// element get for newDir
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for newDir
// setter function is is COVISE_isDirExist.prototype.setNewDir
//
function COVISE_isDirExist_getNewDir() { return this._newDir;}

COVISE_isDirExist.prototype.getNewDir = COVISE_isDirExist_getNewDir;

function COVISE_isDirExist_setNewDir(value) { this._newDir = value;}

COVISE_isDirExist.prototype.setNewDir = COVISE_isDirExist_setNewDir;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}isDirExist
//
function COVISE_isDirExist_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<path>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._path);
     xml = xml + '</path>';
    }
    // block for local variables
    {
     xml = xml + '<newDir>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._newDir);
     xml = xml + '</newDir>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_isDirExist.prototype.serialize = COVISE_isDirExist_serialize;

function COVISE_isDirExist_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_isDirExist();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing path');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setPath(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing newDir');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setNewDir(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}Port
//
function COVISE_Port () {
    this.typeMarker = 'COVISE_Port';
    this._name = '';
    this._types = [];
    this._portType = '';
    this._id = '';
    this._moduleID = '';
}

//
// accessor is COVISE_Port.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_Port.prototype.setName
//
function COVISE_Port_getName() { return this._name;}

COVISE_Port.prototype.getName = COVISE_Port_getName;

function COVISE_Port_setName(value) { this._name = value;}

COVISE_Port.prototype.setName = COVISE_Port_setName;
//
// accessor is COVISE_Port.prototype.getTypes
// element get for types
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
// - array
//
// element set for types
// setter function is is COVISE_Port.prototype.setTypes
//
function COVISE_Port_getTypes() { return this._types;}

COVISE_Port.prototype.getTypes = COVISE_Port_getTypes;

function COVISE_Port_setTypes(value) { this._types = value;}

COVISE_Port.prototype.setTypes = COVISE_Port_setTypes;
//
// accessor is COVISE_Port.prototype.getPortType
// element get for portType
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for portType
// setter function is is COVISE_Port.prototype.setPortType
//
function COVISE_Port_getPortType() { return this._portType;}

COVISE_Port.prototype.getPortType = COVISE_Port_getPortType;

function COVISE_Port_setPortType(value) { this._portType = value;}

COVISE_Port.prototype.setPortType = COVISE_Port_setPortType;
//
// accessor is COVISE_Port.prototype.getId
// element get for id
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for id
// setter function is is COVISE_Port.prototype.setId
//
function COVISE_Port_getId() { return this._id;}

COVISE_Port.prototype.getId = COVISE_Port_getId;

function COVISE_Port_setId(value) { this._id = value;}

COVISE_Port.prototype.setId = COVISE_Port_setId;
//
// accessor is COVISE_Port.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_Port.prototype.setModuleID
//
function COVISE_Port_getModuleID() { return this._moduleID;}

COVISE_Port.prototype.getModuleID = COVISE_Port_getModuleID;

function COVISE_Port_setModuleID(value) { this._moduleID = value;}

COVISE_Port.prototype.setModuleID = COVISE_Port_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}Port
//
function COVISE_Port_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     if (this._types != null) {
      for (var ax = 0;ax < this._types.length;ax ++) {
       if (this._types[ax] == null) {
        xml = xml + '<types/>';
       } else {
        xml = xml + '<types>';
        xml = xml + cxfjsutils.escapeXmlEntities(this._types[ax]);
        xml = xml + '</types>';
       }
      }
     }
    }
    // block for local variables
    {
     xml = xml + '<portType>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._portType);
     xml = xml + '</portType>';
    }
    // block for local variables
    {
     xml = xml + '<id>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._id);
     xml = xml + '</id>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_Port.prototype.serialize = COVISE_Port_serialize;

function COVISE_Port_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_Port();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing types');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'types')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       value = cxfjsutils.getNodeText(curElement);
       arrayItem = value;
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'types'));
     newobject.setTypes(item);
     var item = null;
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing portType');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setPortType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing id');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setId(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}Module
//
function COVISE_Module () {
    this.typeMarker = 'COVISE_Module';
    this._name = '';
    this._category = '';
    this._host = '';
    this._description = '';
    this._instance = '';
    this._id = '';
    this._title = '';
    this._position = null;
    this._parameters = [];
    this._inputPorts = [];
    this._outputPorts = [];
}

//
// accessor is COVISE_Module.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_Module.prototype.setName
//
function COVISE_Module_getName() { return this._name;}

COVISE_Module.prototype.getName = COVISE_Module_getName;

function COVISE_Module_setName(value) { this._name = value;}

COVISE_Module.prototype.setName = COVISE_Module_setName;
//
// accessor is COVISE_Module.prototype.getCategory
// element get for category
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for category
// setter function is is COVISE_Module.prototype.setCategory
//
function COVISE_Module_getCategory() { return this._category;}

COVISE_Module.prototype.getCategory = COVISE_Module_getCategory;

function COVISE_Module_setCategory(value) { this._category = value;}

COVISE_Module.prototype.setCategory = COVISE_Module_setCategory;
//
// accessor is COVISE_Module.prototype.getHost
// element get for host
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for host
// setter function is is COVISE_Module.prototype.setHost
//
function COVISE_Module_getHost() { return this._host;}

COVISE_Module.prototype.getHost = COVISE_Module_getHost;

function COVISE_Module_setHost(value) { this._host = value;}

COVISE_Module.prototype.setHost = COVISE_Module_setHost;
//
// accessor is COVISE_Module.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_Module.prototype.setDescription
//
function COVISE_Module_getDescription() { return this._description;}

COVISE_Module.prototype.getDescription = COVISE_Module_getDescription;

function COVISE_Module_setDescription(value) { this._description = value;}

COVISE_Module.prototype.setDescription = COVISE_Module_setDescription;
//
// accessor is COVISE_Module.prototype.getInstance
// element get for instance
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for instance
// setter function is is COVISE_Module.prototype.setInstance
//
function COVISE_Module_getInstance() { return this._instance;}

COVISE_Module.prototype.getInstance = COVISE_Module_getInstance;

function COVISE_Module_setInstance(value) { this._instance = value;}

COVISE_Module.prototype.setInstance = COVISE_Module_setInstance;
//
// accessor is COVISE_Module.prototype.getId
// element get for id
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for id
// setter function is is COVISE_Module.prototype.setId
//
function COVISE_Module_getId() { return this._id;}

COVISE_Module.prototype.getId = COVISE_Module_getId;

function COVISE_Module_setId(value) { this._id = value;}

COVISE_Module.prototype.setId = COVISE_Module_setId;
//
// accessor is COVISE_Module.prototype.getTitle
// element get for title
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for title
// setter function is is COVISE_Module.prototype.setTitle
//
function COVISE_Module_getTitle() { return this._title;}

COVISE_Module.prototype.getTitle = COVISE_Module_getTitle;

function COVISE_Module_setTitle(value) { this._title = value;}

COVISE_Module.prototype.setTitle = COVISE_Module_setTitle;
//
// accessor is COVISE_Module.prototype.getPosition
// element get for position
// - element type is {http://www.hlrs.de/organization/vis/covise}Point
// - required element
//
// element set for position
// setter function is is COVISE_Module.prototype.setPosition
//
function COVISE_Module_getPosition() { return this._position;}

COVISE_Module.prototype.getPosition = COVISE_Module_getPosition;

function COVISE_Module_setPosition(value) { this._position = value;}

COVISE_Module.prototype.setPosition = COVISE_Module_setPosition;
//
// accessor is COVISE_Module.prototype.getParameters
// element get for parameters
// - element type is {http://www.hlrs.de/organization/vis/covise}Parameter
// - required element
// - array
// - nillable
//
// element set for parameters
// setter function is is COVISE_Module.prototype.setParameters
//
function COVISE_Module_getParameters() { return this._parameters;}

COVISE_Module.prototype.getParameters = COVISE_Module_getParameters;

function COVISE_Module_setParameters(value) { this._parameters = value;}

COVISE_Module.prototype.setParameters = COVISE_Module_setParameters;
//
// accessor is COVISE_Module.prototype.getInputPorts
// element get for inputPorts
// - element type is {http://www.hlrs.de/organization/vis/covise}Port
// - required element
// - array
//
// element set for inputPorts
// setter function is is COVISE_Module.prototype.setInputPorts
//
function COVISE_Module_getInputPorts() { return this._inputPorts;}

COVISE_Module.prototype.getInputPorts = COVISE_Module_getInputPorts;

function COVISE_Module_setInputPorts(value) { this._inputPorts = value;}

COVISE_Module.prototype.setInputPorts = COVISE_Module_setInputPorts;
//
// accessor is COVISE_Module.prototype.getOutputPorts
// element get for outputPorts
// - element type is {http://www.hlrs.de/organization/vis/covise}Port
// - required element
// - array
//
// element set for outputPorts
// setter function is is COVISE_Module.prototype.setOutputPorts
//
function COVISE_Module_getOutputPorts() { return this._outputPorts;}

COVISE_Module.prototype.getOutputPorts = COVISE_Module_getOutputPorts;

function COVISE_Module_setOutputPorts(value) { this._outputPorts = value;}

COVISE_Module.prototype.setOutputPorts = COVISE_Module_setOutputPorts;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}Module
//
function COVISE_Module_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<category>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._category);
     xml = xml + '</category>';
    }
    // block for local variables
    {
     xml = xml + '<host>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._host);
     xml = xml + '</host>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<instance>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._instance);
     xml = xml + '</instance>';
    }
    // block for local variables
    {
     xml = xml + '<id>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._id);
     xml = xml + '</id>';
    }
    // block for local variables
    {
     xml = xml + '<title>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._title);
     xml = xml + '</title>';
    }
    // block for local variables
    {
     xml = xml + this._position.serialize(cxfjsutils, 'position', null);
    }
    // block for local variables
    {
     if (this._parameters != null) {
      for (var ax = 0;ax < this._parameters.length;ax ++) {
       if (this._parameters[ax] == null) {
        xml = xml + '<parameters xsi:nil=\'true\'/>';
       } else {
        xml = xml + this._parameters[ax].serialize(cxfjsutils, 'parameters', null);
       }
      }
     }
    }
    // block for local variables
    {
     if (this._inputPorts != null) {
      for (var ax = 0;ax < this._inputPorts.length;ax ++) {
       if (this._inputPorts[ax] == null) {
        xml = xml + '<inputPorts/>';
       } else {
        xml = xml + this._inputPorts[ax].serialize(cxfjsutils, 'inputPorts', null);
       }
      }
     }
    }
    // block for local variables
    {
     if (this._outputPorts != null) {
      for (var ax = 0;ax < this._outputPorts.length;ax ++) {
       if (this._outputPorts[ax] == null) {
        xml = xml + '<outputPorts/>';
       } else {
        xml = xml + this._outputPorts[ax].serialize(cxfjsutils, 'outputPorts', null);
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_Module.prototype.serialize = COVISE_Module_serialize;

function COVISE_Module_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_Module();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing category');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setCategory(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing host');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setHost(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing instance');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setInstance(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing id');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setId(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing title');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setTitle(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing position');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Point_deserialize(cxfjsutils, curElement);
    }
    newobject.setPosition(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing parameters');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'parameters')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_Parameter_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'parameters'));
     newobject.setParameters(item);
     var item = null;
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing inputPorts');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'inputPorts')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_Port_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'inputPorts'));
     newobject.setInputPorts(item);
     var item = null;
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing outputPorts');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'outputPorts')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_Port_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'outputPorts'));
     newobject.setOutputPorts(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}deleteDirResponse
//
function COVISE_deleteDirResponse () {
    this.typeMarker = 'COVISE_deleteDirResponse';
    this._result = '';
}

//
// accessor is COVISE_deleteDirResponse.prototype.getResult
// element get for result
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for result
// setter function is is COVISE_deleteDirResponse.prototype.setResult
//
function COVISE_deleteDirResponse_getResult() { return this._result;}

COVISE_deleteDirResponse.prototype.getResult = COVISE_deleteDirResponse_getResult;

function COVISE_deleteDirResponse_setResult(value) { this._result = value;}

COVISE_deleteDirResponse.prototype.setResult = COVISE_deleteDirResponse_setResult;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}deleteDirResponse
//
function COVISE_deleteDirResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<result>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._result);
     xml = xml + '</result>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_deleteDirResponse.prototype.serialize = COVISE_deleteDirResponse_serialize;

function COVISE_deleteDirResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_deleteDirResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing result');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setResult(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}unlink
//
function COVISE_unlink () {
    this.typeMarker = 'COVISE_unlink';
    this._linkID = '';
}

//
// accessor is COVISE_unlink.prototype.getLinkID
// element get for linkID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for linkID
// setter function is is COVISE_unlink.prototype.setLinkID
//
function COVISE_unlink_getLinkID() { return this._linkID;}

COVISE_unlink.prototype.getLinkID = COVISE_unlink_getLinkID;

function COVISE_unlink_setLinkID(value) { this._linkID = value;}

COVISE_unlink.prototype.setLinkID = COVISE_unlink_setLinkID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}unlink
//
function COVISE_unlink_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<linkID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._linkID);
     xml = xml + '</linkID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_unlink.prototype.serialize = COVISE_unlink_serialize;

function COVISE_unlink_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_unlink();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing linkID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setLinkID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}unlinkResponse
//
function COVISE_unlinkResponse () {
    this.typeMarker = 'COVISE_unlinkResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}unlinkResponse
//
function COVISE_unlinkResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_unlinkResponse.prototype.serialize = COVISE_unlinkResponse_serialize;

function COVISE_unlinkResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_unlinkResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}OpenNetEvent
//
function COVISE_OpenNetEvent () {
    this.typeMarker = 'COVISE_OpenNetEvent';
    this._type = '';
    this._mapname = '';
}

//
// accessor is COVISE_OpenNetEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_OpenNetEvent.prototype.setType
//
function COVISE_OpenNetEvent_getType() { return this._type;}

COVISE_OpenNetEvent.prototype.getType = COVISE_OpenNetEvent_getType;

function COVISE_OpenNetEvent_setType(value) { this._type = value;}

COVISE_OpenNetEvent.prototype.setType = COVISE_OpenNetEvent_setType;
//
// accessor is COVISE_OpenNetEvent.prototype.getMapname
// element get for mapname
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for mapname
// setter function is is COVISE_OpenNetEvent.prototype.setMapname
//
function COVISE_OpenNetEvent_getMapname() { return this._mapname;}

COVISE_OpenNetEvent.prototype.getMapname = COVISE_OpenNetEvent_getMapname;

function COVISE_OpenNetEvent_setMapname(value) { this._mapname = value;}

COVISE_OpenNetEvent.prototype.setMapname = COVISE_OpenNetEvent_setMapname;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}OpenNetEvent
//
function COVISE_OpenNetEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<mapname>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapname);
     xml = xml + '</mapname>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_OpenNetEvent.prototype.serialize = COVISE_OpenNetEvent_serialize;

function COVISE_OpenNetEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_OpenNetEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapname');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setMapname(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getRunningModule
//
function COVISE_getRunningModule () {
    this.typeMarker = 'COVISE_getRunningModule';
    this._moduleID = '';
}

//
// accessor is COVISE_getRunningModule.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_getRunningModule.prototype.setModuleID
//
function COVISE_getRunningModule_getModuleID() { return this._moduleID;}

COVISE_getRunningModule.prototype.getModuleID = COVISE_getRunningModule_getModuleID;

function COVISE_getRunningModule_setModuleID(value) { this._moduleID = value;}

COVISE_getRunningModule.prototype.setModuleID = COVISE_getRunningModule_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getRunningModule
//
function COVISE_getRunningModule_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getRunningModule.prototype.serialize = COVISE_getRunningModule_serialize;

function COVISE_getRunningModule_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getRunningModule();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ExecuteFinishEvent
//
function COVISE_ExecuteFinishEvent () {
    this.typeMarker = 'COVISE_ExecuteFinishEvent';
    this._type = '';
}

//
// accessor is COVISE_ExecuteFinishEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ExecuteFinishEvent.prototype.setType
//
function COVISE_ExecuteFinishEvent_getType() { return this._type;}

COVISE_ExecuteFinishEvent.prototype.getType = COVISE_ExecuteFinishEvent_getType;

function COVISE_ExecuteFinishEvent_setType(value) { this._type = value;}

COVISE_ExecuteFinishEvent.prototype.setType = COVISE_ExecuteFinishEvent_setType;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ExecuteFinishEvent
//
function COVISE_ExecuteFinishEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ExecuteFinishEvent.prototype.serialize = COVISE_ExecuteFinishEvent_serialize;

function COVISE_ExecuteFinishEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ExecuteFinishEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}executeNet
//
function COVISE_executeNet () {
    this.typeMarker = 'COVISE_executeNet';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}executeNet
//
function COVISE_executeNet_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_executeNet.prototype.serialize = COVISE_executeNet_serialize;

function COVISE_executeNet_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_executeNet();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}executeNetResponse
//
function COVISE_executeNetResponse () {
    this.typeMarker = 'COVISE_executeNetResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}executeNetResponse
//
function COVISE_executeNetResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_executeNetResponse.prototype.serialize = COVISE_executeNetResponse_serialize;

function COVISE_executeNetResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_executeNetResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ModuleExecuteFinishEvent
//
function COVISE_ModuleExecuteFinishEvent () {
    this.typeMarker = 'COVISE_ModuleExecuteFinishEvent';
    this._type = '';
    this._moduleID = '';
}

//
// accessor is COVISE_ModuleExecuteFinishEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ModuleExecuteFinishEvent.prototype.setType
//
function COVISE_ModuleExecuteFinishEvent_getType() { return this._type;}

COVISE_ModuleExecuteFinishEvent.prototype.getType = COVISE_ModuleExecuteFinishEvent_getType;

function COVISE_ModuleExecuteFinishEvent_setType(value) { this._type = value;}

COVISE_ModuleExecuteFinishEvent.prototype.setType = COVISE_ModuleExecuteFinishEvent_setType;
//
// accessor is COVISE_ModuleExecuteFinishEvent.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_ModuleExecuteFinishEvent.prototype.setModuleID
//
function COVISE_ModuleExecuteFinishEvent_getModuleID() { return this._moduleID;}

COVISE_ModuleExecuteFinishEvent.prototype.getModuleID = COVISE_ModuleExecuteFinishEvent_getModuleID;

function COVISE_ModuleExecuteFinishEvent_setModuleID(value) { this._moduleID = value;}

COVISE_ModuleExecuteFinishEvent.prototype.setModuleID = COVISE_ModuleExecuteFinishEvent_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ModuleExecuteFinishEvent
//
function COVISE_ModuleExecuteFinishEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ModuleExecuteFinishEvent.prototype.serialize = COVISE_ModuleExecuteFinishEvent_serialize;

function COVISE_ModuleExecuteFinishEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ModuleExecuteFinishEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ColormapPin
//
function COVISE_ColormapPin () {
    this.typeMarker = 'COVISE_ColormapPin';
    this._r = 0.0;
    this._g = 0.0;
    this._b = 0.0;
    this._a = 0.0;
    this._position = 0.0;
}

//
// accessor is COVISE_ColormapPin.prototype.getR
// element get for r
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for r
// setter function is is COVISE_ColormapPin.prototype.setR
//
function COVISE_ColormapPin_getR() { return this._r;}

COVISE_ColormapPin.prototype.getR = COVISE_ColormapPin_getR;

function COVISE_ColormapPin_setR(value) { this._r = value;}

COVISE_ColormapPin.prototype.setR = COVISE_ColormapPin_setR;
//
// accessor is COVISE_ColormapPin.prototype.getG
// element get for g
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for g
// setter function is is COVISE_ColormapPin.prototype.setG
//
function COVISE_ColormapPin_getG() { return this._g;}

COVISE_ColormapPin.prototype.getG = COVISE_ColormapPin_getG;

function COVISE_ColormapPin_setG(value) { this._g = value;}

COVISE_ColormapPin.prototype.setG = COVISE_ColormapPin_setG;
//
// accessor is COVISE_ColormapPin.prototype.getB
// element get for b
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for b
// setter function is is COVISE_ColormapPin.prototype.setB
//
function COVISE_ColormapPin_getB() { return this._b;}

COVISE_ColormapPin.prototype.getB = COVISE_ColormapPin_getB;

function COVISE_ColormapPin_setB(value) { this._b = value;}

COVISE_ColormapPin.prototype.setB = COVISE_ColormapPin_setB;
//
// accessor is COVISE_ColormapPin.prototype.getA
// element get for a
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for a
// setter function is is COVISE_ColormapPin.prototype.setA
//
function COVISE_ColormapPin_getA() { return this._a;}

COVISE_ColormapPin.prototype.getA = COVISE_ColormapPin_getA;

function COVISE_ColormapPin_setA(value) { this._a = value;}

COVISE_ColormapPin.prototype.setA = COVISE_ColormapPin_setA;
//
// accessor is COVISE_ColormapPin.prototype.getPosition
// element get for position
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for position
// setter function is is COVISE_ColormapPin.prototype.setPosition
//
function COVISE_ColormapPin_getPosition() { return this._position;}

COVISE_ColormapPin.prototype.getPosition = COVISE_ColormapPin_getPosition;

function COVISE_ColormapPin_setPosition(value) { this._position = value;}

COVISE_ColormapPin.prototype.setPosition = COVISE_ColormapPin_setPosition;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ColormapPin
//
function COVISE_ColormapPin_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<r>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._r);
     xml = xml + '</r>';
    }
    // block for local variables
    {
     xml = xml + '<g>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._g);
     xml = xml + '</g>';
    }
    // block for local variables
    {
     xml = xml + '<b>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._b);
     xml = xml + '</b>';
    }
    // block for local variables
    {
     xml = xml + '<a>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._a);
     xml = xml + '</a>';
    }
    // block for local variables
    {
     xml = xml + '<position>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._position);
     xml = xml + '</position>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ColormapPin.prototype.serialize = COVISE_ColormapPin_serialize;

function COVISE_ColormapPin_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ColormapPin();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing r');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setR(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing g');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setG(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing b');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setB(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing a');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setA(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing position');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setPosition(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getConfigEntryResponse
//
function COVISE_getConfigEntryResponse () {
    this.typeMarker = 'COVISE_getConfigEntryResponse';
    this._value = '';
}

//
// accessor is COVISE_getConfigEntryResponse.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for value
// setter function is is COVISE_getConfigEntryResponse.prototype.setValue
//
function COVISE_getConfigEntryResponse_getValue() { return this._value;}

COVISE_getConfigEntryResponse.prototype.getValue = COVISE_getConfigEntryResponse_getValue;

function COVISE_getConfigEntryResponse_setValue(value) { this._value = value;}

COVISE_getConfigEntryResponse.prototype.setValue = COVISE_getConfigEntryResponse_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getConfigEntryResponse
//
function COVISE_getConfigEntryResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getConfigEntryResponse.prototype.serialize = COVISE_getConfigEntryResponse_serialize;

function COVISE_getConfigEntryResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getConfigEntryResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}Point
//
function COVISE_Point () {
    this.typeMarker = 'COVISE_Point';
    this._x = 0;
    this._y = 0;
}

//
// accessor is COVISE_Point.prototype.getX
// element get for x
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for x
// setter function is is COVISE_Point.prototype.setX
//
function COVISE_Point_getX() { return this._x;}

COVISE_Point.prototype.getX = COVISE_Point_getX;

function COVISE_Point_setX(value) { this._x = value;}

COVISE_Point.prototype.setX = COVISE_Point_setX;
//
// accessor is COVISE_Point.prototype.getY
// element get for y
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for y
// setter function is is COVISE_Point.prototype.setY
//
function COVISE_Point_getY() { return this._y;}

COVISE_Point.prototype.getY = COVISE_Point_getY;

function COVISE_Point_setY(value) { this._y = value;}

COVISE_Point.prototype.setY = COVISE_Point_setY;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}Point
//
function COVISE_Point_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<x>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._x);
     xml = xml + '</x>';
    }
    // block for local variables
    {
     xml = xml + '<y>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._y);
     xml = xml + '</y>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_Point.prototype.serialize = COVISE_Point_serialize;

function COVISE_Point_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_Point();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing x');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setX(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing y');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setY(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ChoiceParameter
//
function COVISE_ChoiceParameter () {
    this.typeMarker = 'COVISE_ChoiceParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._selected = 0;
    this._choices = [];
}

//
// accessor is COVISE_ChoiceParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_ChoiceParameter.prototype.setName
//
function COVISE_ChoiceParameter_getName() { return this._name;}

COVISE_ChoiceParameter.prototype.getName = COVISE_ChoiceParameter_getName;

function COVISE_ChoiceParameter_setName(value) { this._name = value;}

COVISE_ChoiceParameter.prototype.setName = COVISE_ChoiceParameter_setName;
//
// accessor is COVISE_ChoiceParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ChoiceParameter.prototype.setType
//
function COVISE_ChoiceParameter_getType() { return this._type;}

COVISE_ChoiceParameter.prototype.getType = COVISE_ChoiceParameter_getType;

function COVISE_ChoiceParameter_setType(value) { this._type = value;}

COVISE_ChoiceParameter.prototype.setType = COVISE_ChoiceParameter_setType;
//
// accessor is COVISE_ChoiceParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_ChoiceParameter.prototype.setDescription
//
function COVISE_ChoiceParameter_getDescription() { return this._description;}

COVISE_ChoiceParameter.prototype.getDescription = COVISE_ChoiceParameter_getDescription;

function COVISE_ChoiceParameter_setDescription(value) { this._description = value;}

COVISE_ChoiceParameter.prototype.setDescription = COVISE_ChoiceParameter_setDescription;
//
// accessor is COVISE_ChoiceParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_ChoiceParameter.prototype.setMapped
//
function COVISE_ChoiceParameter_getMapped() { return this._mapped;}

COVISE_ChoiceParameter.prototype.getMapped = COVISE_ChoiceParameter_getMapped;

function COVISE_ChoiceParameter_setMapped(value) { this._mapped = value;}

COVISE_ChoiceParameter.prototype.setMapped = COVISE_ChoiceParameter_setMapped;
//
// accessor is COVISE_ChoiceParameter.prototype.getSelected
// element get for selected
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for selected
// setter function is is COVISE_ChoiceParameter.prototype.setSelected
//
function COVISE_ChoiceParameter_getSelected() { return this._selected;}

COVISE_ChoiceParameter.prototype.getSelected = COVISE_ChoiceParameter_getSelected;

function COVISE_ChoiceParameter_setSelected(value) { this._selected = value;}

COVISE_ChoiceParameter.prototype.setSelected = COVISE_ChoiceParameter_setSelected;
//
// accessor is COVISE_ChoiceParameter.prototype.getChoices
// element get for choices
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
// - array
//
// element set for choices
// setter function is is COVISE_ChoiceParameter.prototype.setChoices
//
function COVISE_ChoiceParameter_getChoices() { return this._choices;}

COVISE_ChoiceParameter.prototype.getChoices = COVISE_ChoiceParameter_getChoices;

function COVISE_ChoiceParameter_setChoices(value) { this._choices = value;}

COVISE_ChoiceParameter.prototype.setChoices = COVISE_ChoiceParameter_setChoices;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ChoiceParameter
//
function COVISE_ChoiceParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<selected>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._selected);
     xml = xml + '</selected>';
    }
    // block for local variables
    {
     if (this._choices != null) {
      for (var ax = 0;ax < this._choices.length;ax ++) {
       if (this._choices[ax] == null) {
        xml = xml + '<choices/>';
       } else {
        xml = xml + '<choices>';
        xml = xml + cxfjsutils.escapeXmlEntities(this._choices[ax]);
        xml = xml + '</choices>';
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ChoiceParameter.prototype.serialize = COVISE_ChoiceParameter_serialize;

function COVISE_ChoiceParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ChoiceParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing selected');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setSelected(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing choices');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'choices')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       value = cxfjsutils.getNodeText(curElement);
       arrayItem = value;
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'choices'));
     newobject.setChoices(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}deleteModule
//
function COVISE_deleteModule () {
    this.typeMarker = 'COVISE_deleteModule';
    this._moduleID = '';
}

//
// accessor is COVISE_deleteModule.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_deleteModule.prototype.setModuleID
//
function COVISE_deleteModule_getModuleID() { return this._moduleID;}

COVISE_deleteModule.prototype.getModuleID = COVISE_deleteModule_getModuleID;

function COVISE_deleteModule_setModuleID(value) { this._moduleID = value;}

COVISE_deleteModule.prototype.setModuleID = COVISE_deleteModule_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}deleteModule
//
function COVISE_deleteModule_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_deleteModule.prototype.serialize = COVISE_deleteModule_serialize;

function COVISE_deleteModule_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_deleteModule();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}openNetResponse
//
function COVISE_openNetResponse () {
    this.typeMarker = 'COVISE_openNetResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}openNetResponse
//
function COVISE_openNetResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_openNetResponse.prototype.serialize = COVISE_openNetResponse_serialize;

function COVISE_openNetResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_openNetResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}setParameterFromStringResponse
//
function COVISE_setParameterFromStringResponse () {
    this.typeMarker = 'COVISE_setParameterFromStringResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}setParameterFromStringResponse
//
function COVISE_setParameterFromStringResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_setParameterFromStringResponse.prototype.serialize = COVISE_setParameterFromStringResponse_serialize;

function COVISE_setParameterFromStringResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_setParameterFromStringResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}StringParameter
//
function COVISE_StringParameter () {
    this.typeMarker = 'COVISE_StringParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = '';
}

//
// accessor is COVISE_StringParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_StringParameter.prototype.setName
//
function COVISE_StringParameter_getName() { return this._name;}

COVISE_StringParameter.prototype.getName = COVISE_StringParameter_getName;

function COVISE_StringParameter_setName(value) { this._name = value;}

COVISE_StringParameter.prototype.setName = COVISE_StringParameter_setName;
//
// accessor is COVISE_StringParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_StringParameter.prototype.setType
//
function COVISE_StringParameter_getType() { return this._type;}

COVISE_StringParameter.prototype.getType = COVISE_StringParameter_getType;

function COVISE_StringParameter_setType(value) { this._type = value;}

COVISE_StringParameter.prototype.setType = COVISE_StringParameter_setType;
//
// accessor is COVISE_StringParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_StringParameter.prototype.setDescription
//
function COVISE_StringParameter_getDescription() { return this._description;}

COVISE_StringParameter.prototype.getDescription = COVISE_StringParameter_getDescription;

function COVISE_StringParameter_setDescription(value) { this._description = value;}

COVISE_StringParameter.prototype.setDescription = COVISE_StringParameter_setDescription;
//
// accessor is COVISE_StringParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_StringParameter.prototype.setMapped
//
function COVISE_StringParameter_getMapped() { return this._mapped;}

COVISE_StringParameter.prototype.getMapped = COVISE_StringParameter_getMapped;

function COVISE_StringParameter_setMapped(value) { this._mapped = value;}

COVISE_StringParameter.prototype.setMapped = COVISE_StringParameter_setMapped;
//
// accessor is COVISE_StringParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for value
// setter function is is COVISE_StringParameter.prototype.setValue
//
function COVISE_StringParameter_getValue() { return this._value;}

COVISE_StringParameter.prototype.getValue = COVISE_StringParameter_getValue;

function COVISE_StringParameter_setValue(value) { this._value = value;}

COVISE_StringParameter.prototype.setValue = COVISE_StringParameter_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}StringParameter
//
function COVISE_StringParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_StringParameter.prototype.serialize = COVISE_StringParameter_serialize;

function COVISE_StringParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_StringParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}listHostsResponse
//
function COVISE_listHostsResponse () {
    this.typeMarker = 'COVISE_listHostsResponse';
    this._hosts = [];
}

//
// accessor is COVISE_listHostsResponse.prototype.getHosts
// element get for hosts
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
// - array
//
// element set for hosts
// setter function is is COVISE_listHostsResponse.prototype.setHosts
//
function COVISE_listHostsResponse_getHosts() { return this._hosts;}

COVISE_listHostsResponse.prototype.getHosts = COVISE_listHostsResponse_getHosts;

function COVISE_listHostsResponse_setHosts(value) { this._hosts = value;}

COVISE_listHostsResponse.prototype.setHosts = COVISE_listHostsResponse_setHosts;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}listHostsResponse
//
function COVISE_listHostsResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     if (this._hosts != null) {
      for (var ax = 0;ax < this._hosts.length;ax ++) {
       if (this._hosts[ax] == null) {
        xml = xml + '<hosts/>';
       } else {
        xml = xml + '<hosts>';
        xml = xml + cxfjsutils.escapeXmlEntities(this._hosts[ax]);
        xml = xml + '</hosts>';
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_listHostsResponse.prototype.serialize = COVISE_listHostsResponse_serialize;

function COVISE_listHostsResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_listHostsResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing hosts');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'hosts')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       value = cxfjsutils.getNodeText(curElement);
       arrayItem = value;
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'hosts'));
     newobject.setHosts(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}getConfigEntry
//
function COVISE_getConfigEntry () {
    this.typeMarker = 'COVISE_getConfigEntry';
    this._section = '';
    this._variable = '';
}

//
// accessor is COVISE_getConfigEntry.prototype.getSection
// element get for section
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for section
// setter function is is COVISE_getConfigEntry.prototype.setSection
//
function COVISE_getConfigEntry_getSection() { return this._section;}

COVISE_getConfigEntry.prototype.getSection = COVISE_getConfigEntry_getSection;

function COVISE_getConfigEntry_setSection(value) { this._section = value;}

COVISE_getConfigEntry.prototype.setSection = COVISE_getConfigEntry_setSection;
//
// accessor is COVISE_getConfigEntry.prototype.getVariable
// element get for variable
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for variable
// setter function is is COVISE_getConfigEntry.prototype.setVariable
//
function COVISE_getConfigEntry_getVariable() { return this._variable;}

COVISE_getConfigEntry.prototype.getVariable = COVISE_getConfigEntry_getVariable;

function COVISE_getConfigEntry_setVariable(value) { this._variable = value;}

COVISE_getConfigEntry.prototype.setVariable = COVISE_getConfigEntry_setVariable;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}getConfigEntry
//
function COVISE_getConfigEntry_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<section>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._section);
     xml = xml + '</section>';
    }
    // block for local variables
    {
     xml = xml + '<variable>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._variable);
     xml = xml + '</variable>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_getConfigEntry.prototype.serialize = COVISE_getConfigEntry_serialize;

function COVISE_getConfigEntry_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_getConfigEntry();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing section');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setSection(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing variable');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setVariable(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}IntScalarParameter
//
function COVISE_IntScalarParameter () {
    this.typeMarker = 'COVISE_IntScalarParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = 0;
}

//
// accessor is COVISE_IntScalarParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_IntScalarParameter.prototype.setName
//
function COVISE_IntScalarParameter_getName() { return this._name;}

COVISE_IntScalarParameter.prototype.getName = COVISE_IntScalarParameter_getName;

function COVISE_IntScalarParameter_setName(value) { this._name = value;}

COVISE_IntScalarParameter.prototype.setName = COVISE_IntScalarParameter_setName;
//
// accessor is COVISE_IntScalarParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_IntScalarParameter.prototype.setType
//
function COVISE_IntScalarParameter_getType() { return this._type;}

COVISE_IntScalarParameter.prototype.getType = COVISE_IntScalarParameter_getType;

function COVISE_IntScalarParameter_setType(value) { this._type = value;}

COVISE_IntScalarParameter.prototype.setType = COVISE_IntScalarParameter_setType;
//
// accessor is COVISE_IntScalarParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_IntScalarParameter.prototype.setDescription
//
function COVISE_IntScalarParameter_getDescription() { return this._description;}

COVISE_IntScalarParameter.prototype.getDescription = COVISE_IntScalarParameter_getDescription;

function COVISE_IntScalarParameter_setDescription(value) { this._description = value;}

COVISE_IntScalarParameter.prototype.setDescription = COVISE_IntScalarParameter_setDescription;
//
// accessor is COVISE_IntScalarParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_IntScalarParameter.prototype.setMapped
//
function COVISE_IntScalarParameter_getMapped() { return this._mapped;}

COVISE_IntScalarParameter.prototype.getMapped = COVISE_IntScalarParameter_getMapped;

function COVISE_IntScalarParameter_setMapped(value) { this._mapped = value;}

COVISE_IntScalarParameter.prototype.setMapped = COVISE_IntScalarParameter_setMapped;
//
// accessor is COVISE_IntScalarParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for value
// setter function is is COVISE_IntScalarParameter.prototype.setValue
//
function COVISE_IntScalarParameter_getValue() { return this._value;}

COVISE_IntScalarParameter.prototype.getValue = COVISE_IntScalarParameter_getValue;

function COVISE_IntScalarParameter_setValue(value) { this._value = value;}

COVISE_IntScalarParameter.prototype.setValue = COVISE_IntScalarParameter_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}IntScalarParameter
//
function COVISE_IntScalarParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_IntScalarParameter.prototype.serialize = COVISE_IntScalarParameter_serialize;

function COVISE_IntScalarParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_IntScalarParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}linkResponse
//
function COVISE_linkResponse () {
    this.typeMarker = 'COVISE_linkResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}linkResponse
//
function COVISE_linkResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_linkResponse.prototype.serialize = COVISE_linkResponse_serialize;

function COVISE_linkResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_linkResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}IntVectorParameter
//
function COVISE_IntVectorParameter () {
    this.typeMarker = 'COVISE_IntVectorParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = [];
}

//
// accessor is COVISE_IntVectorParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_IntVectorParameter.prototype.setName
//
function COVISE_IntVectorParameter_getName() { return this._name;}

COVISE_IntVectorParameter.prototype.getName = COVISE_IntVectorParameter_getName;

function COVISE_IntVectorParameter_setName(value) { this._name = value;}

COVISE_IntVectorParameter.prototype.setName = COVISE_IntVectorParameter_setName;
//
// accessor is COVISE_IntVectorParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_IntVectorParameter.prototype.setType
//
function COVISE_IntVectorParameter_getType() { return this._type;}

COVISE_IntVectorParameter.prototype.getType = COVISE_IntVectorParameter_getType;

function COVISE_IntVectorParameter_setType(value) { this._type = value;}

COVISE_IntVectorParameter.prototype.setType = COVISE_IntVectorParameter_setType;
//
// accessor is COVISE_IntVectorParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_IntVectorParameter.prototype.setDescription
//
function COVISE_IntVectorParameter_getDescription() { return this._description;}

COVISE_IntVectorParameter.prototype.getDescription = COVISE_IntVectorParameter_getDescription;

function COVISE_IntVectorParameter_setDescription(value) { this._description = value;}

COVISE_IntVectorParameter.prototype.setDescription = COVISE_IntVectorParameter_setDescription;
//
// accessor is COVISE_IntVectorParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_IntVectorParameter.prototype.setMapped
//
function COVISE_IntVectorParameter_getMapped() { return this._mapped;}

COVISE_IntVectorParameter.prototype.getMapped = COVISE_IntVectorParameter_getMapped;

function COVISE_IntVectorParameter_setMapped(value) { this._mapped = value;}

COVISE_IntVectorParameter.prototype.setMapped = COVISE_IntVectorParameter_setMapped;
//
// accessor is COVISE_IntVectorParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
// - array
//
// element set for value
// setter function is is COVISE_IntVectorParameter.prototype.setValue
//
function COVISE_IntVectorParameter_getValue() { return this._value;}

COVISE_IntVectorParameter.prototype.getValue = COVISE_IntVectorParameter_getValue;

function COVISE_IntVectorParameter_setValue(value) { this._value = value;}

COVISE_IntVectorParameter.prototype.setValue = COVISE_IntVectorParameter_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}IntVectorParameter
//
function COVISE_IntVectorParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     if (this._value != null) {
      for (var ax = 0;ax < this._value.length;ax ++) {
       if (this._value[ax] == null) {
        xml = xml + '<value/>';
       } else {
        xml = xml + '<value>';
        xml = xml + cxfjsutils.escapeXmlEntities(this._value[ax]);
        xml = xml + '</value>';
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_IntVectorParameter.prototype.serialize = COVISE_IntVectorParameter_serialize;

function COVISE_IntVectorParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_IntVectorParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'value')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       value = cxfjsutils.getNodeText(curElement);
       arrayItem = parseInt(value);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'value'));
     newobject.setValue(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}isDirExistResponse
//
function COVISE_isDirExistResponse () {
    this.typeMarker = 'COVISE_isDirExistResponse';
    this._result = '';
    this._isDirExist = '';
}

//
// accessor is COVISE_isDirExistResponse.prototype.getResult
// element get for result
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for result
// setter function is is COVISE_isDirExistResponse.prototype.setResult
//
function COVISE_isDirExistResponse_getResult() { return this._result;}

COVISE_isDirExistResponse.prototype.getResult = COVISE_isDirExistResponse_getResult;

function COVISE_isDirExistResponse_setResult(value) { this._result = value;}

COVISE_isDirExistResponse.prototype.setResult = COVISE_isDirExistResponse_setResult;
//
// accessor is COVISE_isDirExistResponse.prototype.getIsDirExist
// element get for isDirExist
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for isDirExist
// setter function is is COVISE_isDirExistResponse.prototype.setIsDirExist
//
function COVISE_isDirExistResponse_getIsDirExist() { return this._isDirExist;}

COVISE_isDirExistResponse.prototype.getIsDirExist = COVISE_isDirExistResponse_getIsDirExist;

function COVISE_isDirExistResponse_setIsDirExist(value) { this._isDirExist = value;}

COVISE_isDirExistResponse.prototype.setIsDirExist = COVISE_isDirExistResponse_setIsDirExist;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}isDirExistResponse
//
function COVISE_isDirExistResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<result>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._result);
     xml = xml + '</result>';
    }
    // block for local variables
    {
     xml = xml + '<isDirExist>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._isDirExist);
     xml = xml + '</isDirExist>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_isDirExistResponse.prototype.serialize = COVISE_isDirExistResponse_serialize;

function COVISE_isDirExistResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_isDirExistResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing result');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setResult(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing isDirExist');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setIsDirExist(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}Link
//
function COVISE_Link () {
    this.typeMarker = 'COVISE_Link';
    this._id = '';
    this._from = null;
    this._to = null;
}

//
// accessor is COVISE_Link.prototype.getId
// element get for id
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for id
// setter function is is COVISE_Link.prototype.setId
//
function COVISE_Link_getId() { return this._id;}

COVISE_Link.prototype.getId = COVISE_Link_getId;

function COVISE_Link_setId(value) { this._id = value;}

COVISE_Link.prototype.setId = COVISE_Link_setId;
//
// accessor is COVISE_Link.prototype.getFrom
// element get for from
// - element type is {http://www.hlrs.de/organization/vis/covise}Port
// - required element
//
// element set for from
// setter function is is COVISE_Link.prototype.setFrom
//
function COVISE_Link_getFrom() { return this._from;}

COVISE_Link.prototype.getFrom = COVISE_Link_getFrom;

function COVISE_Link_setFrom(value) { this._from = value;}

COVISE_Link.prototype.setFrom = COVISE_Link_setFrom;
//
// accessor is COVISE_Link.prototype.getTo
// element get for to
// - element type is {http://www.hlrs.de/organization/vis/covise}Port
// - required element
//
// element set for to
// setter function is is COVISE_Link.prototype.setTo
//
function COVISE_Link_getTo() { return this._to;}

COVISE_Link.prototype.getTo = COVISE_Link_getTo;

function COVISE_Link_setTo(value) { this._to = value;}

COVISE_Link.prototype.setTo = COVISE_Link_setTo;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}Link
//
function COVISE_Link_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<id>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._id);
     xml = xml + '</id>';
    }
    // block for local variables
    {
     xml = xml + this._from.serialize(cxfjsutils, 'from', null);
    }
    // block for local variables
    {
     xml = xml + this._to.serialize(cxfjsutils, 'to', null);
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_Link.prototype.serialize = COVISE_Link_serialize;

function COVISE_Link_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_Link();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing id');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setId(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing from');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Port_deserialize(cxfjsutils, curElement);
    }
    newobject.setFrom(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing to');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = COVISE_Port_deserialize(cxfjsutils, curElement);
    }
    newobject.setTo(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}quit
//
function COVISE_quit () {
    this.typeMarker = 'COVISE_quit';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}quit
//
function COVISE_quit_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_quit.prototype.serialize = COVISE_quit_serialize;

function COVISE_quit_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_quit();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}uploadFileMtom
//
function COVISE_uploadFileMtom () {
    this.typeMarker = 'COVISE_uploadFileMtom';
    this._path = '';
    this._fileName = '';
    this._fileData = '';
    this._fileTrunc = '';
    this._fileSize = 0;
}

//
// accessor is COVISE_uploadFileMtom.prototype.getPath
// element get for path
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for path
// setter function is is COVISE_uploadFileMtom.prototype.setPath
//
function COVISE_uploadFileMtom_getPath() { return this._path;}

COVISE_uploadFileMtom.prototype.getPath = COVISE_uploadFileMtom_getPath;

function COVISE_uploadFileMtom_setPath(value) { this._path = value;}

COVISE_uploadFileMtom.prototype.setPath = COVISE_uploadFileMtom_setPath;
//
// accessor is COVISE_uploadFileMtom.prototype.getFileName
// element get for fileName
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for fileName
// setter function is is COVISE_uploadFileMtom.prototype.setFileName
//
function COVISE_uploadFileMtom_getFileName() { return this._fileName;}

COVISE_uploadFileMtom.prototype.getFileName = COVISE_uploadFileMtom_getFileName;

function COVISE_uploadFileMtom_setFileName(value) { this._fileName = value;}

COVISE_uploadFileMtom.prototype.setFileName = COVISE_uploadFileMtom_setFileName;
//
// accessor is COVISE_uploadFileMtom.prototype.getFileData
// element get for fileData
// - element type is {http://www.w3.org/2001/XMLSchema}base64Binary
// - required element
//
// element set for fileData
// setter function is is COVISE_uploadFileMtom.prototype.setFileData
//
function COVISE_uploadFileMtom_getFileData() { return this._fileData;}

COVISE_uploadFileMtom.prototype.getFileData = COVISE_uploadFileMtom_getFileData;

function COVISE_uploadFileMtom_setFileData(value) { this._fileData = value;}

COVISE_uploadFileMtom.prototype.setFileData = COVISE_uploadFileMtom_setFileData;
//
// accessor is COVISE_uploadFileMtom.prototype.getFileTrunc
// element get for fileTrunc
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for fileTrunc
// setter function is is COVISE_uploadFileMtom.prototype.setFileTrunc
//
function COVISE_uploadFileMtom_getFileTrunc() { return this._fileTrunc;}

COVISE_uploadFileMtom.prototype.getFileTrunc = COVISE_uploadFileMtom_getFileTrunc;

function COVISE_uploadFileMtom_setFileTrunc(value) { this._fileTrunc = value;}

COVISE_uploadFileMtom.prototype.setFileTrunc = COVISE_uploadFileMtom_setFileTrunc;
//
// accessor is COVISE_uploadFileMtom.prototype.getFileSize
// element get for fileSize
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for fileSize
// setter function is is COVISE_uploadFileMtom.prototype.setFileSize
//
function COVISE_uploadFileMtom_getFileSize() { return this._fileSize;}

COVISE_uploadFileMtom.prototype.getFileSize = COVISE_uploadFileMtom_getFileSize;

function COVISE_uploadFileMtom_setFileSize(value) { this._fileSize = value;}

COVISE_uploadFileMtom.prototype.setFileSize = COVISE_uploadFileMtom_setFileSize;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}uploadFileMtom
//
function COVISE_uploadFileMtom_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<path>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._path);
     xml = xml + '</path>';
    }
    // block for local variables
    {
     xml = xml + '<fileName>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileName);
     xml = xml + '</fileName>';
    }
    // block for local variables
    {
     xml = xml + '<fileData>';
     xml = xml + cxfjsutils.packageMtom(this._fileData);
     xml = xml + '</fileData>';
    }
    // block for local variables
    {
     xml = xml + '<fileTrunc>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileTrunc);
     xml = xml + '</fileTrunc>';
    }
    // block for local variables
    {
     xml = xml + '<fileSize>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileSize);
     xml = xml + '</fileSize>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_uploadFileMtom.prototype.serialize = COVISE_uploadFileMtom_serialize;

function COVISE_uploadFileMtom_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_uploadFileMtom();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing path');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setPath(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileName');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFileName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileData');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     item = cxfjsutils.deserializeBase64orMom(curElement);
    }
    newobject.setFileData(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileTrunc');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setFileTrunc(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileSize');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setFileSize(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}removeEventListener
//
function COVISE_removeEventListener () {
    this.typeMarker = 'COVISE_removeEventListener';
    this._uuid = '';
}

//
// accessor is COVISE_removeEventListener.prototype.getUuid
// element get for uuid
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for uuid
// setter function is is COVISE_removeEventListener.prototype.setUuid
//
function COVISE_removeEventListener_getUuid() { return this._uuid;}

COVISE_removeEventListener.prototype.getUuid = COVISE_removeEventListener_getUuid;

function COVISE_removeEventListener_setUuid(value) { this._uuid = value;}

COVISE_removeEventListener.prototype.setUuid = COVISE_removeEventListener_setUuid;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}removeEventListener
//
function COVISE_removeEventListener_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<uuid>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._uuid);
     xml = xml + '</uuid>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_removeEventListener.prototype.serialize = COVISE_removeEventListener_serialize;

function COVISE_removeEventListener_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_removeEventListener();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing uuid');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setUuid(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}deleteModuleResponse
//
function COVISE_deleteModuleResponse () {
    this.typeMarker = 'COVISE_deleteModuleResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}deleteModuleResponse
//
function COVISE_deleteModuleResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_deleteModuleResponse.prototype.serialize = COVISE_deleteModuleResponse_serialize;

function COVISE_deleteModuleResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_deleteModuleResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}addEventListener
//
function COVISE_addEventListener () {
    this.typeMarker = 'COVISE_addEventListener';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}addEventListener
//
function COVISE_addEventListener_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_addEventListener.prototype.serialize = COVISE_addEventListener_serialize;

function COVISE_addEventListener_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_addEventListener();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}openNet
//
function COVISE_openNet () {
    this.typeMarker = 'COVISE_openNet';
    this._filename = '';
}

//
// accessor is COVISE_openNet.prototype.getFilename
// element get for filename
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for filename
// setter function is is COVISE_openNet.prototype.setFilename
//
function COVISE_openNet_getFilename() { return this._filename;}

COVISE_openNet.prototype.getFilename = COVISE_openNet_getFilename;

function COVISE_openNet_setFilename(value) { this._filename = value;}

COVISE_openNet.prototype.setFilename = COVISE_openNet_setFilename;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}openNet
//
function COVISE_openNet_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<filename>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._filename);
     xml = xml + '</filename>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_openNet.prototype.serialize = COVISE_openNet_serialize;

function COVISE_openNet_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_openNet();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing filename');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFilename(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}FloatScalarParameter
//
function COVISE_FloatScalarParameter () {
    this.typeMarker = 'COVISE_FloatScalarParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = 0.0;
}

//
// accessor is COVISE_FloatScalarParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_FloatScalarParameter.prototype.setName
//
function COVISE_FloatScalarParameter_getName() { return this._name;}

COVISE_FloatScalarParameter.prototype.getName = COVISE_FloatScalarParameter_getName;

function COVISE_FloatScalarParameter_setName(value) { this._name = value;}

COVISE_FloatScalarParameter.prototype.setName = COVISE_FloatScalarParameter_setName;
//
// accessor is COVISE_FloatScalarParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_FloatScalarParameter.prototype.setType
//
function COVISE_FloatScalarParameter_getType() { return this._type;}

COVISE_FloatScalarParameter.prototype.getType = COVISE_FloatScalarParameter_getType;

function COVISE_FloatScalarParameter_setType(value) { this._type = value;}

COVISE_FloatScalarParameter.prototype.setType = COVISE_FloatScalarParameter_setType;
//
// accessor is COVISE_FloatScalarParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_FloatScalarParameter.prototype.setDescription
//
function COVISE_FloatScalarParameter_getDescription() { return this._description;}

COVISE_FloatScalarParameter.prototype.getDescription = COVISE_FloatScalarParameter_getDescription;

function COVISE_FloatScalarParameter_setDescription(value) { this._description = value;}

COVISE_FloatScalarParameter.prototype.setDescription = COVISE_FloatScalarParameter_setDescription;
//
// accessor is COVISE_FloatScalarParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_FloatScalarParameter.prototype.setMapped
//
function COVISE_FloatScalarParameter_getMapped() { return this._mapped;}

COVISE_FloatScalarParameter.prototype.getMapped = COVISE_FloatScalarParameter_getMapped;

function COVISE_FloatScalarParameter_setMapped(value) { this._mapped = value;}

COVISE_FloatScalarParameter.prototype.setMapped = COVISE_FloatScalarParameter_setMapped;
//
// accessor is COVISE_FloatScalarParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}float
// - required element
//
// element set for value
// setter function is is COVISE_FloatScalarParameter.prototype.setValue
//
function COVISE_FloatScalarParameter_getValue() { return this._value;}

COVISE_FloatScalarParameter.prototype.getValue = COVISE_FloatScalarParameter_getValue;

function COVISE_FloatScalarParameter_setValue(value) { this._value = value;}

COVISE_FloatScalarParameter.prototype.setValue = COVISE_FloatScalarParameter_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}FloatScalarParameter
//
function COVISE_FloatScalarParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_FloatScalarParameter.prototype.serialize = COVISE_FloatScalarParameter_serialize;

function COVISE_FloatScalarParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_FloatScalarParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseFloat(value);
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFile
//
function COVISE_setParameterFromUploadedFile () {
    this.typeMarker = 'COVISE_setParameterFromUploadedFile';
    this._moduleID = '';
    this._parameter = '';
    this._value = '';
}

//
// accessor is COVISE_setParameterFromUploadedFile.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_setParameterFromUploadedFile.prototype.setModuleID
//
function COVISE_setParameterFromUploadedFile_getModuleID() { return this._moduleID;}

COVISE_setParameterFromUploadedFile.prototype.getModuleID = COVISE_setParameterFromUploadedFile_getModuleID;

function COVISE_setParameterFromUploadedFile_setModuleID(value) { this._moduleID = value;}

COVISE_setParameterFromUploadedFile.prototype.setModuleID = COVISE_setParameterFromUploadedFile_setModuleID;
//
// accessor is COVISE_setParameterFromUploadedFile.prototype.getParameter
// element get for parameter
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for parameter
// setter function is is COVISE_setParameterFromUploadedFile.prototype.setParameter
//
function COVISE_setParameterFromUploadedFile_getParameter() { return this._parameter;}

COVISE_setParameterFromUploadedFile.prototype.getParameter = COVISE_setParameterFromUploadedFile_getParameter;

function COVISE_setParameterFromUploadedFile_setParameter(value) { this._parameter = value;}

COVISE_setParameterFromUploadedFile.prototype.setParameter = COVISE_setParameterFromUploadedFile_setParameter;
//
// accessor is COVISE_setParameterFromUploadedFile.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for value
// setter function is is COVISE_setParameterFromUploadedFile.prototype.setValue
//
function COVISE_setParameterFromUploadedFile_getValue() { return this._value;}

COVISE_setParameterFromUploadedFile.prototype.getValue = COVISE_setParameterFromUploadedFile_getValue;

function COVISE_setParameterFromUploadedFile_setValue(value) { this._value = value;}

COVISE_setParameterFromUploadedFile.prototype.setValue = COVISE_setParameterFromUploadedFile_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFile
//
function COVISE_setParameterFromUploadedFile_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    // block for local variables
    {
     xml = xml + '<parameter>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._parameter);
     xml = xml + '</parameter>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_setParameterFromUploadedFile.prototype.serialize = COVISE_setParameterFromUploadedFile_serialize;

function COVISE_setParameterFromUploadedFile_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_setParameterFromUploadedFile();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing parameter');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setParameter(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}link
//
function COVISE_link () {
    this.typeMarker = 'COVISE_link';
    this._fromModule = '';
    this._fromPort = '';
    this._toModule = '';
    this._toPort = '';
}

//
// accessor is COVISE_link.prototype.getFromModule
// element get for fromModule
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for fromModule
// setter function is is COVISE_link.prototype.setFromModule
//
function COVISE_link_getFromModule() { return this._fromModule;}

COVISE_link.prototype.getFromModule = COVISE_link_getFromModule;

function COVISE_link_setFromModule(value) { this._fromModule = value;}

COVISE_link.prototype.setFromModule = COVISE_link_setFromModule;
//
// accessor is COVISE_link.prototype.getFromPort
// element get for fromPort
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for fromPort
// setter function is is COVISE_link.prototype.setFromPort
//
function COVISE_link_getFromPort() { return this._fromPort;}

COVISE_link.prototype.getFromPort = COVISE_link_getFromPort;

function COVISE_link_setFromPort(value) { this._fromPort = value;}

COVISE_link.prototype.setFromPort = COVISE_link_setFromPort;
//
// accessor is COVISE_link.prototype.getToModule
// element get for toModule
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for toModule
// setter function is is COVISE_link.prototype.setToModule
//
function COVISE_link_getToModule() { return this._toModule;}

COVISE_link.prototype.getToModule = COVISE_link_getToModule;

function COVISE_link_setToModule(value) { this._toModule = value;}

COVISE_link.prototype.setToModule = COVISE_link_setToModule;
//
// accessor is COVISE_link.prototype.getToPort
// element get for toPort
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for toPort
// setter function is is COVISE_link.prototype.setToPort
//
function COVISE_link_getToPort() { return this._toPort;}

COVISE_link.prototype.getToPort = COVISE_link_getToPort;

function COVISE_link_setToPort(value) { this._toPort = value;}

COVISE_link.prototype.setToPort = COVISE_link_setToPort;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}link
//
function COVISE_link_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<fromModule>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fromModule);
     xml = xml + '</fromModule>';
    }
    // block for local variables
    {
     xml = xml + '<fromPort>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fromPort);
     xml = xml + '</fromPort>';
    }
    // block for local variables
    {
     xml = xml + '<toModule>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._toModule);
     xml = xml + '</toModule>';
    }
    // block for local variables
    {
     xml = xml + '<toPort>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._toPort);
     xml = xml + '</toPort>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_link.prototype.serialize = COVISE_link_serialize;

function COVISE_link_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_link();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fromModule');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFromModule(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fromPort');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFromPort(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing toModule');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setToModule(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing toPort');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setToPort(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}BooleanParameter
//
function COVISE_BooleanParameter () {
    this.typeMarker = 'COVISE_BooleanParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._value = '';
}

//
// accessor is COVISE_BooleanParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_BooleanParameter.prototype.setName
//
function COVISE_BooleanParameter_getName() { return this._name;}

COVISE_BooleanParameter.prototype.getName = COVISE_BooleanParameter_getName;

function COVISE_BooleanParameter_setName(value) { this._name = value;}

COVISE_BooleanParameter.prototype.setName = COVISE_BooleanParameter_setName;
//
// accessor is COVISE_BooleanParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_BooleanParameter.prototype.setType
//
function COVISE_BooleanParameter_getType() { return this._type;}

COVISE_BooleanParameter.prototype.getType = COVISE_BooleanParameter_getType;

function COVISE_BooleanParameter_setType(value) { this._type = value;}

COVISE_BooleanParameter.prototype.setType = COVISE_BooleanParameter_setType;
//
// accessor is COVISE_BooleanParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_BooleanParameter.prototype.setDescription
//
function COVISE_BooleanParameter_getDescription() { return this._description;}

COVISE_BooleanParameter.prototype.getDescription = COVISE_BooleanParameter_getDescription;

function COVISE_BooleanParameter_setDescription(value) { this._description = value;}

COVISE_BooleanParameter.prototype.setDescription = COVISE_BooleanParameter_setDescription;
//
// accessor is COVISE_BooleanParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_BooleanParameter.prototype.setMapped
//
function COVISE_BooleanParameter_getMapped() { return this._mapped;}

COVISE_BooleanParameter.prototype.getMapped = COVISE_BooleanParameter_getMapped;

function COVISE_BooleanParameter_setMapped(value) { this._mapped = value;}

COVISE_BooleanParameter.prototype.setMapped = COVISE_BooleanParameter_setMapped;
//
// accessor is COVISE_BooleanParameter.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for value
// setter function is is COVISE_BooleanParameter.prototype.setValue
//
function COVISE_BooleanParameter_getValue() { return this._value;}

COVISE_BooleanParameter.prototype.getValue = COVISE_BooleanParameter_getValue;

function COVISE_BooleanParameter_setValue(value) { this._value = value;}

COVISE_BooleanParameter.prototype.setValue = COVISE_BooleanParameter_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}BooleanParameter
//
function COVISE_BooleanParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_BooleanParameter.prototype.serialize = COVISE_BooleanParameter_serialize;

function COVISE_BooleanParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_BooleanParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}removeEventListenerResponse
//
function COVISE_removeEventListenerResponse () {
    this.typeMarker = 'COVISE_removeEventListenerResponse';
}

//
// Serialize {http://www.hlrs.de/organization/vis/covise}removeEventListenerResponse
//
function COVISE_removeEventListenerResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_removeEventListenerResponse.prototype.serialize = COVISE_removeEventListenerResponse_serialize;

function COVISE_removeEventListenerResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_removeEventListenerResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}listModulesResponse
//
function COVISE_listModulesResponse () {
    this.typeMarker = 'COVISE_listModulesResponse';
    this._ipaddr = '';
    this._modules = [];
}

//
// accessor is COVISE_listModulesResponse.prototype.getIpaddr
// element get for ipaddr
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for ipaddr
// setter function is is COVISE_listModulesResponse.prototype.setIpaddr
//
function COVISE_listModulesResponse_getIpaddr() { return this._ipaddr;}

COVISE_listModulesResponse.prototype.getIpaddr = COVISE_listModulesResponse_getIpaddr;

function COVISE_listModulesResponse_setIpaddr(value) { this._ipaddr = value;}

COVISE_listModulesResponse.prototype.setIpaddr = COVISE_listModulesResponse_setIpaddr;
//
// accessor is COVISE_listModulesResponse.prototype.getModules
// element get for modules
// - element type is {http://www.hlrs.de/organization/vis/covise}StringPair
// - required element
// - array
//
// element set for modules
// setter function is is COVISE_listModulesResponse.prototype.setModules
//
function COVISE_listModulesResponse_getModules() { return this._modules;}

COVISE_listModulesResponse.prototype.getModules = COVISE_listModulesResponse_getModules;

function COVISE_listModulesResponse_setModules(value) { this._modules = value;}

COVISE_listModulesResponse.prototype.setModules = COVISE_listModulesResponse_setModules;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}listModulesResponse
//
function COVISE_listModulesResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<ipaddr>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._ipaddr);
     xml = xml + '</ipaddr>';
    }
    // block for local variables
    {
     if (this._modules != null) {
      for (var ax = 0;ax < this._modules.length;ax ++) {
       if (this._modules[ax] == null) {
        xml = xml + '<modules/>';
       } else {
        xml = xml + this._modules[ax].serialize(cxfjsutils, 'modules', null);
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_listModulesResponse.prototype.serialize = COVISE_listModulesResponse_serialize;

function COVISE_listModulesResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_listModulesResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing ipaddr');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setIpaddr(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing modules');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'modules')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_StringPair_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'modules'));
     newobject.setModules(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}setParameterFromString
//
function COVISE_setParameterFromString () {
    this.typeMarker = 'COVISE_setParameterFromString';
    this._moduleID = '';
    this._parameter = '';
    this._value = '';
}

//
// accessor is COVISE_setParameterFromString.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_setParameterFromString.prototype.setModuleID
//
function COVISE_setParameterFromString_getModuleID() { return this._moduleID;}

COVISE_setParameterFromString.prototype.getModuleID = COVISE_setParameterFromString_getModuleID;

function COVISE_setParameterFromString_setModuleID(value) { this._moduleID = value;}

COVISE_setParameterFromString.prototype.setModuleID = COVISE_setParameterFromString_setModuleID;
//
// accessor is COVISE_setParameterFromString.prototype.getParameter
// element get for parameter
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for parameter
// setter function is is COVISE_setParameterFromString.prototype.setParameter
//
function COVISE_setParameterFromString_getParameter() { return this._parameter;}

COVISE_setParameterFromString.prototype.getParameter = COVISE_setParameterFromString_getParameter;

function COVISE_setParameterFromString_setParameter(value) { this._parameter = value;}

COVISE_setParameterFromString.prototype.setParameter = COVISE_setParameterFromString_setParameter;
//
// accessor is COVISE_setParameterFromString.prototype.getValue
// element get for value
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for value
// setter function is is COVISE_setParameterFromString.prototype.setValue
//
function COVISE_setParameterFromString_getValue() { return this._value;}

COVISE_setParameterFromString.prototype.getValue = COVISE_setParameterFromString_getValue;

function COVISE_setParameterFromString_setValue(value) { this._value = value;}

COVISE_setParameterFromString.prototype.setValue = COVISE_setParameterFromString_setValue;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}setParameterFromString
//
function COVISE_setParameterFromString_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    // block for local variables
    {
     xml = xml + '<parameter>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._parameter);
     xml = xml + '</parameter>';
    }
    // block for local variables
    {
     xml = xml + '<value>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._value);
     xml = xml + '</value>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_setParameterFromString.prototype.serialize = COVISE_setParameterFromString_serialize;

function COVISE_setParameterFromString_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_setParameterFromString();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing parameter');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setParameter(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing value');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setValue(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ColormapChoiceParameter
//
function COVISE_ColormapChoiceParameter () {
    this.typeMarker = 'COVISE_ColormapChoiceParameter';
    this._name = '';
    this._type = '';
    this._description = '';
    this._mapped = '';
    this._selected = 0;
    this._colormaps = [];
}

//
// accessor is COVISE_ColormapChoiceParameter.prototype.getName
// element get for name
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for name
// setter function is is COVISE_ColormapChoiceParameter.prototype.setName
//
function COVISE_ColormapChoiceParameter_getName() { return this._name;}

COVISE_ColormapChoiceParameter.prototype.getName = COVISE_ColormapChoiceParameter_getName;

function COVISE_ColormapChoiceParameter_setName(value) { this._name = value;}

COVISE_ColormapChoiceParameter.prototype.setName = COVISE_ColormapChoiceParameter_setName;
//
// accessor is COVISE_ColormapChoiceParameter.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ColormapChoiceParameter.prototype.setType
//
function COVISE_ColormapChoiceParameter_getType() { return this._type;}

COVISE_ColormapChoiceParameter.prototype.getType = COVISE_ColormapChoiceParameter_getType;

function COVISE_ColormapChoiceParameter_setType(value) { this._type = value;}

COVISE_ColormapChoiceParameter.prototype.setType = COVISE_ColormapChoiceParameter_setType;
//
// accessor is COVISE_ColormapChoiceParameter.prototype.getDescription
// element get for description
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for description
// setter function is is COVISE_ColormapChoiceParameter.prototype.setDescription
//
function COVISE_ColormapChoiceParameter_getDescription() { return this._description;}

COVISE_ColormapChoiceParameter.prototype.getDescription = COVISE_ColormapChoiceParameter_getDescription;

function COVISE_ColormapChoiceParameter_setDescription(value) { this._description = value;}

COVISE_ColormapChoiceParameter.prototype.setDescription = COVISE_ColormapChoiceParameter_setDescription;
//
// accessor is COVISE_ColormapChoiceParameter.prototype.getMapped
// element get for mapped
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for mapped
// setter function is is COVISE_ColormapChoiceParameter.prototype.setMapped
//
function COVISE_ColormapChoiceParameter_getMapped() { return this._mapped;}

COVISE_ColormapChoiceParameter.prototype.getMapped = COVISE_ColormapChoiceParameter_getMapped;

function COVISE_ColormapChoiceParameter_setMapped(value) { this._mapped = value;}

COVISE_ColormapChoiceParameter.prototype.setMapped = COVISE_ColormapChoiceParameter_setMapped;
//
// accessor is COVISE_ColormapChoiceParameter.prototype.getSelected
// element get for selected
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for selected
// setter function is is COVISE_ColormapChoiceParameter.prototype.setSelected
//
function COVISE_ColormapChoiceParameter_getSelected() { return this._selected;}

COVISE_ColormapChoiceParameter.prototype.getSelected = COVISE_ColormapChoiceParameter_getSelected;

function COVISE_ColormapChoiceParameter_setSelected(value) { this._selected = value;}

COVISE_ColormapChoiceParameter.prototype.setSelected = COVISE_ColormapChoiceParameter_setSelected;
//
// accessor is COVISE_ColormapChoiceParameter.prototype.getColormaps
// element get for colormaps
// - element type is {http://www.hlrs.de/organization/vis/covise}Colormap
// - required element
// - array
//
// element set for colormaps
// setter function is is COVISE_ColormapChoiceParameter.prototype.setColormaps
//
function COVISE_ColormapChoiceParameter_getColormaps() { return this._colormaps;}

COVISE_ColormapChoiceParameter.prototype.getColormaps = COVISE_ColormapChoiceParameter_getColormaps;

function COVISE_ColormapChoiceParameter_setColormaps(value) { this._colormaps = value;}

COVISE_ColormapChoiceParameter.prototype.setColormaps = COVISE_ColormapChoiceParameter_setColormaps;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ColormapChoiceParameter
//
function COVISE_ColormapChoiceParameter_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<name>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._name);
     xml = xml + '</name>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<description>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._description);
     xml = xml + '</description>';
    }
    // block for local variables
    {
     xml = xml + '<mapped>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._mapped);
     xml = xml + '</mapped>';
    }
    // block for local variables
    {
     xml = xml + '<selected>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._selected);
     xml = xml + '</selected>';
    }
    // block for local variables
    {
     if (this._colormaps != null) {
      for (var ax = 0;ax < this._colormaps.length;ax ++) {
       if (this._colormaps[ax] == null) {
        xml = xml + '<colormaps/>';
       } else {
        xml = xml + this._colormaps[ax].serialize(cxfjsutils, 'colormaps', null);
       }
      }
     }
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ColormapChoiceParameter.prototype.serialize = COVISE_ColormapChoiceParameter_serialize;

function COVISE_ColormapChoiceParameter_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ColormapChoiceParameter();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing name');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing description');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setDescription(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing mapped');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setMapped(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing selected');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setSelected(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing colormaps');
    if (curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'colormaps')) {
     item = [];
     do  {
      var arrayItem;
      var value = null;
      if (!cxfjsutils.isElementNil(curElement)) {
       arrayItem = COVISE_Colormap_deserialize(cxfjsutils, curElement);
      }
      item.push(arrayItem);
      curElement = cxfjsutils.getNextElementSibling(curElement);
     }
       while(curElement != null && cxfjsutils.isNodeNamedNS(curElement, '', 'colormaps'));
     newobject.setColormaps(item);
     var item = null;
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}FileInfo
//
function COVISE_FileInfo () {
    this.typeMarker = 'COVISE_FileInfo';
    this._fileName = '';
    this._isDir = '';
    this._fileSize = 0;
    this._fileDate = '';
}

//
// accessor is COVISE_FileInfo.prototype.getFileName
// element get for fileName
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for fileName
// setter function is is COVISE_FileInfo.prototype.setFileName
//
function COVISE_FileInfo_getFileName() { return this._fileName;}

COVISE_FileInfo.prototype.getFileName = COVISE_FileInfo_getFileName;

function COVISE_FileInfo_setFileName(value) { this._fileName = value;}

COVISE_FileInfo.prototype.setFileName = COVISE_FileInfo_setFileName;
//
// accessor is COVISE_FileInfo.prototype.getIsDir
// element get for isDir
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for isDir
// setter function is is COVISE_FileInfo.prototype.setIsDir
//
function COVISE_FileInfo_getIsDir() { return this._isDir;}

COVISE_FileInfo.prototype.getIsDir = COVISE_FileInfo_getIsDir;

function COVISE_FileInfo_setIsDir(value) { this._isDir = value;}

COVISE_FileInfo.prototype.setIsDir = COVISE_FileInfo_setIsDir;
//
// accessor is COVISE_FileInfo.prototype.getFileSize
// element get for fileSize
// - element type is {http://www.w3.org/2001/XMLSchema}int
// - required element
//
// element set for fileSize
// setter function is is COVISE_FileInfo.prototype.setFileSize
//
function COVISE_FileInfo_getFileSize() { return this._fileSize;}

COVISE_FileInfo.prototype.getFileSize = COVISE_FileInfo_getFileSize;

function COVISE_FileInfo_setFileSize(value) { this._fileSize = value;}

COVISE_FileInfo.prototype.setFileSize = COVISE_FileInfo_setFileSize;
//
// accessor is COVISE_FileInfo.prototype.getFileDate
// element get for fileDate
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for fileDate
// setter function is is COVISE_FileInfo.prototype.setFileDate
//
function COVISE_FileInfo_getFileDate() { return this._fileDate;}

COVISE_FileInfo.prototype.getFileDate = COVISE_FileInfo_getFileDate;

function COVISE_FileInfo_setFileDate(value) { this._fileDate = value;}

COVISE_FileInfo.prototype.setFileDate = COVISE_FileInfo_setFileDate;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}FileInfo
//
function COVISE_FileInfo_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<fileName>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileName);
     xml = xml + '</fileName>';
    }
    // block for local variables
    {
     xml = xml + '<isDir>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._isDir);
     xml = xml + '</isDir>';
    }
    // block for local variables
    {
     xml = xml + '<fileSize>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileSize);
     xml = xml + '</fileSize>';
    }
    // block for local variables
    {
     xml = xml + '<fileDate>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._fileDate);
     xml = xml + '</fileDate>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_FileInfo.prototype.serialize = COVISE_FileInfo_serialize;

function COVISE_FileInfo_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_FileInfo();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileName');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFileName(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing isDir');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setIsDir(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileSize');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = parseInt(value);
    }
    newobject.setFileSize(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing fileDate');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setFileDate(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}ModuleDelEvent
//
function COVISE_ModuleDelEvent () {
    this.typeMarker = 'COVISE_ModuleDelEvent';
    this._type = '';
    this._moduleID = '';
}

//
// accessor is COVISE_ModuleDelEvent.prototype.getType
// element get for type
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for type
// setter function is is COVISE_ModuleDelEvent.prototype.setType
//
function COVISE_ModuleDelEvent_getType() { return this._type;}

COVISE_ModuleDelEvent.prototype.getType = COVISE_ModuleDelEvent_getType;

function COVISE_ModuleDelEvent_setType(value) { this._type = value;}

COVISE_ModuleDelEvent.prototype.setType = COVISE_ModuleDelEvent_setType;
//
// accessor is COVISE_ModuleDelEvent.prototype.getModuleID
// element get for moduleID
// - element type is {http://www.w3.org/2001/XMLSchema}string
// - required element
//
// element set for moduleID
// setter function is is COVISE_ModuleDelEvent.prototype.setModuleID
//
function COVISE_ModuleDelEvent_getModuleID() { return this._moduleID;}

COVISE_ModuleDelEvent.prototype.getModuleID = COVISE_ModuleDelEvent_getModuleID;

function COVISE_ModuleDelEvent_setModuleID(value) { this._moduleID = value;}

COVISE_ModuleDelEvent.prototype.setModuleID = COVISE_ModuleDelEvent_setModuleID;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}ModuleDelEvent
//
function COVISE_ModuleDelEvent_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<type>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._type);
     xml = xml + '</type>';
    }
    // block for local variables
    {
     xml = xml + '<moduleID>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._moduleID);
     xml = xml + '</moduleID>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_ModuleDelEvent.prototype.serialize = COVISE_ModuleDelEvent_serialize;

function COVISE_ModuleDelEvent_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_ModuleDelEvent();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing type');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setType(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing moduleID');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = value;
    }
    newobject.setModuleID(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Constructor for XML Schema item {http://www.hlrs.de/organization/vis/covise}uploadFileMtomResponse
//
function COVISE_uploadFileMtomResponse () {
    this.typeMarker = 'COVISE_uploadFileMtomResponse';
    this._result = '';
    this._lastChunk = '';
}

//
// accessor is COVISE_uploadFileMtomResponse.prototype.getResult
// element get for result
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for result
// setter function is is COVISE_uploadFileMtomResponse.prototype.setResult
//
function COVISE_uploadFileMtomResponse_getResult() { return this._result;}

COVISE_uploadFileMtomResponse.prototype.getResult = COVISE_uploadFileMtomResponse_getResult;

function COVISE_uploadFileMtomResponse_setResult(value) { this._result = value;}

COVISE_uploadFileMtomResponse.prototype.setResult = COVISE_uploadFileMtomResponse_setResult;
//
// accessor is COVISE_uploadFileMtomResponse.prototype.getLastChunk
// element get for lastChunk
// - element type is {http://www.w3.org/2001/XMLSchema}boolean
// - required element
//
// element set for lastChunk
// setter function is is COVISE_uploadFileMtomResponse.prototype.setLastChunk
//
function COVISE_uploadFileMtomResponse_getLastChunk() { return this._lastChunk;}

COVISE_uploadFileMtomResponse.prototype.getLastChunk = COVISE_uploadFileMtomResponse_getLastChunk;

function COVISE_uploadFileMtomResponse_setLastChunk(value) { this._lastChunk = value;}

COVISE_uploadFileMtomResponse.prototype.setLastChunk = COVISE_uploadFileMtomResponse_setLastChunk;
//
// Serialize {http://www.hlrs.de/organization/vis/covise}uploadFileMtomResponse
//
function COVISE_uploadFileMtomResponse_serialize(cxfjsutils, elementName, extraNamespaces) {
    var xml = '';
    if (elementName != null) {
     xml = xml + '<';
     xml = xml + elementName;
     xml = xml + ' ';
     xml = xml + 'xmlns:jns0=\'http://www.w3.org/2004/08/xop/include\' xmlns:jns1=\'http://www.w3.org/2005/05/xmlmime\' ';
     if (extraNamespaces) {
      xml = xml + ' ' + extraNamespaces;
     }
     xml = xml + '>';
    }
    // block for local variables
    {
     xml = xml + '<result>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._result);
     xml = xml + '</result>';
    }
    // block for local variables
    {
     xml = xml + '<lastChunk>';
     xml = xml + cxfjsutils.escapeXmlEntities(this._lastChunk);
     xml = xml + '</lastChunk>';
    }
    if (elementName != null) {
     xml = xml + '</';
     xml = xml + elementName;
     xml = xml + '>';
    }
    return xml;
}

COVISE_uploadFileMtomResponse.prototype.serialize = COVISE_uploadFileMtomResponse_serialize;

function COVISE_uploadFileMtomResponse_deserialize (cxfjsutils, element) {
    var newobject = new COVISE_uploadFileMtomResponse();
    cxfjsutils.trace('element: ' + cxfjsutils.traceElementName(element));
    var curElement = cxfjsutils.getFirstElementChild(element);
    var item;
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing result');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setResult(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    cxfjsutils.trace('curElement: ' + cxfjsutils.traceElementName(curElement));
    cxfjsutils.trace('processing lastChunk');
    var value = null;
    if (!cxfjsutils.isElementNil(curElement)) {
     value = cxfjsutils.getNodeText(curElement);
     item = (value == 'true');
    }
    newobject.setLastChunk(item);
    var item = null;
    if (curElement != null) {
     curElement = cxfjsutils.getNextElementSibling(curElement);
    }
    return newobject;
}

//
// Definitions for schema: http://www.w3.org/2005/05/xmlmime
//  http://www.w3.org/2005/05/xmlmime
//
//
// Definitions for service: {http://www.hlrs.de/organization/vis/covise}COVISE
//

// Javascript for {http://www.hlrs.de/organization/vis/covise}ServiceSoap

function COVISE_ServiceSoap () {
    this.jsutils = new CxfApacheOrgUtil();
    this.jsutils.interfaceObject = this;
    this.synchronous = false;
    this.url = null;
    this.client = null;
    this.response = null;
    this.globalElementSerializers = [];
    this.globalElementDeserializers = [];
    this.globalElementSerializers['{http://www.w3.org/2004/08/xop/include}Include'] = XOP_Include_serialize;
    this.globalElementDeserializers['{http://www.w3.org/2004/08/xop/include}Include'] = XOP_Include_deserialize;
    this.globalElementSerializers['{http://www.w3.org/2004/08/xop/include}Include'] = XOP_Include_serialize;
    this.globalElementDeserializers['{http://www.w3.org/2004/08/xop/include}Include'] = XOP_Include_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterResponse'] = COVISE_setParameterResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterResponse'] = COVISE_setParameterResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFileResponse'] = COVISE_setParameterFromUploadedFileResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFileResponse'] = COVISE_setParameterFromUploadedFileResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}addPartner'] = COVISE_addPartner_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}addPartner'] = COVISE_addPartner_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}executeModule'] = COVISE_executeModule_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}executeModule'] = COVISE_executeModule_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}isDirExist'] = COVISE_isDirExist_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}isDirExist'] = COVISE_isDirExist_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getEventResponse'] = COVISE_getEventResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getEventResponse'] = COVISE_getEventResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}createNewDir'] = COVISE_createNewDir_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}createNewDir'] = COVISE_createNewDir_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}isFileExistResponse'] = COVISE_isFileExistResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}isFileExistResponse'] = COVISE_isFileExistResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getModuleID'] = COVISE_getModuleID_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getModuleID'] = COVISE_getModuleID_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getParameterAsString'] = COVISE_getParameterAsString_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getParameterAsString'] = COVISE_getParameterAsString_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}instantiateModuleResponse'] = COVISE_instantiateModuleResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}instantiateModuleResponse'] = COVISE_instantiateModuleResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}deleteDirResponse'] = COVISE_deleteDirResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}deleteDirResponse'] = COVISE_deleteDirResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}unlink'] = COVISE_unlink_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}unlink'] = COVISE_unlink_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}unlinkResponse'] = COVISE_unlinkResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}unlinkResponse'] = COVISE_unlinkResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getRunningModule'] = COVISE_getRunningModule_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getRunningModule'] = COVISE_getRunningModule_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getFileInfoList'] = COVISE_getFileInfoList_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getFileInfoList'] = COVISE_getFileInfoList_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getParameterAsStringResponse'] = COVISE_getParameterAsStringResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getParameterAsStringResponse'] = COVISE_getParameterAsStringResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}executeNet'] = COVISE_executeNet_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}executeNet'] = COVISE_executeNet_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}executeNetResponse'] = COVISE_executeNetResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}executeNetResponse'] = COVISE_executeNetResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}executeModuleResponse'] = COVISE_executeModuleResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}executeModuleResponse'] = COVISE_executeModuleResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getRunningModules'] = COVISE_getRunningModules_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getRunningModules'] = COVISE_getRunningModules_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getConfigEntryResponse'] = COVISE_getConfigEntryResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getConfigEntryResponse'] = COVISE_getConfigEntryResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getRunningModulesResponse'] = COVISE_getRunningModulesResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getRunningModulesResponse'] = COVISE_getRunningModulesResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}isFileExist'] = COVISE_isFileExist_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}isFileExist'] = COVISE_isFileExist_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getModuleIDResponse'] = COVISE_getModuleIDResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getModuleIDResponse'] = COVISE_getModuleIDResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getLinksResponse'] = COVISE_getLinksResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getLinksResponse'] = COVISE_getLinksResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}quitResponse'] = COVISE_quitResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}quitResponse'] = COVISE_quitResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}uploadFileResponse'] = COVISE_uploadFileResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}uploadFileResponse'] = COVISE_uploadFileResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}uploadFile'] = COVISE_uploadFile_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}uploadFile'] = COVISE_uploadFile_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}deleteModule'] = COVISE_deleteModule_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}deleteModule'] = COVISE_deleteModule_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getLinks'] = COVISE_getLinks_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getLinks'] = COVISE_getLinks_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}openNetResponse'] = COVISE_openNetResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}openNetResponse'] = COVISE_openNetResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromStringResponse'] = COVISE_setParameterFromStringResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromStringResponse'] = COVISE_setParameterFromStringResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}createNewDirResponse'] = COVISE_createNewDirResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}createNewDirResponse'] = COVISE_createNewDirResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}listModules'] = COVISE_listModules_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}listModules'] = COVISE_listModules_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getConfigEntry'] = COVISE_getConfigEntry_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getConfigEntry'] = COVISE_getConfigEntry_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}listHostsResponse'] = COVISE_listHostsResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}listHostsResponse'] = COVISE_listHostsResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}linkResponse'] = COVISE_linkResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}linkResponse'] = COVISE_linkResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}deleteDir'] = COVISE_deleteDir_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}deleteDir'] = COVISE_deleteDir_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameter'] = COVISE_setParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameter'] = COVISE_setParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}addPartnerResponse'] = COVISE_addPartnerResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}addPartnerResponse'] = COVISE_addPartnerResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}isDirExistResponse'] = COVISE_isDirExistResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}isDirExistResponse'] = COVISE_isDirExistResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}quit'] = COVISE_quit_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}quit'] = COVISE_quit_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getEvent'] = COVISE_getEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getEvent'] = COVISE_getEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}uploadFileMtom'] = COVISE_uploadFileMtom_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}uploadFileMtom'] = COVISE_uploadFileMtom_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}addEventListenerResponse'] = COVISE_addEventListenerResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}addEventListenerResponse'] = COVISE_addEventListenerResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}removeEventListener'] = COVISE_removeEventListener_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}removeEventListener'] = COVISE_removeEventListener_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}instantiateModule'] = COVISE_instantiateModule_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}instantiateModule'] = COVISE_instantiateModule_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getFileInfoListResponse'] = COVISE_getFileInfoListResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getFileInfoListResponse'] = COVISE_getFileInfoListResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}listHosts'] = COVISE_listHosts_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}listHosts'] = COVISE_listHosts_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}deleteModuleResponse'] = COVISE_deleteModuleResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}deleteModuleResponse'] = COVISE_deleteModuleResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}addEventListener'] = COVISE_addEventListener_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}addEventListener'] = COVISE_addEventListener_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}openNet'] = COVISE_openNet_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}openNet'] = COVISE_openNet_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFile'] = COVISE_setParameterFromUploadedFile_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFile'] = COVISE_setParameterFromUploadedFile_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}link'] = COVISE_link_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}link'] = COVISE_link_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}removeEventListenerResponse'] = COVISE_removeEventListenerResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}removeEventListenerResponse'] = COVISE_removeEventListenerResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}listModulesResponse'] = COVISE_listModulesResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}listModulesResponse'] = COVISE_listModulesResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getRunningModuleResponse'] = COVISE_getRunningModuleResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getRunningModuleResponse'] = COVISE_getRunningModuleResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromString'] = COVISE_setParameterFromString_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromString'] = COVISE_setParameterFromString_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}uploadFileMtomResponse'] = COVISE_uploadFileMtomResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}uploadFileMtomResponse'] = COVISE_uploadFileMtomResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}OpenNetDoneEvent'] = COVISE_OpenNetDoneEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}OpenNetDoneEvent'] = COVISE_OpenNetDoneEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}StringPair'] = COVISE_StringPair_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}StringPair'] = COVISE_StringPair_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}Event'] = COVISE_Event_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}Event'] = COVISE_Event_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}FileData'] = COVISE_FileData_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}FileData'] = COVISE_FileData_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterResponse'] = COVISE_setParameterResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterResponse'] = COVISE_setParameterResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ModuleExecuteStartEvent'] = COVISE_ModuleExecuteStartEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ModuleExecuteStartEvent'] = COVISE_ModuleExecuteStartEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ModuleAddEvent'] = COVISE_ModuleAddEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ModuleAddEvent'] = COVISE_ModuleAddEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ModuleChangeEvent'] = COVISE_ModuleChangeEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ModuleChangeEvent'] = COVISE_ModuleChangeEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}QuitEvent'] = COVISE_QuitEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}QuitEvent'] = COVISE_QuitEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getEventResponse'] = COVISE_getEventResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getEventResponse'] = COVISE_getEventResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}createNewDir'] = COVISE_createNewDir_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}createNewDir'] = COVISE_createNewDir_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}LinkAddEvent'] = COVISE_LinkAddEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}LinkAddEvent'] = COVISE_LinkAddEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getParameterAsString'] = COVISE_getParameterAsString_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getParameterAsString'] = COVISE_getParameterAsString_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getModuleID'] = COVISE_getModuleID_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getModuleID'] = COVISE_getModuleID_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}isFileExistResponse'] = COVISE_isFileExistResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}isFileExistResponse'] = COVISE_isFileExistResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}instantiateModuleResponse'] = COVISE_instantiateModuleResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}instantiateModuleResponse'] = COVISE_instantiateModuleResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}FloatSliderParameter'] = COVISE_FloatSliderParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}FloatSliderParameter'] = COVISE_FloatSliderParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ExecuteStartEvent'] = COVISE_ExecuteStartEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ExecuteStartEvent'] = COVISE_ExecuteStartEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ParameterChangeEvent'] = COVISE_ParameterChangeEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ParameterChangeEvent'] = COVISE_ParameterChangeEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getParameterAsStringResponse'] = COVISE_getParameterAsStringResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getParameterAsStringResponse'] = COVISE_getParameterAsStringResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getFileInfoList'] = COVISE_getFileInfoList_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getFileInfoList'] = COVISE_getFileInfoList_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}executeModuleResponse'] = COVISE_executeModuleResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}executeModuleResponse'] = COVISE_executeModuleResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getRunningModules'] = COVISE_getRunningModules_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getRunningModules'] = COVISE_getRunningModules_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getRunningModulesResponse'] = COVISE_getRunningModulesResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getRunningModulesResponse'] = COVISE_getRunningModulesResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getModuleIDResponse'] = COVISE_getModuleIDResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getModuleIDResponse'] = COVISE_getModuleIDResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}isFileExist'] = COVISE_isFileExist_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}isFileExist'] = COVISE_isFileExist_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getLinksResponse'] = COVISE_getLinksResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getLinksResponse'] = COVISE_getLinksResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}quitResponse'] = COVISE_quitResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}quitResponse'] = COVISE_quitResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}uploadFile'] = COVISE_uploadFile_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}uploadFile'] = COVISE_uploadFile_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}uploadFileResponse'] = COVISE_uploadFileResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}uploadFileResponse'] = COVISE_uploadFileResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getLinks'] = COVISE_getLinks_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getLinks'] = COVISE_getLinks_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}createNewDirResponse'] = COVISE_createNewDirResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}createNewDirResponse'] = COVISE_createNewDirResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}listModules'] = COVISE_listModules_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}listModules'] = COVISE_listModules_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ModuleDiedEvent'] = COVISE_ModuleDiedEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ModuleDiedEvent'] = COVISE_ModuleDiedEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}deleteDir'] = COVISE_deleteDir_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}deleteDir'] = COVISE_deleteDir_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}addPartnerResponse'] = COVISE_addPartnerResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}addPartnerResponse'] = COVISE_addPartnerResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameter'] = COVISE_setParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameter'] = COVISE_setParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}FloatVectorParameter'] = COVISE_FloatVectorParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}FloatVectorParameter'] = COVISE_FloatVectorParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}IntSliderParameter'] = COVISE_IntSliderParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}IntSliderParameter'] = COVISE_IntSliderParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}addEventListenerResponse'] = COVISE_addEventListenerResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}addEventListenerResponse'] = COVISE_addEventListenerResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getEvent'] = COVISE_getEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getEvent'] = COVISE_getEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}instantiateModule'] = COVISE_instantiateModule_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}instantiateModule'] = COVISE_instantiateModule_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}listHosts'] = COVISE_listHosts_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}listHosts'] = COVISE_listHosts_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getFileInfoListResponse'] = COVISE_getFileInfoListResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getFileInfoListResponse'] = COVISE_getFileInfoListResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}LinkDelEvent'] = COVISE_LinkDelEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}LinkDelEvent'] = COVISE_LinkDelEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}Parameter'] = COVISE_Parameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}Parameter'] = COVISE_Parameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getRunningModuleResponse'] = COVISE_getRunningModuleResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getRunningModuleResponse'] = COVISE_getRunningModuleResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFileResponse'] = COVISE_setParameterFromUploadedFileResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFileResponse'] = COVISE_setParameterFromUploadedFileResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}Colormap'] = COVISE_Colormap_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}Colormap'] = COVISE_Colormap_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}FileBrowserParameter'] = COVISE_FileBrowserParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}FileBrowserParameter'] = COVISE_FileBrowserParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}addPartner'] = COVISE_addPartner_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}addPartner'] = COVISE_addPartner_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}executeModule'] = COVISE_executeModule_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}executeModule'] = COVISE_executeModule_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}isDirExist'] = COVISE_isDirExist_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}isDirExist'] = COVISE_isDirExist_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}Port'] = COVISE_Port_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}Port'] = COVISE_Port_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}Module'] = COVISE_Module_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}Module'] = COVISE_Module_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}deleteDirResponse'] = COVISE_deleteDirResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}deleteDirResponse'] = COVISE_deleteDirResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}unlink'] = COVISE_unlink_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}unlink'] = COVISE_unlink_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}unlinkResponse'] = COVISE_unlinkResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}unlinkResponse'] = COVISE_unlinkResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}OpenNetEvent'] = COVISE_OpenNetEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}OpenNetEvent'] = COVISE_OpenNetEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getRunningModule'] = COVISE_getRunningModule_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getRunningModule'] = COVISE_getRunningModule_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ExecuteFinishEvent'] = COVISE_ExecuteFinishEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ExecuteFinishEvent'] = COVISE_ExecuteFinishEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}executeNet'] = COVISE_executeNet_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}executeNet'] = COVISE_executeNet_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}executeNetResponse'] = COVISE_executeNetResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}executeNetResponse'] = COVISE_executeNetResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ModuleExecuteFinishEvent'] = COVISE_ModuleExecuteFinishEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ModuleExecuteFinishEvent'] = COVISE_ModuleExecuteFinishEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ColormapPin'] = COVISE_ColormapPin_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ColormapPin'] = COVISE_ColormapPin_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getConfigEntryResponse'] = COVISE_getConfigEntryResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getConfigEntryResponse'] = COVISE_getConfigEntryResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}Point'] = COVISE_Point_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}Point'] = COVISE_Point_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ChoiceParameter'] = COVISE_ChoiceParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ChoiceParameter'] = COVISE_ChoiceParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}deleteModule'] = COVISE_deleteModule_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}deleteModule'] = COVISE_deleteModule_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}openNetResponse'] = COVISE_openNetResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}openNetResponse'] = COVISE_openNetResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromStringResponse'] = COVISE_setParameterFromStringResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromStringResponse'] = COVISE_setParameterFromStringResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}StringParameter'] = COVISE_StringParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}StringParameter'] = COVISE_StringParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}listHostsResponse'] = COVISE_listHostsResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}listHostsResponse'] = COVISE_listHostsResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}getConfigEntry'] = COVISE_getConfigEntry_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}getConfigEntry'] = COVISE_getConfigEntry_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}IntScalarParameter'] = COVISE_IntScalarParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}IntScalarParameter'] = COVISE_IntScalarParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}linkResponse'] = COVISE_linkResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}linkResponse'] = COVISE_linkResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}IntVectorParameter'] = COVISE_IntVectorParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}IntVectorParameter'] = COVISE_IntVectorParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}isDirExistResponse'] = COVISE_isDirExistResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}isDirExistResponse'] = COVISE_isDirExistResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}Link'] = COVISE_Link_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}Link'] = COVISE_Link_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}quit'] = COVISE_quit_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}quit'] = COVISE_quit_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}uploadFileMtom'] = COVISE_uploadFileMtom_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}uploadFileMtom'] = COVISE_uploadFileMtom_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}removeEventListener'] = COVISE_removeEventListener_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}removeEventListener'] = COVISE_removeEventListener_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}deleteModuleResponse'] = COVISE_deleteModuleResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}deleteModuleResponse'] = COVISE_deleteModuleResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}addEventListener'] = COVISE_addEventListener_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}addEventListener'] = COVISE_addEventListener_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}openNet'] = COVISE_openNet_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}openNet'] = COVISE_openNet_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}FloatScalarParameter'] = COVISE_FloatScalarParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}FloatScalarParameter'] = COVISE_FloatScalarParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFile'] = COVISE_setParameterFromUploadedFile_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFile'] = COVISE_setParameterFromUploadedFile_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}link'] = COVISE_link_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}link'] = COVISE_link_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}BooleanParameter'] = COVISE_BooleanParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}BooleanParameter'] = COVISE_BooleanParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}removeEventListenerResponse'] = COVISE_removeEventListenerResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}removeEventListenerResponse'] = COVISE_removeEventListenerResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}listModulesResponse'] = COVISE_listModulesResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}listModulesResponse'] = COVISE_listModulesResponse_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromString'] = COVISE_setParameterFromString_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}setParameterFromString'] = COVISE_setParameterFromString_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ColormapChoiceParameter'] = COVISE_ColormapChoiceParameter_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ColormapChoiceParameter'] = COVISE_ColormapChoiceParameter_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}FileInfo'] = COVISE_FileInfo_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}FileInfo'] = COVISE_FileInfo_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}ModuleDelEvent'] = COVISE_ModuleDelEvent_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}ModuleDelEvent'] = COVISE_ModuleDelEvent_deserialize;
    this.globalElementSerializers['{http://www.hlrs.de/organization/vis/covise}uploadFileMtomResponse'] = COVISE_uploadFileMtomResponse_serialize;
    this.globalElementDeserializers['{http://www.hlrs.de/organization/vis/covise}uploadFileMtomResponse'] = COVISE_uploadFileMtomResponse_deserialize;
}

function COVISE_setParameterFromUploadedFile_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_setParameterFromUploadedFileResponse_deserializeResponse');
     responseObject = COVISE_setParameterFromUploadedFileResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.setParameterFromUploadedFile_onsuccess = COVISE_setParameterFromUploadedFile_op_onsuccess;

function COVISE_setParameterFromUploadedFile_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.setParameterFromUploadedFile_onerror = COVISE_setParameterFromUploadedFile_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}setParameterFromUploadedFile
// Wrapped operation.
// parameter moduleID
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter parameter
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter value
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_setParameterFromUploadedFile_op(successCallback, errorCallback, moduleID, parameter, value) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(3);
    args[0] = moduleID;
    args[1] = parameter;
    args[2] = value;
    xml = this.setParameterFromUploadedFile_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.setParameterFromUploadedFile_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.setParameterFromUploadedFile_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/setParameterFromUploadedFile';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.setParameterFromUploadedFile = COVISE_setParameterFromUploadedFile_op;

function COVISE_setParameterFromUploadedFile_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_setParameterFromUploadedFile();
    wrapperObj.setModuleID(args[0]);
    wrapperObj.setParameter(args[1]);
    wrapperObj.setValue(args[2]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:setParameterFromUploadedFile', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.setParameterFromUploadedFile_serializeInput = COVISE_setParameterFromUploadedFile_serializeInput;

function COVISE_setParameterFromUploadedFileResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_getLinks_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_getLinksResponse_deserializeResponse');
     responseObject = COVISE_getLinksResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.getLinks_onsuccess = COVISE_getLinks_op_onsuccess;

function COVISE_getLinks_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.getLinks_onerror = COVISE_getLinks_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}getLinks
// Wrapped operation.
//
function COVISE_getLinks_op(successCallback, errorCallback) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(0);
    xml = this.getLinks_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.getLinks_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.getLinks_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/getLinks';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.getLinks = COVISE_getLinks_op;

function COVISE_getLinks_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_getLinks();
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:getLinks', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.getLinks_serializeInput = COVISE_getLinks_serializeInput;

function COVISE_getLinksResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_getLinksResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_removeEventListener_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_removeEventListenerResponse_deserializeResponse');
     responseObject = COVISE_removeEventListenerResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.removeEventListener_onsuccess = COVISE_removeEventListener_op_onsuccess;

function COVISE_removeEventListener_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.removeEventListener_onerror = COVISE_removeEventListener_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}removeEventListener
// Wrapped operation.
// parameter uuid
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_removeEventListener_op(successCallback, errorCallback, uuid) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = uuid;
    xml = this.removeEventListener_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.removeEventListener_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.removeEventListener_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/removeEventListener';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.removeEventListener = COVISE_removeEventListener_op;

function COVISE_removeEventListener_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_removeEventListener();
    wrapperObj.setUuid(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:removeEventListener', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.removeEventListener_serializeInput = COVISE_removeEventListener_serializeInput;

function COVISE_removeEventListenerResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_addEventListener_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_addEventListenerResponse_deserializeResponse');
     responseObject = COVISE_addEventListenerResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.addEventListener_onsuccess = COVISE_addEventListener_op_onsuccess;

function COVISE_addEventListener_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.addEventListener_onerror = COVISE_addEventListener_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}addEventListener
// Wrapped operation.
//
function COVISE_addEventListener_op(successCallback, errorCallback) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(0);
    xml = this.addEventListener_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.addEventListener_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.addEventListener_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/addEventListener';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.addEventListener = COVISE_addEventListener_op;

function COVISE_addEventListener_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_addEventListener();
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:addEventListener', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.addEventListener_serializeInput = COVISE_addEventListener_serializeInput;

function COVISE_addEventListenerResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_addEventListenerResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_getModuleID_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_getModuleIDResponse_deserializeResponse');
     responseObject = COVISE_getModuleIDResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.getModuleID_onsuccess = COVISE_getModuleID_op_onsuccess;

function COVISE_getModuleID_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.getModuleID_onerror = COVISE_getModuleID_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}getModuleID
// Wrapped operation.
// parameter module
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter instance
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter host
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_getModuleID_op(successCallback, errorCallback, module, instance, host) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(3);
    args[0] = module;
    args[1] = instance;
    args[2] = host;
    xml = this.getModuleID_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.getModuleID_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.getModuleID_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/getModuleID';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.getModuleID = COVISE_getModuleID_op;

function COVISE_getModuleID_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_getModuleID();
    wrapperObj.setModule(args[0]);
    wrapperObj.setInstance(args[1]);
    wrapperObj.setHost(args[2]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:getModuleID', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.getModuleID_serializeInput = COVISE_getModuleID_serializeInput;

function COVISE_getModuleIDResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_getModuleIDResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_createNewDir_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_createNewDirResponse_deserializeResponse');
     responseObject = COVISE_createNewDirResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.createNewDir_onsuccess = COVISE_createNewDir_op_onsuccess;

function COVISE_createNewDir_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.createNewDir_onerror = COVISE_createNewDir_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}createNewDir
// Wrapped operation.
// parameter path
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter newDir
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_createNewDir_op(successCallback, errorCallback, path, newDir) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(2);
    args[0] = path;
    args[1] = newDir;
    xml = this.createNewDir_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.createNewDir_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.createNewDir_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/createNewDir';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.createNewDir = COVISE_createNewDir_op;

function COVISE_createNewDir_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_createNewDir();
    wrapperObj.setPath(args[0]);
    wrapperObj.setNewDir(args[1]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:createNewDir', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.createNewDir_serializeInput = COVISE_createNewDir_serializeInput;

function COVISE_createNewDirResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_createNewDirResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_unlink_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_unlinkResponse_deserializeResponse');
     responseObject = COVISE_unlinkResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.unlink_onsuccess = COVISE_unlink_op_onsuccess;

function COVISE_unlink_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.unlink_onerror = COVISE_unlink_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}unlink
// Wrapped operation.
// parameter linkID
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_unlink_op(successCallback, errorCallback, linkID) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = linkID;
    xml = this.unlink_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.unlink_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.unlink_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/unlink';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.unlink = COVISE_unlink_op;

function COVISE_unlink_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_unlink();
    wrapperObj.setLinkID(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:unlink', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.unlink_serializeInput = COVISE_unlink_serializeInput;

function COVISE_unlinkResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_deleteDir_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_deleteDirResponse_deserializeResponse');
     responseObject = COVISE_deleteDirResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.deleteDir_onsuccess = COVISE_deleteDir_op_onsuccess;

function COVISE_deleteDir_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.deleteDir_onerror = COVISE_deleteDir_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}deleteDir
// Wrapped operation.
// parameter path
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_deleteDir_op(successCallback, errorCallback, path) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = path;
    xml = this.deleteDir_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.deleteDir_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.deleteDir_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/deleteDir';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.deleteDir = COVISE_deleteDir_op;

function COVISE_deleteDir_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_deleteDir();
    wrapperObj.setPath(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:deleteDir', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.deleteDir_serializeInput = COVISE_deleteDir_serializeInput;

function COVISE_deleteDirResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_deleteDirResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_setParameterFromString_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_setParameterFromStringResponse_deserializeResponse');
     responseObject = COVISE_setParameterFromStringResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.setParameterFromString_onsuccess = COVISE_setParameterFromString_op_onsuccess;

function COVISE_setParameterFromString_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.setParameterFromString_onerror = COVISE_setParameterFromString_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}setParameterFromString
// Wrapped operation.
// parameter moduleID
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter parameter
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter value
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_setParameterFromString_op(successCallback, errorCallback, moduleID, parameter, value) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(3);
    args[0] = moduleID;
    args[1] = parameter;
    args[2] = value;
    xml = this.setParameterFromString_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.setParameterFromString_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.setParameterFromString_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/setParameterFromString';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.setParameterFromString = COVISE_setParameterFromString_op;

function COVISE_setParameterFromString_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_setParameterFromString();
    wrapperObj.setModuleID(args[0]);
    wrapperObj.setParameter(args[1]);
    wrapperObj.setValue(args[2]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:setParameterFromString', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.setParameterFromString_serializeInput = COVISE_setParameterFromString_serializeInput;

function COVISE_setParameterFromStringResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_link_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_linkResponse_deserializeResponse');
     responseObject = COVISE_linkResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.link_onsuccess = COVISE_link_op_onsuccess;

function COVISE_link_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.link_onerror = COVISE_link_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}link
// Wrapped operation.
// parameter fromModule
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter fromPort
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter toModule
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter toPort
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_link_op(successCallback, errorCallback, fromModule, fromPort, toModule, toPort) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(4);
    args[0] = fromModule;
    args[1] = fromPort;
    args[2] = toModule;
    args[3] = toPort;
    xml = this.link_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.link_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.link_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/link';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.link = COVISE_link_op;

function COVISE_link_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_link();
    wrapperObj.setFromModule(args[0]);
    wrapperObj.setFromPort(args[1]);
    wrapperObj.setToModule(args[2]);
    wrapperObj.setToPort(args[3]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:link', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.link_serializeInput = COVISE_link_serializeInput;

function COVISE_linkResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_executeNet_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_executeNetResponse_deserializeResponse');
     responseObject = COVISE_executeNetResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.executeNet_onsuccess = COVISE_executeNet_op_onsuccess;

function COVISE_executeNet_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.executeNet_onerror = COVISE_executeNet_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}executeNet
// Wrapped operation.
//
function COVISE_executeNet_op(successCallback, errorCallback) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(0);
    xml = this.executeNet_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.executeNet_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.executeNet_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/executeNet';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.executeNet = COVISE_executeNet_op;

function COVISE_executeNet_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_executeNet();
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:executeNet', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.executeNet_serializeInput = COVISE_executeNet_serializeInput;

function COVISE_executeNetResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_getFileInfoList_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_getFileInfoListResponse_deserializeResponse');
     responseObject = COVISE_getFileInfoListResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.getFileInfoList_onsuccess = COVISE_getFileInfoList_op_onsuccess;

function COVISE_getFileInfoList_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.getFileInfoList_onerror = COVISE_getFileInfoList_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}getFileInfoList
// Wrapped operation.
// parameter path
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_getFileInfoList_op(successCallback, errorCallback, path) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = path;
    xml = this.getFileInfoList_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.getFileInfoList_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.getFileInfoList_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/getFileInfoList';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.getFileInfoList = COVISE_getFileInfoList_op;

function COVISE_getFileInfoList_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_getFileInfoList();
    wrapperObj.setPath(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:getFileInfoList', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.getFileInfoList_serializeInput = COVISE_getFileInfoList_serializeInput;

function COVISE_getFileInfoListResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_getFileInfoListResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_addPartner_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_addPartnerResponse_deserializeResponse');
     responseObject = COVISE_addPartnerResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.addPartner_onsuccess = COVISE_addPartner_op_onsuccess;

function COVISE_addPartner_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.addPartner_onerror = COVISE_addPartner_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}addPartner
// Wrapped operation.
// parameter method
// - simple type {http://www.hlrs.de/organization/vis/covise}AddPartnerMethod// parameter ip
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter user
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter password
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter timeout
// - simple type {http://www.w3.org/2001/XMLSchema}int// parameter display
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_addPartner_op(successCallback, errorCallback, method, ip, user, password, timeout, display) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(6);
    args[0] = method;
    args[1] = ip;
    args[2] = user;
    args[3] = password;
    args[4] = timeout;
    args[5] = display;
    xml = this.addPartner_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.addPartner_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.addPartner_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/addPartner';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.addPartner = COVISE_addPartner_op;

function COVISE_addPartner_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_addPartner();
    wrapperObj.setMethod(args[0]);
    wrapperObj.setIp(args[1]);
    wrapperObj.setUser(args[2]);
    wrapperObj.setPassword(args[3]);
    wrapperObj.setTimeout(args[4]);
    wrapperObj.setDisplay(args[5]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:addPartner', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.addPartner_serializeInput = COVISE_addPartner_serializeInput;

function COVISE_addPartnerResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_addPartnerResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_listHosts_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_listHostsResponse_deserializeResponse');
     responseObject = COVISE_listHostsResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.listHosts_onsuccess = COVISE_listHosts_op_onsuccess;

function COVISE_listHosts_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.listHosts_onerror = COVISE_listHosts_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}listHosts
// Wrapped operation.
//
function COVISE_listHosts_op(successCallback, errorCallback) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(0);
    xml = this.listHosts_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.listHosts_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.listHosts_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/listHosts';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.listHosts = COVISE_listHosts_op;

function COVISE_listHosts_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_listHosts();
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:listHosts', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.listHosts_serializeInput = COVISE_listHosts_serializeInput;

function COVISE_listHostsResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_listHostsResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_uploadFileMtom_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_uploadFileMtomResponse_deserializeResponse');
     responseObject = COVISE_uploadFileMtomResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.uploadFileMtom_onsuccess = COVISE_uploadFileMtom_op_onsuccess;

function COVISE_uploadFileMtom_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.uploadFileMtom_onerror = COVISE_uploadFileMtom_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}uploadFileMtom
// Wrapped operation.
// parameter path
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter fileName
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter fileData
// - simple type {http://www.w3.org/2001/XMLSchema}base64Binary// parameter fileTrunc
// - simple type {http://www.w3.org/2001/XMLSchema}boolean// parameter fileSize
// - simple type {http://www.w3.org/2001/XMLSchema}int//
function COVISE_uploadFileMtom_op(successCallback, errorCallback, path, fileName, fileData, fileTrunc, fileSize) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(5);
    args[0] = path;
    args[1] = fileName;
    args[2] = fileData;
    args[3] = fileTrunc;
    args[4] = fileSize;
    xml = this.uploadFileMtom_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.uploadFileMtom_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.uploadFileMtom_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/uploadFileMtom';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.uploadFileMtom = COVISE_uploadFileMtom_op;

function COVISE_uploadFileMtom_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_uploadFileMtom();
    wrapperObj.setPath(args[0]);
    wrapperObj.setFileName(args[1]);
    wrapperObj.setFileData(args[2]);
    wrapperObj.setFileTrunc(args[3]);
    wrapperObj.setFileSize(args[4]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:uploadFileMtom', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.uploadFileMtom_serializeInput = COVISE_uploadFileMtom_serializeInput;

function COVISE_uploadFileMtomResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_uploadFileMtomResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_listModules_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_listModulesResponse_deserializeResponse');
     responseObject = COVISE_listModulesResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.listModules_onsuccess = COVISE_listModules_op_onsuccess;

function COVISE_listModules_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.listModules_onerror = COVISE_listModules_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}listModules
// Wrapped operation.
// parameter ipaddr
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_listModules_op(successCallback, errorCallback, ipaddr) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = ipaddr;
    xml = this.listModules_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.listModules_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.listModules_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/listModules';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.listModules = COVISE_listModules_op;

function COVISE_listModules_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_listModules();
    wrapperObj.setIpaddr(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:listModules', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.listModules_serializeInput = COVISE_listModules_serializeInput;

function COVISE_listModulesResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_listModulesResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_getEvent_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_getEventResponse_deserializeResponse');
     responseObject = COVISE_getEventResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.getEvent_onsuccess = COVISE_getEvent_op_onsuccess;

function COVISE_getEvent_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.getEvent_onerror = COVISE_getEvent_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}getEvent
// Wrapped operation.
// parameter uuid
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter timeout
// - simple type {http://www.w3.org/2001/XMLSchema}int//
function COVISE_getEvent_op(successCallback, errorCallback, uuid, timeout) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(2);
    args[0] = uuid;
    args[1] = timeout;
    xml = this.getEvent_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.getEvent_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.getEvent_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/getEvent';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.getEvent = COVISE_getEvent_op;

function COVISE_getEvent_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_getEvent();
    wrapperObj.setUuid(args[0]);
    wrapperObj.setTimeout(args[1]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:getEvent', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.getEvent_serializeInput = COVISE_getEvent_serializeInput;

function COVISE_getEventResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_getEventResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_instantiateModule_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_instantiateModuleResponse_deserializeResponse');
     responseObject = COVISE_instantiateModuleResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.instantiateModule_onsuccess = COVISE_instantiateModule_op_onsuccess;

function COVISE_instantiateModule_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.instantiateModule_onerror = COVISE_instantiateModule_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}instantiateModule
// Wrapped operation.
// parameter module
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter host
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter x
// - simple type {http://www.w3.org/2001/XMLSchema}int// parameter y
// - simple type {http://www.w3.org/2001/XMLSchema}int//
function COVISE_instantiateModule_op(successCallback, errorCallback, module, host, x, y) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(4);
    args[0] = module;
    args[1] = host;
    args[2] = x;
    args[3] = y;
    xml = this.instantiateModule_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.instantiateModule_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.instantiateModule_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/instantiateModule';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.instantiateModule = COVISE_instantiateModule_op;

function COVISE_instantiateModule_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_instantiateModule();
    wrapperObj.setModule(args[0]);
    wrapperObj.setHost(args[1]);
    wrapperObj.setX(args[2]);
    wrapperObj.setY(args[3]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:instantiateModule', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.instantiateModule_serializeInput = COVISE_instantiateModule_serializeInput;

function COVISE_instantiateModuleResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_getParameterAsString_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_getParameterAsStringResponse_deserializeResponse');
     responseObject = COVISE_getParameterAsStringResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.getParameterAsString_onsuccess = COVISE_getParameterAsString_op_onsuccess;

function COVISE_getParameterAsString_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.getParameterAsString_onerror = COVISE_getParameterAsString_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}getParameterAsString
// Wrapped operation.
// parameter moduleID
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter parameter
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_getParameterAsString_op(successCallback, errorCallback, moduleID, parameter) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(2);
    args[0] = moduleID;
    args[1] = parameter;
    xml = this.getParameterAsString_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.getParameterAsString_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.getParameterAsString_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/getParameterAsString';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.getParameterAsString = COVISE_getParameterAsString_op;

function COVISE_getParameterAsString_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_getParameterAsString();
    wrapperObj.setModuleID(args[0]);
    wrapperObj.setParameter(args[1]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:getParameterAsString', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.getParameterAsString_serializeInput = COVISE_getParameterAsString_serializeInput;

function COVISE_getParameterAsStringResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_getParameterAsStringResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_isFileExist_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_isFileExistResponse_deserializeResponse');
     responseObject = COVISE_isFileExistResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.isFileExist_onsuccess = COVISE_isFileExist_op_onsuccess;

function COVISE_isFileExist_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.isFileExist_onerror = COVISE_isFileExist_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}isFileExist
// Wrapped operation.
// parameter path
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter fileName
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_isFileExist_op(successCallback, errorCallback, path, fileName) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(2);
    args[0] = path;
    args[1] = fileName;
    xml = this.isFileExist_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.isFileExist_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.isFileExist_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/isFileExist';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.isFileExist = COVISE_isFileExist_op;

function COVISE_isFileExist_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_isFileExist();
    wrapperObj.setPath(args[0]);
    wrapperObj.setFileName(args[1]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:isFileExist', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.isFileExist_serializeInput = COVISE_isFileExist_serializeInput;

function COVISE_isFileExistResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_isFileExistResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_getRunningModules_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_getRunningModulesResponse_deserializeResponse');
     responseObject = COVISE_getRunningModulesResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.getRunningModules_onsuccess = COVISE_getRunningModules_op_onsuccess;

function COVISE_getRunningModules_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.getRunningModules_onerror = COVISE_getRunningModules_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}getRunningModules
// Wrapped operation.
//
function COVISE_getRunningModules_op(successCallback, errorCallback) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(0);
    xml = this.getRunningModules_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.getRunningModules_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.getRunningModules_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/getRunningModules';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.getRunningModules = COVISE_getRunningModules_op;

function COVISE_getRunningModules_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_getRunningModules();
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:getRunningModules', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.getRunningModules_serializeInput = COVISE_getRunningModules_serializeInput;

function COVISE_getRunningModulesResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_getRunningModulesResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_quit_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_quitResponse_deserializeResponse');
     responseObject = COVISE_quitResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.quit_onsuccess = COVISE_quit_op_onsuccess;

function COVISE_quit_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.quit_onerror = COVISE_quit_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}quit
// Wrapped operation.
//
function COVISE_quit_op(successCallback, errorCallback) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(0);
    xml = this.quit_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.quit_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.quit_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/quit';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.quit = COVISE_quit_op;

function COVISE_quit_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_quit();
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:quit', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.quit_serializeInput = COVISE_quit_serializeInput;

function COVISE_quitResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_getConfigEntry_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_getConfigEntryResponse_deserializeResponse');
     responseObject = COVISE_getConfigEntryResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.getConfigEntry_onsuccess = COVISE_getConfigEntry_op_onsuccess;

function COVISE_getConfigEntry_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.getConfigEntry_onerror = COVISE_getConfigEntry_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}getConfigEntry
// Wrapped operation.
// parameter section
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter variable
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_getConfigEntry_op(successCallback, errorCallback, section, variable) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(2);
    args[0] = section;
    args[1] = variable;
    xml = this.getConfigEntry_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.getConfigEntry_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.getConfigEntry_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/getConfigEntry';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.getConfigEntry = COVISE_getConfigEntry_op;

function COVISE_getConfigEntry_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_getConfigEntry();
    wrapperObj.setSection(args[0]);
    wrapperObj.setVariable(args[1]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:getConfigEntry', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.getConfigEntry_serializeInput = COVISE_getConfigEntry_serializeInput;

function COVISE_getConfigEntryResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_getConfigEntryResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_executeModule_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_executeModuleResponse_deserializeResponse');
     responseObject = COVISE_executeModuleResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.executeModule_onsuccess = COVISE_executeModule_op_onsuccess;

function COVISE_executeModule_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.executeModule_onerror = COVISE_executeModule_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}executeModule
// Wrapped operation.
// parameter moduleID
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_executeModule_op(successCallback, errorCallback, moduleID) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = moduleID;
    xml = this.executeModule_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.executeModule_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.executeModule_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/executeModule';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.executeModule = COVISE_executeModule_op;

function COVISE_executeModule_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_executeModule();
    wrapperObj.setModuleID(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:executeModule', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.executeModule_serializeInput = COVISE_executeModule_serializeInput;

function COVISE_executeModuleResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_setParameter_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_setParameterResponse_deserializeResponse');
     responseObject = COVISE_setParameterResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.setParameter_onsuccess = COVISE_setParameter_op_onsuccess;

function COVISE_setParameter_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.setParameter_onerror = COVISE_setParameter_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}setParameter
// Wrapped operation.
// parameter moduleID
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter parameter
// - Object constructor is COVISE_Parameter
//
function COVISE_setParameter_op(successCallback, errorCallback, moduleID, parameter) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(2);
    args[0] = moduleID;
    args[1] = parameter;
    xml = this.setParameter_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.setParameter_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.setParameter_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/setParameter';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.setParameter = COVISE_setParameter_op;

function COVISE_setParameter_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_setParameter();
    wrapperObj.setModuleID(args[0]);
    wrapperObj.setParameter(args[1]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:setParameter', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.setParameter_serializeInput = COVISE_setParameter_serializeInput;

function COVISE_setParameterResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_openNet_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_openNetResponse_deserializeResponse');
     responseObject = COVISE_openNetResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.openNet_onsuccess = COVISE_openNet_op_onsuccess;

function COVISE_openNet_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.openNet_onerror = COVISE_openNet_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}openNet
// Wrapped operation.
// parameter filename
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_openNet_op(successCallback, errorCallback, filename) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = filename;
    xml = this.openNet_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.openNet_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.openNet_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/openNet';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.openNet = COVISE_openNet_op;

function COVISE_openNet_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_openNet();
    wrapperObj.setFilename(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:openNet', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.openNet_serializeInput = COVISE_openNet_serializeInput;

function COVISE_openNetResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_uploadFile_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_uploadFileResponse_deserializeResponse');
     responseObject = COVISE_uploadFileResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.uploadFile_onsuccess = COVISE_uploadFile_op_onsuccess;

function COVISE_uploadFile_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.uploadFile_onerror = COVISE_uploadFile_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}uploadFile
// Wrapped operation.
// parameter path
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter fileName
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter resource
// - simple type {http://www.w3.org/2001/XMLSchema}base64Binary// parameter chunkIndex
// - simple type {http://www.w3.org/2001/XMLSchema}int// parameter chunkNr
// - simple type {http://www.w3.org/2001/XMLSchema}int// parameter chunkSize
// - simple type {http://www.w3.org/2001/XMLSchema}int// parameter fileSize
// - simple type {http://www.w3.org/2001/XMLSchema}int// parameter fileTruncated
// - simple type {http://www.w3.org/2001/XMLSchema}boolean//
function COVISE_uploadFile_op(successCallback, errorCallback, path, fileName, resource, chunkIndex, chunkNr, chunkSize, fileSize, fileTruncated) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(8);
    args[0] = path;
    args[1] = fileName;
    args[2] = resource;
    args[3] = chunkIndex;
    args[4] = chunkNr;
    args[5] = chunkSize;
    args[6] = fileSize;
    args[7] = fileTruncated;
    xml = this.uploadFile_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.uploadFile_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.uploadFile_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/uploadFile';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.uploadFile = COVISE_uploadFile_op;

function COVISE_uploadFile_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_uploadFile();
    wrapperObj.setPath(args[0]);
    wrapperObj.setFileName(args[1]);
    wrapperObj.setResource(args[2]);
    wrapperObj.setChunkIndex(args[3]);
    wrapperObj.setChunkNr(args[4]);
    wrapperObj.setChunkSize(args[5]);
    wrapperObj.setFileSize(args[6]);
    wrapperObj.setFileTruncated(args[7]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:uploadFile', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.uploadFile_serializeInput = COVISE_uploadFile_serializeInput;

function COVISE_uploadFileResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_uploadFileResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_deleteModule_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_deleteModuleResponse_deserializeResponse');
     responseObject = COVISE_deleteModuleResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.deleteModule_onsuccess = COVISE_deleteModule_op_onsuccess;

function COVISE_deleteModule_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.deleteModule_onerror = COVISE_deleteModule_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}deleteModule
// Wrapped operation.
// parameter moduleID
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_deleteModule_op(successCallback, errorCallback, moduleID) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = moduleID;
    xml = this.deleteModule_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.deleteModule_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.deleteModule_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/deleteModule';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.deleteModule = COVISE_deleteModule_op;

function COVISE_deleteModule_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_deleteModule();
    wrapperObj.setModuleID(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:deleteModule', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.deleteModule_serializeInput = COVISE_deleteModule_serializeInput;

function COVISE_deleteModuleResponse_deserializeResponse(cxfjsutils, partElement) {
}
function COVISE_isDirExist_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_isDirExistResponse_deserializeResponse');
     responseObject = COVISE_isDirExistResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.isDirExist_onsuccess = COVISE_isDirExist_op_onsuccess;

function COVISE_isDirExist_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.isDirExist_onerror = COVISE_isDirExist_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}isDirExist
// Wrapped operation.
// parameter path
// - simple type {http://www.w3.org/2001/XMLSchema}string// parameter newDir
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_isDirExist_op(successCallback, errorCallback, path, newDir) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(2);
    args[0] = path;
    args[1] = newDir;
    xml = this.isDirExist_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.isDirExist_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.isDirExist_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/isDirExist';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.isDirExist = COVISE_isDirExist_op;

function COVISE_isDirExist_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_isDirExist();
    wrapperObj.setPath(args[0]);
    wrapperObj.setNewDir(args[1]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:isDirExist', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.isDirExist_serializeInput = COVISE_isDirExist_serializeInput;

function COVISE_isDirExistResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_isDirExistResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_getRunningModule_op_onsuccess(client, responseXml) {
    if (client.user_onsuccess) {
     var responseObject = null;
     var element = responseXml.documentElement;
     this.jsutils.trace('responseXml: ' + this.jsutils.traceElementName(element));
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('first element child: ' + this.jsutils.traceElementName(element));
     while (!this.jsutils.isNodeNamedNS(element, 'http://schemas.xmlsoap.org/soap/envelope/', 'Body')) {
      element = this.jsutils.getNextElementSibling(element);
      if (element == null) {
       throw 'No env:Body in message.'
      }
     }
     element = this.jsutils.getFirstElementChild(element);
     this.jsutils.trace('part element: ' + this.jsutils.traceElementName(element));
     this.jsutils.trace('calling COVISE_getRunningModuleResponse_deserializeResponse');
     responseObject = COVISE_getRunningModuleResponse_deserializeResponse(this.jsutils, element);
     client.user_onsuccess(responseObject);
    }
}

COVISE_ServiceSoap.prototype.getRunningModule_onsuccess = COVISE_getRunningModule_op_onsuccess;

function COVISE_getRunningModule_op_onerror(client) {
    if (client.user_onerror) {
     var httpStatus;
     var httpStatusText;
     try {
      httpStatus = client.req.status;
      httpStatusText = client.req.statusText;
     } catch(e) {
      httpStatus = -1;
      httpStatusText = 'Error opening connection to server';
     }
     client.user_onerror(httpStatus, httpStatusText);
    }
}

COVISE_ServiceSoap.prototype.getRunningModule_onerror = COVISE_getRunningModule_op_onerror;

//
// Operation {http://www.hlrs.de/organization/vis/covise}getRunningModule
// Wrapped operation.
// parameter moduleID
// - simple type {http://www.w3.org/2001/XMLSchema}string//
function COVISE_getRunningModule_op(successCallback, errorCallback, moduleID) {
    this.client = new CxfApacheOrgClient(this.jsutils);
    var xml = null;
    var args = new Array(1);
    args[0] = moduleID;
    xml = this.getRunningModule_serializeInput(this.jsutils, args);
    this.client.user_onsuccess = successCallback;
    this.client.user_onerror = errorCallback;
    var closureThis = this;
    this.client.onsuccess = function(client, responseXml) { closureThis.getRunningModule_onsuccess(client, responseXml); };
    this.client.onerror = function(client) { closureThis.getRunningModule_onerror(client); };
    var requestHeaders = [];
    requestHeaders['SOAPAction'] = 'http://www.hlrs.de/organization/vis/covise/getRunningModule';
    this.jsutils.trace('synchronous = ' + this.synchronous);
    this.client.request(this.url, xml, null, this.synchronous, requestHeaders);
}

COVISE_ServiceSoap.prototype.getRunningModule = COVISE_getRunningModule_op;

function COVISE_getRunningModule_serializeInput(cxfjsutils, args) {
    var wrapperObj = new COVISE_getRunningModule();
    wrapperObj.setModuleID(args[0]);
    var xml;
    xml = cxfjsutils.beginSoap11Message("xmlns:jns0='http://www.w3.org/2004/08/xop/include' xmlns:jns1='http://www.w3.org/2005/05/xmlmime' xmlns:jns2='http://www.hlrs.de/organization/vis/covise' ");
    // block for local variables
    {
     xml = xml + wrapperObj.serialize(cxfjsutils, 'jns2:getRunningModule', null);
    }
    xml = xml + cxfjsutils.endSoap11Message();
    return xml;
}

COVISE_ServiceSoap.prototype.getRunningModule_serializeInput = COVISE_getRunningModule_serializeInput;

function COVISE_getRunningModuleResponse_deserializeResponse(cxfjsutils, partElement) {
    var returnObject = COVISE_getRunningModuleResponse_deserialize (cxfjsutils, partElement);

    return returnObject;
}
function COVISE_ServiceSoap_COVISE_COVISE () {
  this.url = 'http://localhost:31111/';
}
COVISE_ServiceSoap_COVISE_COVISE.prototype = new COVISE_ServiceSoap;
