/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

namespace covise
{

#import "stlvector.h"

#import "xop.h"
#import "xmime5.h"

//gsoap covise    schema namespace:	http://www.hlrs.de/organization/vis/covise
//gsoap covise    service style:	rpc
//gsoap covise    service encoding:	encoded

class covise__FileInfo
{
public:
    std::string fileName 1; ///< Required element
    bool isDir 1; ///< Required element
    int fileSize 1; ///< Required element
    std::string fileDate 1; ///< Required element
};

class xsd__base64Binary
{
public:
    unsigned char *__ptr;
    int __size;
};

class covise__FileData
{
public:
    _xop__Include xop__Include;
    @ char *xmime5__contentType;
};

class xsd__anyType
{
    _XML __item;
    struct soap *soap;
};

/// Class wrapper for built-in type "xs:string" derived from xsd__anyType
class xsd__string : public xsd__anyType
{
public:
    std::string __item;
};

// -------------- Data Structures

class covise__StringPair : public xsd__anyType
{
public:
    std::string first 1; ///< Required element.
    std::string second 1; ///< Required element.
};

class covise__Point : public xsd__anyType
{
public:
    int x 1; ///< Required element.
    int y 1; ///< Required element.
};

class covise__ColormapPin : public xsd__anyType
{
public:
    float r 1; ///< Required element.
    float g 1; ///< Required element.
    float b 1; ///< Required element.
    float a 1; ///< Required element.
    float position 1; ///< Required element.
};

class covise__Colormap : public xsd__anyType
{
public:
    std::string name 1; ///< Required element.
    std::vector<covise__ColormapPin> pins 1; ///< Required element.
};

class covise__Parameter : public xsd__anyType
{
public:
    std::string name 1; ///< Required element.
    std::string type 1; ///< Required element.
    std::string description 1; ///< Required element.
    bool mapped 1; ///< Required element.

    virtual covise__Parameter *clone() const;
    virtual ~covise__Parameter();
};

class covise__BooleanParameter : public covise__Parameter
{
public:
    bool value 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__ChoiceParameter : public covise__Parameter
{
public:
    int selected 1; ///< Required element.
    std::vector<std::string> choices 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__FileBrowserParameter : public covise__Parameter
{
public:
    std::string value 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__FloatScalarParameter : public covise__Parameter
{
public:
    float value 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__FloatSliderParameter : public covise__Parameter
{
public:
    float value 1; ///< Required element.
    float min 1; ///< Required element.
    float max 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__FloatVectorParameter : public covise__Parameter
{
public:
    std::vector<float> value 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__IntScalarParameter : public covise__Parameter
{
public:
    int value 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__IntSliderParameter : public covise__Parameter
{
public:
    int value 1; ///< Required element.
    int min 1; ///< Required element.
    int max 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__IntVectorParameter : public covise__Parameter
{
public:
    std::vector<int> value 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__StringParameter : public covise__Parameter
{
public:
    std::string value 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__ColormapChoiceParameter : public covise__Parameter
{
public:
    int selected 1; ///< Required element.
    std::vector<covise__Colormap> colormaps 1; ///< Required element.

    virtual covise__Parameter *clone() const;
};

class covise__Port : public xsd__anyType
{
public:
    std::string name 1; ///< Required element.
    std::vector<std::string> types 1; ///< Required element.
    std::string portType 1; ///< Required element.
    std::string id 1; ///< Required element.
    std::string moduleID 1; ///< Required element.
};

class covise__Module : public xsd__anyType
{
public:
    std::string name 1; ///< Required element.
    std::string category 1; ///< Required element.
    std::string host 1; ///< Required element.
    std::string description 1; ///< Required element.
    std::string instance 1; ///< Required element.
    std::string id 1; ///< Required element.
    std::string title 1; ///< Required element.
    covise__Point position 1; ///< Required element.
    std::vector<covise__Parameter *> parameters 1; ///< Required element.
    std::vector<covise__Port> inputPorts 1; ///< Required element.
    std::vector<covise__Port> outputPorts 1; ///< Required element.

    covise__Module();
    virtual ~covise__Module();
    covise__Module(const covise__Module &);
};

class covise__Link : public xsd__anyType
{
public:
    std::string id 1; ///< Required element.
    covise__Port from 1; ///< Required element.
    covise__Port to 1; ///< Required element.
};

#import "wscoviseevents.h"

// -------------- Forward declarations

class _covise__addEventListener;
class _covise__addEventListenerResponse;

class _covise__removeEventListener;
class _covise__removeEventListenerResponse;

class _covise__executeNet;
class _covise__executeNetResponse;

class _covise__openNet;
class _covise__openNetResponse;

class _covise__quit;
class _covise__quitResponse;

class _covise__addPartner;
class _covise__addPartnerResponse;

enum covise_AddPartnerMethod
{
    covise__AddPartnerMethod__RExec, ///< xs:string value="RExec"
    covise__AddPartnerMethod__RSH, ///< xs:string value="RSH"
    covise__AddPartnerMethod__SSH, ///< xs:string value="SSH"
    covise__AddPartnerMethod__NQS, ///< xs:string value="NQS"
    covise__AddPartnerMethod__Manual, ///< xs:string value="Manual"
    covise__AddPartnerMethod__RemoteDaemon ///< xs:string value="RemoteDaemon"
};

/// Class wrapper
//class covise__AddPartnerMethod_ : public xsd__anyType
//{ public:
//   enum covise__AddPartnerMethod            __item;
//};

class _covise__listModules;
class _covise__listModulesResponse;

class _covise__listHosts;
class _covise__listHostsResponse;

class _covise__getRunningModules;
class _covise__getRunningModulesResponse;

class _covise__getRunningModule;
class _covise__getRunningModuleResponse;

class _covise__setParameter;
class _covise__setParameterResponse;

class _covise__setParameterFromString;
class _covise__setParameterFromStringResponse;

class _covise__getParameterAsString;
class _covise__getParameterAsStringResponse;

class _covise__executeModule;
class _covise__executeModuleResponse;

class _covise__getEvent;
class _covise__getEventResponse;

class _covise__getModuleID;
class _covise__getModuleIDResponse;

class _covise__getConfigEntry;
class _covise__getConfigEntryResponse;

class _covise__deleteModule;
class _covise__deleteModuleResponse;

class _covise__instantiateModule;
class _covise__instantiateModuleResponse;

class _covise__link;
class _covise__linkResponse;

class _covise__getLinks;
class _covise__getLinksResponse;

class _covise__unlink;
class _covise__unlinkResponse;

class _covise__getFileInfoList;
class _covise__getFileInfoListResponse;

class _covise__isFileExist;
class _covise__isFileExistResponse;

class _covise__isDirExist;
class _covise__isDirExistResponse;

class _covise__uploadFile;
class _covise__uploadFileResponse;

class _covise__setParameterFromUploadedFile;
class _covise__setParameterFromUploadedFileResponse;

class _covise__createNewDir;
class _covise__createNewDirResponse;

class _covise__deleteDir;
class _covise__deleteDirResponse;

class _covise__uploadFileMtom;
class _covise__uploadFileMtomResponse;

// -------------- Request / Response classes

class _covise__addEventListener
{
public:
    struct soap *soap;
};

class _covise__addEventListenerResponse
{
public:
    std::string uuid 1;
    struct soap *soap;
};

class _covise__removeEventListener
{
public:
    std::string uuid 1; ///< Required element.
    struct soap *soap;
};

class _covise__removeEventListenerResponse
{
public:
    struct soap *soap;
};

class _covise__executeNet
{
public:
    struct soap *soap;
};

class _covise__executeNetResponse
{
public:
    struct soap *soap;
};

class _covise__openNet
{
public:
    std::string filename 1; ///< Required element.
    struct soap *soap;
};

class _covise__openNetResponse
{
public:
    struct soap *soap;
};

class _covise__quit
{
public:
    struct soap *soap;
};

class _covise__quitResponse
{
public:
    struct soap *soap;
};

class _covise__addPartner
{
public:
    enum covise__AddPartnerMethod *method 1; ///< Required element.
    std::string *ip 0; ///< Optional element.
    std::string *user 0; ///< Optional element.
    std::string *password 0; ///< Optional element.
    int *timeout 0; ///< Optional element.
    std::string *display 0; ///< Optional element.
    struct soap *soap;
};

class _covise__listModules
{
public:
    std::string ipaddr 1; ///< Required element
    struct soap *soap;
};

class _covise__listModulesResponse
{
public:
    std::string ipaddr 1; ///< Required element
    std::vector<covise__StringPair> modules 1; ///< Required element
    struct soap *soap;
};

class _covise__addPartnerResponse
{
public:
    bool *success;
    struct soap *soap;
};

class _covise__listHosts
{
public:
    struct soap *soap;
};

class _covise__listHostsResponse
{
public:
    std::vector<std::string> hosts 1; ///< Required element
    struct soap *soap;
};

class _covise__getRunningModules
{
public:
    struct soap *soap;
};

class _covise__getRunningModulesResponse
{
public:
    std::vector<covise__Module> modules 1; ///< Required element
    std::string networkFile 1; ///< Required element
    struct soap *soap;
};

class _covise__getRunningModule
{
public:
    std::string moduleID 1; ///< Required element
    struct soap *soap;
};

class _covise__getRunningModuleResponse
{
public:
    covise__Module module 1; ///< Required element
    struct soap *soap;
};

class _covise__setParameter
{
public:
    std::string moduleID 1; ///< Required element
    covise__Parameter *parameter 1; ///< Required element
    struct soap *soap;
};

class _covise__setParameterResponse
{
public:
    struct soap *soap;
};

class _covise__setParameterFromString
{
public:
    std::string moduleID 1; ///< Required element
    std::string parameter 1; ///< Required element
    std::string value 1; ///< Required element
    struct soap *soap;
};

class _covise__setParameterFromStringResponse
{
public:
    struct soap *soap;
};

class _covise__getParameterAsString
{
public:
    struct soap *soap;
    std::string moduleID 1; ///< Required element
    std::string parameter 1; ///< Required element
};

class _covise__getParameterAsStringResponse
{
public:
    struct soap *soap;
    std::string value 1; ///< Required element
};

class _covise__executeModule
{
public:
    std::string moduleID 1; ///< Required element
    struct soap *soap;
};

class _covise__executeModuleResponse
{
public:
    struct soap *soap;
};

class _covise__getEvent
{
public:
    std::string uuid 1; ///< Required element
    int *timeout 0; ///< Optional element.
    struct soap *soap;
};

class _covise__getEventResponse
{
public:
    covise__Event *event;
    std::string uuid;
    struct soap *soap;
};

class _covise__getModuleID
{
public:
    std::string module 1; ///< Required element
    std::string instance 1; ///< Required element
    std::string host 1; ///< Required element
    struct soap *soap;
};

class _covise__getModuleIDResponse
{
public:
    std::string moduleID 1; ///< Required element
    struct soap *soap;
};

class _covise__getConfigEntry
{
public:
    std::string section 1; ///< Required element
    std::string variable 1; ///< Required element
    struct soap *soap;
};

class _covise__getConfigEntryResponse
{
public:
    std::string value 1; ///< Required element
    struct soap *soap;
};

class _covise__deleteModule
{
public:
    std::string moduleID 1; ///< Required element
    struct soap *soap;
};

class _covise__deleteModuleResponse
{
public:
    struct soap *soap;
};

class _covise__instantiateModule
{
public:
    std::string module 1; ///< Required element
    std::string host 1; ///< Required element
    int *x 0; ///< Optional element
    int *y 0; ///< Optional element
    struct soap *soap;
};

class _covise__instantiateModuleResponse
{
public:
    struct soap *soap;
};

class _covise__link
{
public:
    std::string fromModule 1; ///< Required element
    std::string fromPort 1; ///< Required element
    std::string toModule 1; ///< Required element
    std::string toPort 1; ///< Required element
    struct soap *soap;
};

class _covise__linkResponse
{
public:
    struct soap *soap;
};

class _covise__getLinks
{
public:
    struct soap *soap;
};

class _covise__getLinksResponse
{
public:
    std::vector<covise__Link> links 1; ///< Required element
    struct soap *soap;
};

class _covise__unlink
{
public:
    std::string linkID 1; ///< Required element
    struct soap *soap;
};

class _covise__unlinkResponse
{
public:
    struct soap *soap;
};

class _covise__getFileInfoList
{
public:
    std::string path 1; ///< Required element
    struct soap *soap;
};

class _covise__getFileInfoListResponse
{
public:
    std::vector<covise__FileInfo> fileInfoList 1; ///< Required element
    struct soap *soap;
};

class _covise__isFileExist
{
public:
    std::string path 1; ///< Required element
    std::string fileName 1; ///< Required element
    struct soap *soap;
};

class _covise__isFileExistResponse
{
public:
    bool result 1; ///< Required element
    bool isFileExist 1; ///< Required element
    struct soap *soap;
};

class _covise__isDirExist
{
public:
    std::string path 1; ///< Required element
    std::string newDir 1; ///< Required element
    struct soap *soap;
};

class _covise__isDirExistResponse
{
public:
    bool result 1; ///< Required element
    bool isDirExist 1; ///< Required element
    struct soap *soap;
};

class _covise__uploadFile
{
public:
    std::string path 1; ///< Required element
    std::string fileName 1; ///< Required element
    xsd__base64Binary resource 1; ///< Required element
    int chunkIndex 1; ///< Required element
    int chunkNr 1; ///< Required element
    int chunkSize 1; ///< Required element
    int fileSize 1; ///< Required element
    bool fileTruncated 1; ///< Required element
    struct soap *soap;
};

class _covise__uploadFileResponse
{
public:
    bool result 1; ///< Required element
    bool lastChunk 1; ///< Required element
    struct soap *soap;
};

class _covise__createNewDir
{
public:
    std::string path 1; ///< Required element
    std::string newDir 1; ///< Required element
    struct soap *soap;
};

class _covise__createNewDirResponse
{
public:
    bool result 1; ///< Required element
    struct soap *soap;
};

class _covise__deleteDir
{
public:
    std::string path 1; ///< Required element
    struct soap *soap;
};

class _covise__deleteDirResponse
{
public:
    bool result 1; ///< Required element
    struct soap *soap;
};

class _covise__setParameterFromUploadedFile
{
public:
    std::string moduleID 1; ///< Required element
    std::string parameter 1; ///< Required element
    std::string value 1; ///< Required element
    struct soap *soap;
};

class _covise__setParameterFromUploadedFileResponse
{
public:
    struct soap *soap;
};

class _covise__uploadFileMtom
{
public:
    std::string path 1; ///< Required element
    std::string fileName 1; ///< Required element
    covise__FileData fileData 1; ///< Required element
    bool fileTrunc 1; ///< Required element
    int fileSize 1; ///< Required element
    struct soap *soap;
};

class _covise__uploadFileMtomResponse
{
public:
    bool result 1; ///< Required element
    bool lastChunk 1; ///< Required element
    struct soap *soap;
};

// -------------- Methods

//gsoap covise   service name:	COVISE
//gsoap covise   service type:	ServiceSoap
//gsoap covise   service port:	http://localhost:31111/
//gsoap covise   service namespace:	http://www.hlrs.de/organization/vis/covise
//gsoap covise   service transport:	http://schemas.xmlsoap.org/soap/http

//gsoap covise   service method-style:	  addEventListener document
//gsoap covise   service method-encoding: addEventListener literal
//gsoap covise   service method-action:	  addEventListener http://www.hlrs.de/organization/vis/covise/addEventListener

int __covise__addEventListener(
    _covise__addEventListener *covise__addEventListener, ///< Request parameter
    _covise__addEventListenerResponse *covise__addEventListenerResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  removeEventListener document
//gsoap covise   service method-encoding: removeEventListener literal
//gsoap covise   service method-action:	  removeEventListener http://www.hlrs.de/organization/vis/covise/removeEventListener

int __covise__removeEventListener(
    _covise__removeEventListener *covise__removeEventListener, ///< Request parameter
    _covise__removeEventListenerResponse *covise__removeEventListenerResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  executeNet document
//gsoap covise   service method-encoding: executeNet literal
//gsoap covise   service method-action:	  executeNet http://www.hlrs.de/organization/vis/covise/executeNet

int __covise__executeNet(
    _covise__executeNet *covise__executeNet, ///< Request parameter
    _covise__executeNetResponse *covise__executeNetResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  openNet document
//gsoap covise   service method-encoding: openNet literal
//gsoap covise   service method-action:	  openNet http://www.hlrs.de/organization/vis/covise/openNet

int __covise__openNet(
    _covise__openNet *covise__openNet, ///< Request parameter
    _covise__openNetResponse *covise__openNetResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  addPartner document
//gsoap covise   service method-encoding: addPartner literal
//gsoap covise   service method-action:	  addPartner http://www.hlrs.de/organization/vis/covise/addPartner

int __covise__addPartner(
    _covise__addPartner *covise__addPartner, ///< Request parameter
    _covise__addPartnerResponse *covise__addPartnerResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  quit document
//gsoap covise   service method-encoding: quit literal
//gsoap covise   service method-action:	  quit http://www.hlrs.de/organization/vis/covise/quit

int __covise__quit(
    _covise__quit *covise__quit, ///< Request parameter
    _covise__quitResponse *covise__quitResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  listModules document
//gsoap covise   service method-encoding:      listModules literal
//gsoap covise   service method-action:	  listModules http://www.hlrs.de/organization/vis/covise/listModules

int __covise__listModules(
    _covise__listModules *covise__listModules, ///< Request parameter
    _covise__listModulesResponse *covise__listModulesResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  listHosts document
//gsoap covise   service method-encoding:      listHosts literal
//gsoap covise   service method-action:	  listHosts http://www.hlrs.de/organization/vis/covise/listHosts

int __covise__listHosts(
    _covise__listHosts *covise__listHosts, ///< Request parameter
    _covise__listHostsResponse *covise__listHostsResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  getRunningModules document
//gsoap covise   service method-encoding:      getRunningModules literal
//gsoap covise   service method-action:	  getRunningModules http://www.hlrs.de/organization/vis/covise/getRunningModules

int __covise__getRunningModules(
    _covise__getRunningModules *covise__getRunningModules, ///< Request parameter
    _covise__getRunningModulesResponse *covise__getRunningModulesResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  setParameter document
//gsoap covise   service method-encoding:      setParameter literal
//gsoap covise   service method-action:	  setParameter http://www.hlrs.de/organization/vis/covise/setParameter

int __covise__setParameter(
    _covise__setParameter *covise__setParameter, ///< Request parameter
    _covise__setParameterResponse *covise__setParameterResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  setParameterFromString document
//gsoap covise   service method-encoding:      setParameterFromString literal
//gsoap covise   service method-action:	  setParameterFromString http://www.hlrs.de/organization/vis/covise/setParameterFromString

int __covise__setParameterFromString(
    _covise__setParameterFromString *covise__setParameterFromString, ///< Request parameter
    _covise__setParameterFromStringResponse *covise__setParameterFromStringResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  getParameterAsString document
//gsoap covise   service method-encoding:      getParameterAsString literal
//gsoap covise   service method-action:	  getParameterAsString http://www.hlrs.de/organization/vis/covise/getParameterAsString

int __covise__getParameterAsString(
    _covise__getParameterAsString *covise__getParameterAsString, ///< Request parameter
    _covise__getParameterAsStringResponse *covise__getParameterAsStringResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  executeModule document
//gsoap covise   service method-encoding:      executeModule literal
//gsoap covise   service method-action:	  executeModule http://www.hlrs.de/organization/vis/covise/executeModule

int __covise__executeModule(
    _covise__executeModule *covise__executeModule, ///< Request parameter
    _covise__executeModuleResponse *covise__executeModuleResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  getEvent document
//gsoap covise   service method-encoding:      getEvent literal
//gsoap covise   service method-action:	  getEvent http://www.hlrs.de/organization/vis/covise/getEvent

int __covise__getEvent(
    _covise__getEvent *covise__getEvent, ///< Request parameter
    _covise__getEventResponse *covise__getEventResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  getRunningModule document
//gsoap covise   service method-encoding:      getRunningModule literal
//gsoap covise   service method-action:	  getRunningModule http://www.hlrs.de/organization/vis/covise/getRunningModule

int __covise__getRunningModule(
    _covise__getRunningModule *covise__getRunningModule, ///< Request parameter
    _covise__getRunningModuleResponse *covise__getRunningModuleResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  getModuleID document
//gsoap covise   service method-encoding:      getModuleID literal
//gsoap covise   service method-action:	  getModuleID http://www.hlrs.de/organization/vis/covise/getModuleID

int __covise__getModuleID(
    _covise__getModuleID *covise__getModuleID, ///< Request parameter
    _covise__getModuleIDResponse *covise__getModuleIDResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  getConfigEntry document
//gsoap covise   service method-encoding:      getConfigEntry literal
//gsoap covise   service method-action:	  getConfigEntry http://www.hlrs.de/organization/vis/covise/getConfigEntry

int __covise__getConfigEntry(
    _covise__getConfigEntry *covise__getConfigEntry, ///< Request parameter
    _covise__getConfigEntryResponse *covise__getConfigEntryResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  deleteModule document
//gsoap covise   service method-encoding:      deleteModule literal
//gsoap covise   service method-action:	  deleteModule http://www.hlrs.de/organization/vis/covise/deleteModule

int __covise__deleteModule(
    _covise__deleteModule *covise__deleteModule, ///< Request parameter
    _covise__deleteModuleResponse *covise__deleteModuleResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  instantiateModule document
//gsoap covise   service method-encoding:      instantiateModule literal
//gsoap covise   service method-action:	  instantiateModule http://www.hlrs.de/organization/vis/covise/instantiateModule

int __covise__instantiateModule(
    _covise__instantiateModule *covise__instantiateModule, ///< Request parameter
    _covise__instantiateModuleResponse *covise__instantiateModuleResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  link document
//gsoap covise   service method-encoding:      link literal
//gsoap covise   service method-action:	  link http://www.hlrs.de/organization/vis/covise/link

int __covise__link(
    _covise__link *covise__link, ///< Request parameter
    _covise__linkResponse *covise__linkResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  getLinks document
//gsoap covise   service method-encoding:      getLinks literal
//gsoap covise   service method-action:	  getLinks http://www.hlrs.de/organization/vis/covise/getLinks

int __covise__getLinks(
    _covise__getLinks *covise__getLinks, ///< Request parameter
    _covise__getLinksResponse *covise__getLinksResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  unlink document
//gsoap covise   service method-encoding:      unlink literal
//gsoap covise   service method-action:	  unlink http://www.hlrs.de/organization/vis/covise/unlink

int __covise__unlink(
    _covise__unlink *covise__unlink, ///< Request parameter
    _covise__unlinkResponse *covise__unlinkResponse ///< Response parameter
    );

//gsoap covise   service method-style:    getFileInfoList document
//gsoap covise   service method-encoding:      getFileInfoList literal
//gsoap covise   service method-action:   getFileInfoList http://www.hlrs.de/organization/vis/covise/getFileInfoList

int __covise__getFileInfoList(
    _covise__getFileInfoList *covise__getFileInfoList, ///< Request parameter
    _covise__getFileInfoListResponse *covise__getFileInfoListResponse ///< Response parameter
    );

//gsoap covise   service method-style:    uploadFile document
//gsoap covise   service method-encoding:      uploadFile literal
//gsoap covise   service method-action:   uploadFile http://www.hlrs.de/organization/vis/covise/uploadFile

int __covise__uploadFile(
    _covise__uploadFile *covise__uploadFile, ///< Request parameter
    _covise__uploadFileResponse *covise__uploadFileResponse ///< Response parameter
    );

//gsoap covise   service method-style:    isFileExist document
//gsoap covise   service method-encoding:      isFileExist literal
//gsoap covise   service method-action:   isFileExist http://www.hlrs.de/organization/vis/covise/isFileExist

int __covise__isFileExist(
    _covise__isFileExist *covise__isFileExist, ///< Request parameter
    _covise__isFileExistResponse *covise__isFileExistResponse ///< Response parameter
    );

//gsoap covise   service method-style:    isDirExist document
//gsoap covise   service method-encoding:      isDirExist literal
//gsoap covise   service method-action:   isDirExist http://www.hlrs.de/organization/vis/covise/isDirExist

int __covise__isDirExist(
    _covise__isDirExist *covise__isDirExist, ///< Request parameter
    _covise__isDirExistResponse *covise__isDirExistResponse ///< Response parameter
    );

//gsoap covise   service method-style:    createNewDir document
//gsoap covise   service method-encoding:      createNewDir literal
//gsoap covise   service method-action:   createNewDir http://www.hlrs.de/organization/vis/covise/createNewDir

int __covise__createNewDir(
    _covise__createNewDir *covise__createNewDir, ///< Request parameter
    _covise__createNewDirResponse *covise__createNewDirResponse ///< Response parameter
    );

//gsoap covise   service method-style:    deleteDir document
//gsoap covise   service method-encoding:      deleteDir literal
//gsoap covise   service method-action:   deleteDir http://www.hlrs.de/organization/vis/covise/deleteDir

int __covise__deleteDir(
    _covise__deleteDir *covise__deleteDir, ///< Request parameter
    _covise__deleteDirResponse *covise__deleteDirResponse ///< Response parameter
    );

//gsoap covise   service method-style:	  setParameterFromUploadedFile document
//gsoap covise   service method-encoding:     setParameterFromUploadedFile literal
//gsoap covise   service method-action:	  setParameterFromUploadedFile http://www.hlrs.de/organization/vis/covise/setParameterFromUploadedFile

int __covise__setParameterFromUploadedFile(
    _covise__setParameterFromUploadedFile *covise__setParameterFromUploadedFile, ///< Request parameter
    _covise__setParameterFromUploadedFileResponse *covise__setParameterFromUploadedFileResponse ///< Response parameter
    );

//gsoap covise   service method-style:    uploadFileMtom document
//gsoap covise   service method-encoding:      uploadFileMtom literal
//gsoap covise   service method-action:   uploadFileMtom http://www.hlrs.de/organization/vis/covise/uploadFileMtom

int __covise__uploadFileMtom(
    _covise__uploadFileMtom *covise__uploadFileMtom, ///< Request parameter
    _covise__uploadFileMtomResponse *covise__uploadFileMtomResponse ///< Response parameter
    );

// EndOfDeclarations
}
