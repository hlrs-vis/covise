/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

namespace opencover
{

#import "stlvector.h"

//gsoap opencover    schema namespace:	http://www.hlrs.de/organization/vis/opencover
//gsoap opencover    schema elementForm:	qualified
//gsoap opencover    schema attributeForm:	unqualified

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

// -------------- Forward declarations

class _opencover__quit;
class _opencover__quitResponse;

class _opencover__openFile;
class _opencover__openFileResponse;

class _opencover__connectToVnc;
class _opencover__connectToVncResponse;

class _opencover__disconnectFromVnc;
class _opencover__disconnectFromVncResponse;

class _opencover__setVisibleVnc;
class _opencover__setVisibleVncResponse;

class _opencover__sendCustomMessage;
class _opencover__sendCustomMessageResponse;

class _opencover__show;
class _opencover__showResponse;

class _opencover__hide;
class _opencover__hideResponse;

class _opencover__viewAll;
class _opencover__viewAllResponse;

class _opencover__resetView;
class _opencover__resetViewResponse;

class _opencover__walk;
class _opencover__walkResponse;

class _opencover__fly;
class _opencover__flyResponse;

class _opencover__drive;
class _opencover__driveResponse;

class _opencover__scale;
class _opencover__scaleResponse;

class _opencover__xform;
class _opencover__xformResponse;

class _opencover__wireframe;
class _opencover__wireframeResponse;

class _opencover__addFile;
class _opencover__addFileResponse;

class _opencover__snapshot;
class _opencover__snapshotResponse;

// -------------- Request / Response classes

class _opencover__quit
{
public:
    struct soap *soap;
};

class _opencover__quitResponse
{
public:
    struct soap *soap;
};

class _opencover__openFile
{
public:
    std::string filename 1; ///< Required element.
    struct soap *soap;
};

class _opencover__openFileResponse
{
public:
    struct soap *soap;
};

class _opencover__connectToVnc
{
public:
    std::string host 1; ///< Required element.
    unsigned int port 1; ///< Required element.
    std::string passwd 1; ///< Required element.
    struct soap *soap;
};

class _opencover__connectToVncResponse
{
public:
    struct soap *soap;
};

class _opencover__disconnectFromVnc
{
public:
    struct soap *soap;
};

class _opencover__disconnectFromVncResponse
{
public:
    struct soap *soap;
};

class _opencover__setVisibleVnc
{
public:
    bool on 1; ///< Required element.
    struct soap *soap;
};

class _opencover__setVisibleVncResponse
{
public:
    struct soap *soap;
};

class _opencover__sendCustomMessage
{
public:
    std::string parameter 1; ///< Required element.
    struct soap *soap;
};

class _opencover__sendCustomMessageResponse
{
public:
    struct soap *soap;
};

class _opencover__show
{
public:
    std::string objectName 1; ///< Required element.
    struct soap *soap;
};

class _opencover__showResponse
{
public:
    struct soap *soap;
};

class _opencover__hide
{
public:
    std::string objectName 1; ///< Required element.
    struct soap *soap;
};

class _opencover__hideResponse
{
public:
    struct soap *soap;
};

class _opencover__viewAll
{
public:
    struct soap *soap;
};

class _opencover__viewAllResponse
{
public:
    struct soap *soap;
};

class _opencover__resetView
{
public:
    struct soap *soap;
};

class _opencover__resetViewResponse
{
public:
    struct soap *soap;
};

class _opencover__walk
{
public:
    struct soap *soap;
};

class _opencover__walkResponse
{
public:
    struct soap *soap;
};

class _opencover__fly
{
public:
    struct soap *soap;
};

class _opencover__flyResponse
{
public:
    struct soap *soap;
};

class _opencover__drive
{
public:
    struct soap *soap;
};

class _opencover__driveResponse
{
public:
    struct soap *soap;
};

class _opencover__scale
{
public:
    struct soap *soap;
};

class _opencover__scaleResponse
{
public:
    struct soap *soap;
};

class _opencover__xform
{
public:
    struct soap *soap;
};

class _opencover__xformResponse
{
public:
    struct soap *soap;
};

class _opencover__wireframe
{
public:
    bool on;
    struct soap *soap;
};

class _opencover__wireframeResponse
{
public:
    struct soap *soap;
};

class _opencover__addFile
{
public:
    std::string filename 1; ///< Required element.
    struct soap *soap;
};

class _opencover__addFileResponse
{
public:
    struct soap *soap;
};

class _opencover__snapshot
{
public:
    std::string path 1; ///< Required element.
    struct soap *soap;
};

class _opencover__snapshotResponse
{
public:
    struct soap *soap;
};

// -------------- Methods

//gsoap opencover   service name:	COVER
//gsoap opencover   service type:	ServiceSoap
//gsoap opencover   service port:	http://localhost:32190/
//gsoap opencover   service namespace:	http://www.hlrs.de/organization/vis/opencover
//gsoap opencover   service transport:	http://schemas.xmlsoap.org/soap/http

//gsoap opencover   service method-style:	  openFile document
//gsoap opencover   service method-encoding: openFile literal
//gsoap opencover   service method-action:	  openNet http://www.hlrs.de/organization/vis/opencover/openFile

int __opencover__openFile(
    _opencover__openFile *opencover__openFile, ///< Request parameter
    _opencover__openFileResponse *opencover__openFileResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  quit document
//gsoap opencover   service method-encoding:      quit literal
//gsoap opencover   service method-action:	  quit http://www.hlrs.de/organization/vis/opencover/quit

int __opencover__quit(
    _opencover__quit *opencover__quit, ///< Request parameter
    _opencover__quitResponse *opencover__quitResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  connectToVnc document
//gsoap opencover   service method-encoding:      connectToVnc literal
//gsoap opencover   service method-action:	  connectToVnc http://www.hlrs.de/organization/vis/opencover/connectToVnc

int __opencover__connectToVnc(
    _opencover__connectToVnc *opencover__connectToVnc, ///< Request parameter
    _opencover__connectToVncResponse *opencover__connectToVncResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  disconnectFromVnc document
//gsoap opencover   service method-encoding:      disconnectFromVnc literal
//gsoap opencover   service method-action:	  disconnectFromVnc http://www.hlrs.de/organization/vis/opencover/disconnectFromVnc

int __opencover__disconnectFromVnc(
    _opencover__disconnectFromVnc *opencover__disconnectFromVnc, ///< Request parameter
    _opencover__disconnectFromVncResponse *opencover__disconnectFromVncResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  setVisibleVnc document
//gsoap opencover   service method-encoding:      setVisibleVnc literal
//gsoap opencover   service method-action:	  setVisibleVnc http://www.hlrs.de/organization/vis/opencover/setVisibleVnc

int __opencover__setVisibleVnc(
    _opencover__setVisibleVnc *opencover__setVisibleVnc, ///< Request parameter
    _opencover__setVisibleVncResponse *opencover__setVisibleVncResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  sendCustomMessage document
//gsoap opencover   service method-encoding:      sendCustomMessage literal
//gsoap opencover   service method-action:	  sendCustomMessage http://www.hlrs.de/organization/vis/opencover/sendCustomMessage

int __opencover__sendCustomMessage(
    _opencover__sendCustomMessage *opencover__sendCustomMessage, ///< Request parameter
    _opencover__sendCustomMessageResponse *opencover__sendCustomMessageResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  show document
//gsoap opencover   service method-encoding:      show literal
//gsoap opencover   service method-action:	  show http://www.hlrs.de/organization/vis/opencover/show

int __opencover__show(
    _opencover__show *opencover__show, ///< Request parameter
    _opencover__showResponse *opencover__showResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  hide document
//gsoap opencover   service method-encoding:      hide literal
//gsoap opencover   service method-action:	  hide http://www.hlrs.de/organization/vis/opencover/hide

int __opencover__hide(
    _opencover__hide *opencover__hide, ///< Request parameter
    _opencover__hideResponse *opencover__hideResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  viewAll document
//gsoap opencover   service method-encoding:      viewAll literal
//gsoap opencover   service method-action:	  viewAll http://www.hlrs.de/organization/vis/opencover/viewAll

int __opencover__viewAll(
    _opencover__viewAll *opencover__viewAll, ///< Request parameter
    _opencover__viewAllResponse *opencover__viewAllResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  resetView document
//gsoap opencover   service method-encoding:      resetView literal
//gsoap opencover   service method-action:	  resetView http://www.hlrs.de/organization/vis/opencover/resetView

int __opencover__resetView(
    _opencover__resetView *opencover__resetView, ///< Request parameter
    _opencover__resetViewResponse *opencover__resetViewResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  walk document
//gsoap opencover   service method-encoding:      walk literal
//gsoap opencover   service method-action:	  walk http://www.hlrs.de/organization/vis/opencover/walk

int __opencover__walk(
    _opencover__walk *opencover__walk, ///< Request parameter
    _opencover__walkResponse *opencover__walkResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  fly document
//gsoap opencover   service method-encoding:      fly literal
//gsoap opencover   service method-action:	  fly http://www.hlrs.de/organization/vis/opencover/fly

int __opencover__fly(
    _opencover__fly *opencover__fly, ///< Request parameter
    _opencover__flyResponse *opencover__flyResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  drive document
//gsoap opencover   service method-encoding:      drive literal
//gsoap opencover   service method-action:	  drive http://www.hlrs.de/organization/vis/opencover/drive

int __opencover__drive(
    _opencover__drive *opencover__drive, ///< Request parameter
    _opencover__driveResponse *opencover__driveResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  scale document
//gsoap opencover   service method-encoding:      scale literal
//gsoap opencover   service method-action:	  scale http://www.hlrs.de/organization/vis/opencover/scale

int __opencover__scale(
    _opencover__scale *opencover__scale, ///< Request parameter
    _opencover__scaleResponse *opencover__scaleResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  xform document
//gsoap opencover   service method-encoding:      xform literal
//gsoap opencover   service method-action:	  xform http://www.hlrs.de/organization/vis/opencover/xform

int __opencover__xform(
    _opencover__xform *opencover__xform, ///< Request parameter
    _opencover__xformResponse *opencover__xformResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  wireframe document
//gsoap opencover   service method-encoding:      wireframe literal
//gsoap opencover   service method-action:	  wireframe http://www.hlrs.de/organization/vis/opencover/wireframe

int __opencover__wireframe(
    _opencover__wireframe *opencover__wireframe, ///< Request parameter
    _opencover__wireframeResponse *opencover__wireframeResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  addFile document
//gsoap opencover   service method-encoding:      addFile literal
//gsoap opencover   service method-action:	  addFile http://www.hlrs.de/organization/vis/opencover/addFile

int __opencover__addFile(
    _opencover__addFile *opencover__addFile, ///< Request parameter
    _opencover__addFileResponse *opencover__addFileResponse ///< Response parameter
    );

//gsoap opencover   service method-style:	  snapshot document
//gsoap opencover   service method-encoding:      snapshot literal
//gsoap opencover   service method-action:	  snapshot http://www.hlrs.de/organization/vis/opencover/snapshot

int __opencover__snapshot(
    _opencover__snapshot *opencover__snapshot, ///< Request parameter
    _opencover__snapshotResponse *opencover__snapshotResponse ///< Response parameter
    );

// EndOfDeclarations
}
