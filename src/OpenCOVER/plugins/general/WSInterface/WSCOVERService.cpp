/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#include <winsock2.h>
#endif

#include "WSInterfacePlugin.h"
#include "opencoverCOVERService.h"

#include "stdsoap2.cpp"

int opencover::COVERService::openFile(opencover::_opencover__openFile *opencover__openFile, opencover::_opencover__openFileResponse *)
{
    WSInterfacePlugin::instance()->openFile(opencover__openFile->filename.c_str());
    return SOAP_OK;
}

int opencover::COVERService::addFile(opencover::_opencover__addFile *opencover__addFile, opencover::_opencover__addFileResponse *)
{
    WSInterfacePlugin::instance()->addFile(opencover__addFile->filename.c_str());
    return SOAP_OK;
}

int opencover::COVERService::quit(opencover::_opencover__quit *, opencover::_opencover__quitResponse *)
{
    WSInterfacePlugin::instance()->quit();
    return SOAP_OK;
}

int opencover::COVERService::connectToVnc(opencover::_opencover__connectToVnc *connectToVnc,
                                          opencover::_opencover__connectToVncResponse *)
{
    WSInterfacePlugin::instance()->connectToVnc(connectToVnc->host.c_str(),
                                                connectToVnc->port,
                                                connectToVnc->passwd.c_str());
    return SOAP_OK;
}

int opencover::COVERService::disconnectFromVnc(opencover::_opencover__disconnectFromVnc *,
                                               opencover::_opencover__disconnectFromVncResponse *)
{

    WSInterfacePlugin::instance()->disconnectFromVnc();
    return SOAP_OK;
}

int opencover::COVERService::setVisibleVnc(opencover::_opencover__setVisibleVnc *setVisibleVnc,
                                           opencover::_opencover__setVisibleVncResponse *)
{
    WSInterfacePlugin::instance()->setVisibleVnc(setVisibleVnc->on);
    return SOAP_OK;
}

int opencover::COVERService::sendCustomMessage(opencover::_opencover__sendCustomMessage *sendCustomMessage,
                                               opencover::_opencover__sendCustomMessageResponse *)
{
    WSInterfacePlugin::instance()->sendCustomMessage(sendCustomMessage->parameter.c_str());
    return SOAP_OK;
}

int opencover::COVERService::show(opencover::_opencover__show *show,
                                  opencover::_opencover__showResponse *)
{
    WSInterfacePlugin::instance()->show(show->objectName.c_str());
    return SOAP_OK;
}

int opencover::COVERService::hide(opencover::_opencover__hide *hide,
                                  opencover::_opencover__hideResponse *)
{
    WSInterfacePlugin::instance()->hide(hide->objectName.c_str());
    return SOAP_OK;
}

int opencover::COVERService::viewAll(opencover::_opencover__viewAll *, opencover::_opencover__viewAllResponse *)
{
    WSInterfacePlugin::instance()->viewAll();
    return SOAP_OK;
}

int opencover::COVERService::resetView(opencover::_opencover__resetView *, opencover::_opencover__resetViewResponse *)
{
    WSInterfacePlugin::instance()->resetView();
    return SOAP_OK;
}

int opencover::COVERService::walk(opencover::_opencover__walk *, opencover::_opencover__walkResponse *)
{
    WSInterfacePlugin::instance()->walk();
    return SOAP_OK;
}

int opencover::COVERService::fly(opencover::_opencover__fly *, opencover::_opencover__flyResponse *)
{
    WSInterfacePlugin::instance()->fly();
    return SOAP_OK;
}

int opencover::COVERService::drive(opencover::_opencover__drive *, opencover::_opencover__driveResponse *)
{
    WSInterfacePlugin::instance()->drive();
    return SOAP_OK;
}

int opencover::COVERService::scale(opencover::_opencover__scale *, opencover::_opencover__scaleResponse *)
{
    WSInterfacePlugin::instance()->scale();
    return SOAP_OK;
}

int opencover::COVERService::xform(opencover::_opencover__xform *, opencover::_opencover__xformResponse *)
{
    WSInterfacePlugin::instance()->xform();
    return SOAP_OK;
}

int opencover::COVERService::wireframe(opencover::_opencover__wireframe *query, opencover::_opencover__wireframeResponse *)
{
    WSInterfacePlugin::instance()->wireframe(query->on);
    return SOAP_OK;
}

int opencover::COVERService::snapshot(opencover::_opencover__snapshot *query, opencover::_opencover__snapshotResponse *)
{
    WSInterfacePlugin::instance()->snapshot(QString::fromStdString(query->path));
    return SOAP_OK;
}
