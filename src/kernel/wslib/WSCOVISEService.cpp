/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#include <winsock2.h>
#endif

#include "WSCoviseStub.h"
#include "coviseCOVISEService.h"

#include <QMutex>

static QMutex sequentialLock;

int covise::COVISEService::executeNet(covise::_covise__executeNet *, covise::_covise__executeNetResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::openNet(covise::_covise__openNet *, covise::_covise__openNetResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::addPartner(covise::_covise__addPartner *,
                                      covise::_covise__addPartnerResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::quit(covise::_covise__quit *, covise::_covise__quitResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::addEventListener(covise::_covise__addEventListener *, covise::_covise__addEventListenerResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::removeEventListener(covise::_covise__removeEventListener *, covise::_covise__removeEventListenerResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::getEvent(covise::_covise__getEvent *, covise::_covise__getEventResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::listModules(_covise__listModules *, _covise__listModulesResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::listHosts(_covise__listHosts *, _covise__listHostsResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::getRunningModules(_covise__getRunningModules *, _covise__getRunningModulesResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::getRunningModule(_covise__getRunningModule *, _covise__getRunningModuleResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::setParameter(_covise__setParameter *, _covise__setParameterResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::setParameterFromString(_covise__setParameterFromString *, _covise__setParameterFromStringResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::getParameterAsString(_covise__getParameterAsString *, _covise__getParameterAsStringResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::executeModule(_covise__executeModule *, _covise__executeModuleResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::getModuleID(_covise__getModuleID *, _covise__getModuleIDResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::getConfigEntry(_covise__getConfigEntry *, _covise__getConfigEntryResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::deleteModule(_covise__deleteModule *, _covise__deleteModuleResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::instantiateModule(_covise__instantiateModule *, _covise__instantiateModuleResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::link(_covise__link *, _covise__linkResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::unlink(_covise__unlink *, _covise__unlinkResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::getLinks(_covise__getLinks *, _covise__getLinksResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::getFileInfoList(_covise__getFileInfoList *, _covise__getFileInfoListResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::uploadFile(_covise__uploadFile *, _covise__uploadFileResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::isFileExist(_covise__isFileExist *, _covise__isFileExistResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::isDirExist(_covise__isDirExist *, _covise__isDirExistResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::createNewDir(_covise__createNewDir *, _covise__createNewDirResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::deleteDir(_covise__deleteDir *, _covise__deleteDirResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::setParameterFromUploadedFile(_covise__setParameterFromUploadedFile *, _covise__setParameterFromUploadedFileResponse *)
{
    return SOAP_NO_METHOD;
}

int covise::COVISEService::uploadFileMtom(_covise__uploadFileMtom *, _covise__uploadFileMtomResponse *)
{
    return SOAP_NO_METHOD;
}
