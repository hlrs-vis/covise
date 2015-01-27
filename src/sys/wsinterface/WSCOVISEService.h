/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSCOVISESERVICE_H
#define WSCOVISESERVICE_H

#include "WSCoviseStub.h"
#include "coviseCOVISEService.h"

namespace covise
{

class WSCOVISEService : public covise::COVISEService
{

public:
    WSCOVISEService()
    {
    }
    //WSCOVISEService(const struct soap&);
    WSCOVISEService(struct soap &);
    virtual ~WSCOVISEService()
    {
    }

    WSCOVISEService *copy();

    virtual int addEventListener(covise::_covise__addEventListener *, covise::_covise__addEventListenerResponse *);
    virtual int removeEventListener(covise::_covise__removeEventListener *, covise::_covise__removeEventListenerResponse *);
    virtual int executeNet(covise::_covise__executeNet *, covise::_covise__executeNetResponse *);
    virtual int openNet(covise::_covise__openNet *, covise::_covise__openNetResponse *);
    virtual int addPartner(covise::_covise__addPartner *, covise::_covise__addPartnerResponse *);
    virtual int quit(covise::_covise__quit *, covise::_covise__quitResponse *);
    virtual int listModules(covise::_covise__listModules *, covise::_covise__listModulesResponse *);
    virtual int listHosts(covise::_covise__listHosts *, covise::_covise__listHostsResponse *);
    virtual int getRunningModules(covise::_covise__getRunningModules *, covise::_covise__getRunningModulesResponse *);
    virtual int setParameter(covise::_covise__setParameter *, covise::_covise__setParameterResponse *);
    virtual int setParameterFromString(covise::_covise__setParameterFromString *, covise::_covise__setParameterFromStringResponse *);
    virtual int getParameterAsString(_covise__getParameterAsString *, _covise__getParameterAsStringResponse *);
    virtual int executeModule(covise::_covise__executeModule *, covise::_covise__executeModuleResponse *);
    virtual int getEvent(covise::_covise__getEvent *, covise::_covise__getEventResponse *);
    virtual int getRunningModule(covise::_covise__getRunningModule *, covise::_covise__getRunningModuleResponse *);
    virtual int getModuleID(covise::_covise__getModuleID *, covise::_covise__getModuleIDResponse *);
    virtual int getConfigEntry(covise::_covise__getConfigEntry *, covise::_covise__getConfigEntryResponse *);
    virtual int deleteModule(covise::_covise__deleteModule *, covise::_covise__deleteModuleResponse *);
    virtual int instantiateModule(covise::_covise__instantiateModule *, covise::_covise__instantiateModuleResponse *);
    virtual int link(covise::_covise__link *, covise::_covise__linkResponse *);
    virtual int unlink(covise::_covise__unlink *, covise::_covise__unlinkResponse *);
    virtual int getLinks(covise::_covise__getLinks *, covise::_covise__getLinksResponse *);
    virtual int getFileInfoList(covise::_covise__getFileInfoList *, covise::_covise__getFileInfoListResponse *);
    virtual int isFileExist(covise::_covise__isFileExist *, covise::_covise__isFileExistResponse *);
    virtual int isDirExist(covise::_covise__isDirExist *, covise::_covise__isDirExistResponse *);
    virtual int uploadFile(covise::_covise__uploadFile *, covise::_covise__uploadFileResponse *);
    virtual int createNewDir(covise::_covise__createNewDir *, covise::_covise__createNewDirResponse *);
    virtual int deleteDir(covise::_covise__deleteDir *, covise::_covise__deleteDirResponse *);
    virtual int setParameterFromUploadedFile(covise::_covise__setParameterFromUploadedFile *, covise::_covise__setParameterFromUploadedFileResponse *);
    virtual int uploadFileMtom(covise::_covise__uploadFileMtom *, covise::_covise__uploadFileMtomResponse *);
};
}
#endif
