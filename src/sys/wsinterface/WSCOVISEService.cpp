/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSCOVISEService.h"
#include "WSMessageHandler.h"
#include "WSMainHandler.h"
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include "WSServer.h"
#include <sstream>
#include <fstream>
#include <QtCore>
#include <QMutex>

static QMutex sequentialLock;

//covise::WSCOVISEService::WSCOVISEService(const struct soap & s)
covise::WSCOVISEService::WSCOVISEService(struct soap &s)
    : covise::COVISEService(s)
{
    // added by hpcaiyin
    s.imode = SOAP_ENC_MTOM;
    s.omode = SOAP_ENC_MTOM;
}

covise::WSCOVISEService *covise::WSCOVISEService::copy()
{
    covise::WSCOVISEService *dup = new covise::WSCOVISEService(*(struct soap *)this);
    return dup;
}

int covise::WSCOVISEService::executeNet(covise::_covise__executeNet *, covise::_covise__executeNetResponse *)
{
    QMutexLocker l(&sequentialLock);
    covise::WSMessageHandler::instance()->executeNet();
    return SOAP_OK;
}

int covise::WSCOVISEService::openNet(covise::_covise__openNet *covise__openNet, covise::_covise__openNetResponse *)
{
    QMutexLocker l(&sequentialLock);
    covise::WSMessageHandler::instance()->openNet(QString::fromStdString(covise__openNet->filename));
    return SOAP_OK;
}

int covise::WSCOVISEService::addPartner(covise::_covise__addPartner *request,
                                        covise::_covise__addPartnerResponse *response)
{

    (void)request;
    (void)response;

    //   std::string ip = (request->ip ? *(request->ip) : "");
    //   std::string user = (request->user ? *(request->user) : "");
    //   std::string password = (request->password ? *(request->password) : "");
    //   int timeout = (request->timeout ? *(request->timeout) : 5);
    //   std::string display = (request->display ? *(request->display) : "");

    //   WSMessageHandler::AddPartnerMethod method;

    //   switch (*(request->method))
    //   {
    //   case covise::covise_AddPartnerMethod::covise__AddPartnerMethod__RExec:
    //     method = WSMessageHandler::RExec;
    //     break;

    //   case covise::covise_AddPartnerMethod::covise__AddPartnerMethod__RSH:
    //     method = WSMessageHandler::RSH;
    //     break;

    //   case covise::covise_AddPartnerMethod::covise__AddPartnerMethod__SSH:
    //     method = WSMessageHandler::SSH;
    //     break;

    //   case covise::covise_AddPartnerMethod::covise__AddPartnerMethod__NQS:
    //     method = WSMessageHandler::NQS;
    //     break;

    //   case covise::covise_AddPartnerMethod::covise__AddPartnerMethod__Manual:
    //     method = WSMessageHandler::Manual;
    //     break;

    //   case covise::covise_AddPartnerMethod::covise__AddPartnerMethod__RemoteDaemon:
    //     method = WSMessageHandler::RemoteDaemon;
    //     break;

    //   default:
    //     std::cerr << "WSCOVISEService::addPartner err: unkown method " << *(request->method) << std::endl;
    //     method = WSMessageHandler::Manual;
    //   }

    //   WSMessageHandler::instance()->addPartner(method, ip, user, password, timeout, display);
    //   return SOAP_OK;
    return SOAP_NO_METHOD;
}

int covise::WSCOVISEService::quit(covise::_covise__quit *, covise::_covise__quitResponse *)
{
    QMutexLocker l(&sequentialLock);
    covise::WSMessageHandler::instance()->quit();
    return SOAP_OK;
}

int covise::WSCOVISEService::addEventListener(covise::_covise__addEventListener *, covise::_covise__addEventListenerResponse *response)
{
    response->uuid = covise::WSMainHandler::instance()->addEventListener().toString().toStdString();
    return SOAP_OK;
}

int covise::WSCOVISEService::removeEventListener(covise::_covise__removeEventListener *request, covise::_covise__removeEventListenerResponse *)
{
    covise::WSMainHandler::instance()->removeEventListener(QString::fromStdString(request->uuid));
    return SOAP_OK;
}

int covise::WSCOVISEService::getEvent(covise::_covise__getEvent *request, covise::_covise__getEventResponse *response)
{
    if (request->timeout == 0)
        response->event = covise::WSMainHandler::instance()->consumeEvent(QString::fromStdString(request->uuid));
    else
        response->event = covise::WSMainHandler::instance()->consumeEvent(QString::fromStdString(request->uuid), *(request->timeout));
    response->uuid = request->uuid;
    return SOAP_OK;
}

int covise::WSCOVISEService::listModules(covise::_covise__listModules *request, covise::_covise__listModulesResponse *response)
{

    QMutexLocker l(&sequentialLock);

    QString ipaddr = QString::fromStdString(request->ipaddr);
    QLinkedList<covise::WSModule *> modules = covise::WSMainHandler::instance()->getAvailableModules()[ipaddr];

    response->ipaddr = request->ipaddr;

    response->modules.clear();

    foreach (covise::WSModule *module, modules)
    {
        covise::covise__StringPair moduleDescription;
        moduleDescription.first = module->getCategory().toStdString();
        moduleDescription.second = module->getName().toStdString();
        response->modules.push_back(moduleDescription);
    }
    return SOAP_OK;
}

int covise::WSCOVISEService::listHosts(covise::_covise__listHosts *request, covise::_covise__listHostsResponse *response)
{
    (void)request;

    QMutexLocker l(&sequentialLock);

    QList<QString> hosts = WSMainHandler::instance()->getAvailableModules().keys();

    response->hosts.clear();

    foreach (QString host, hosts)
    {
        response->hosts.push_back(host.toStdString());
    }

    return SOAP_OK;
}

int covise::WSCOVISEService::getRunningModules(covise::_covise__getRunningModules *request, covise::_covise__getRunningModulesResponse *response)
{
    (void)request;

    QMutexLocker l(&sequentialLock);

    response->networkFile = covise::WSMainHandler::instance()->getMap()->getMapName().toStdString();

    const QList<covise::WSModule *> modules = covise::WSMainHandler::instance()->getMap()->getModules();
    foreach (covise::WSModule *module, modules)
        response->modules.push_back(module->getSerialisable());
    return SOAP_OK;
}

int covise::WSCOVISEService::getRunningModule(covise::_covise__getRunningModule *request, covise::_covise__getRunningModuleResponse *response)
{
    QMutexLocker l(&sequentialLock);

    covise::WSModule *module = WSMainHandler::instance()->getMap()->getModule(QString::fromStdString(request->moduleID));
    if (module != 0)
        response->module = module->getSerialisable();
    return SOAP_OK;
}

int covise::WSCOVISEService::setParameter(covise::_covise__setParameter *request, covise::_covise__setParameterResponse *response)
{
    (void)response;

    QMutexLocker l(&sequentialLock);

    covise::WSMainHandler::instance()->setParameter(QString::fromStdString(request->moduleID), request->parameter);

    return SOAP_OK;
}

int covise::WSCOVISEService::setParameterFromString(covise::_covise__setParameterFromString *request, covise::_covise__setParameterFromStringResponse *response)
{
    (void)response;

    QMutexLocker l(&sequentialLock);

    covise::WSMainHandler::instance()->setParameterFromString(QString::fromStdString(request->moduleID), QString::fromStdString(request->parameter), QString::fromStdString(request->value));

    return SOAP_OK;
}

int covise::WSCOVISEService::getParameterAsString(covise::_covise__getParameterAsString *request, covise::_covise__getParameterAsStringResponse *response)
{
    QMutexLocker l(&sequentialLock);

    covise::WSModule *module = WSMainHandler::instance()->getMap()->getModule(QString::fromStdString(request->moduleID));

    if (module == 0)
        return SOAP_ERR;

    covise::WSParameter *parameter = module->getParameter(QString::fromStdString(request->parameter));

    if (parameter != 0)
        response->value = parameter->toString().toStdString();
    else
        return SOAP_ERR;

    return SOAP_OK;
}

int covise::WSCOVISEService::executeModule(covise::_covise__executeModule *request, covise::_covise__executeModuleResponse *response)
{
    (void)response;

    QMutexLocker l(&sequentialLock);

    WSMainHandler::instance()->executeModule(QString::fromStdString(request->moduleID));
    return SOAP_OK;
}

int covise::WSCOVISEService::getModuleID(covise::_covise__getModuleID *request, covise::_covise__getModuleIDResponse *response)
{
    QMutexLocker l(&sequentialLock);
    response->moduleID = covise::WSMainHandler::instance()->getMap()->makeKeyName(QString::fromStdString(request->module),
                                                                                  QString::fromStdString(request->instance),
                                                                                  QString::fromStdString(request->host))
                             .toStdString();
    return SOAP_OK;
}

int covise::WSCOVISEService::getConfigEntry(covise::_covise__getConfigEntry *request, covise::_covise__getConfigEntryResponse *response)
{
    response->value = covise::coConfig::getInstance()->getValue(QString::fromStdString(request->variable),
                                                                QString::fromStdString(request->section))
                          .toStdString();
    return SOAP_OK;
}

int covise::WSCOVISEService::deleteModule(covise::_covise__deleteModule *request, covise::_covise__deleteModuleResponse *response)
{
    (void)response;

    covise::WSMainHandler::instance()->deleteModule(QString::fromStdString(request->moduleID));

    return SOAP_OK;
}

int covise::WSCOVISEService::instantiateModule(_covise__instantiateModule *request, _covise__instantiateModuleResponse *response)
{
    (void)response;

    if (request->x && request->y)
        covise::WSMainHandler::instance()->instantiateModule(QString::fromStdString(request->module),
                                                             QString::fromStdString(request->host),
                                                             *(request->x), *(request->y));
    else
        covise::WSMainHandler::instance()->instantiateModule(QString::fromStdString(request->module),
                                                             QString::fromStdString(request->host));

    return SOAP_OK;
}

int covise::WSCOVISEService::link(_covise__link *request, _covise__linkResponse *response)
{
    (void)response;

    covise::WSMainHandler::instance()->link(QString::fromStdString(request->fromModule),
                                            QString::fromStdString(request->fromPort),
                                            QString::fromStdString(request->toModule),
                                            QString::fromStdString(request->toPort));

    return SOAP_OK;
}

int covise::WSCOVISEService::unlink(_covise__unlink *request, _covise__unlinkResponse *response)
{
    (void)response;

    covise::WSMainHandler::instance()->unlink(QString::fromStdString(request->linkID));

    return SOAP_OK;
}

int covise::WSCOVISEService::getLinks(_covise__getLinks *request, _covise__getLinksResponse *response)
{
    (void)response;
    return SOAP_OK;
}

// if the system variable "cloudStorageDir" does not exist, theb create user cloud storage dir: home + "owncloud"
std::string getUserCsDir()
{
    std::string csDir = "/ownCloud";
    bool result = true;

    std::string userCsDir = covise::coCoviseConfig::getEntry("cloudStorageDir", "System.WSInterface");

    if (userCsDir.empty())
    {
        QString homeDir = QDir::homePath();
        userCsDir = homeDir.toStdString() + csDir;
    }

    if (!QDir(QString::fromStdString(userCsDir)).exists())
    {
        // create direcrory recursively
        result = QDir().mkpath(QString::fromStdString(userCsDir));
    }

    if (!result)
    {
        std::cerr << "Error during get user cloud storage directory: " << userCsDir << endl;
        return "";
    }

    return userCsDir;
}

int covise::WSCOVISEService::getFileInfoList(covise::_covise__getFileInfoList *request, covise::_covise__getFileInfoListResponse *response)
{
    (void)response;

    std::string &filePath = request->path;

    std::string userCsDir = getUserCsDir();

    if (userCsDir.empty())
    {
        response->fileInfoList.clear();
        //TODO: return SOAP error
        return SOAP_OK;
    }

    std::string fDir = userCsDir + filePath;

    QDir dir(QString::fromStdString(fDir));

    if (!dir.exists())
    {
        // TODO: return error: directory does not exist
        std::cerr << "cloud storage directory does not exist." << fDir << endl;
        response->fileInfoList.clear();
    }
    else
    {
        std::cerr << "cloud storage directory exists." << userCsDir << endl;

        //QStringList fileNameList = dir.entryList(QDir::Files | QDir::Hidden);
        QFileInfoList fileInfoList = dir.entryInfoList(QDir::AllEntries, QDir::DirsFirst);

        response->fileInfoList.clear();

        covise::covise__FileInfo fInfo;

        //TODO: return cloud storage directory -> userCsDir

        foreach (QFileInfo fileInfo, fileInfoList)
        {
            fInfo.fileName = fileInfo.fileName().toStdString();
            fInfo.fileSize = fileInfo.size();
            fInfo.isDir = fileInfo.isDir();
            fInfo.fileDate = fileInfo.lastModified().toString("dd.MM.yyyy").toStdString();

            response->fileInfoList.push_back(fInfo);
        }
    }

    return SOAP_OK;
}

/*
int covise::WSCOVISEService::uploadFileBase64(covise::_covise__uploadFileBase64 *request, covise::_covise__uploadFileBase64Response *response)
{

    (void) response;

    std::string& path = request->path;
    std::string& fileName = request->fileName;
    const char* resource = reinterpret_cast<const char*>(request->resource.__ptr);
    int resourceSize = request->resource.__size;

    bool fileTrunc = request->fileTrunc;
    int fileSize = request->fileSize;

    bool lastChunk = false;

    bool result = true;

    std::string userCsDir = getUserCsDir();

    if (userCsDir.empty()) {
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    // upload file
    std::string uploadPath = userCsDir + path;

    if (!QDir(QString::fromStdString( uploadPath)).exists()) {
        result = QDir().mkpath(QString::fromStdString( uploadPath));
    }

    if (!result) {
        response->result = result;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    std::string uploadFileName = uploadPath + fileName;

    std::ofstream oFile;
    // If FileName already exists, its content is destroyed and the file becomes as new

    if (fileTrunc) {
       oFile.open(uploadFileName.c_str(), std::ios::trunc | std::ios::binary);
    } else {
       oFile.open(uploadFileName.c_str(), std::ios::app | std::ios::binary);
    }

    std::cerr << "file size: " << fileSize << endl;

    if (oFile.is_open()) {
       // write to outfile
       oFile.write(&resource[0],resourceSize);

       if (((size_t)oFile.tellp()) ==  fileSize) {
         lastChunk = true;
       }

       oFile.close();

       // The fileName will be send to covise in order to set the corresponding parameter
       response->lastChunk = lastChunk;
    } else {
      std::cerr << "Output file opening failed." << fileName << endl;     
      response->result = false;
      //TODO: return SOAP error
      return SOAP_OK;

    }

   response->result = result;
   return SOAP_OK;
}
*/

int covise::WSCOVISEService::isFileExist(covise::_covise__isFileExist *request, covise::_covise__isFileExistResponse *response)
{

    (void)response;

    std::string &path = request->path;
    std::string &fileName = request->fileName;

    bool isExist = false;

    std::string userCsDir = getUserCsDir();

    if (userCsDir.empty())
    {
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    if (path.empty() || fileName.empty())
    {
        response->result = false;
        return SOAP_OK;
    }

    std::string uploadedFileName = userCsDir + path + fileName;

    if (QFile::exists(QString::fromStdString(uploadedFileName)))
    {
        isExist = true;
    }

    response->result = true;
    response->isFileExist = isExist;

    return SOAP_OK;
}

int covise::WSCOVISEService::isDirExist(covise::_covise__isDirExist *request, covise::_covise__isDirExistResponse *response)
{

    (void)response;

    std::string &path = request->path;
    std::string &newDir = request->newDir;

    bool isExist = false;

    std::string userCsDir = getUserCsDir();

    if (userCsDir.empty())
    {
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    // upload file
    std::string newCsDir = userCsDir + path + newDir;

    if (path.empty() || newDir.empty())
    {
        response->result = false;
        return SOAP_OK;
    }

    if (QDir(QString::fromStdString(newCsDir)).exists())
    {
        isExist = true;
    }

    response->result = true;
    response->isDirExist = isExist;

    return SOAP_OK;
}

int covise::WSCOVISEService::uploadFile(covise::_covise__uploadFile *request, covise::_covise__uploadFileResponse *response)
{

    (void)response;

    std::string &path = request->path;
    std::string &fileName = request->fileName;
    const char *resource = reinterpret_cast<const char *>(request->resource.__ptr);
    int resourceSize = request->resource.__size;
    int chunkIndex = request->chunkIndex;
    int chunkNr = request->chunkNr;
    int chunkSize = request->chunkSize;
    int fileSize = request->fileSize;
    bool fileTruncated = request->fileTruncated;

    bool lastChunk = false;

    bool result = true;

    if (path.empty() && fileName.empty() && chunkSize == 0 && chunkNr == 0)
    {
        response->result = true;
        response->lastChunk = lastChunk;
        return SOAP_OK;
    }

    // upload file to directory: port, tmp
    /*
    std::string resourceDir = coCoviseConfig::getEntry("tmpDir", "System.WSInterface");

    if (resourceDir.empty()) {
 #ifdef _WIN32
     char* tempDir = getenv("TEMPDIR");
     if (tempDir) {
       resourceDir = tempDir;
         }
         else {
       resourceDir = "c:/temp";
         }
 #else
     resourceDir = "/var/tmp";
 #endif
    }

    int port = WSServer::instance()->getPort();
    std::stringstream portStr;
    portStr << port;
    resourceDir += "/covise_webservice/";
    resourceDir += portStr.str();
    resourceDir += "/";
    */

    // upload file to directory: ownCloud
    std::string userCsDir = getUserCsDir();

    if (userCsDir.empty())
    {
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    // upload file
    std::string uploadPath = userCsDir + path;

    if (!QDir(QString::fromStdString(uploadPath)).exists())
    {
        result = QDir().mkpath(QString::fromStdString(uploadPath));
    }

    if (!result)
    {
        response->result = result;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    std::string uploadFileName = uploadPath + fileName;

    std::ofstream oFile;

    std::cerr << "chunk index: " << chunkIndex << endl;

    if (fileTruncated)
    {
        // If FileName already exists, its content is destroyed and the file becomes as new
        oFile.open(uploadFileName.c_str(), std::ios::trunc | std::ios::binary);
    }
    else
    {
        oFile.open(uploadFileName.c_str(), std::ios::app | std::ios::binary);
    }

    if (oFile.is_open())
    {
        // sets the position where the next character is to be inserted into the output stream
        oFile.seekp(chunkIndex * chunkSize);

        if (chunkIndex < chunkNr - 1)
        {
            if (chunkSize == resourceSize)
            {
                // write to outfile
                oFile.write(resource, resourceSize);
            }
            else
            {
                std::cerr << "Error during upload: " << fileName << " chunk index: " << chunkIndex << endl;
                response->result = false;
            }
        }
        else if (chunkIndex == chunkNr - 1)
        {
            lastChunk = true;
            // write to outfile
            oFile.write(resource, resourceSize);
            if (((size_t)oFile.tellp()) != fileSize)
            {
                std::cerr << "Error during upload: " << fileName << " at last chunk: " << chunkNr << endl;
                response->result = false;
            }
        }
        else
        {
            std::cerr << "Error during upload: " << fileName << " chunk index: " << chunkNr << "chunk number: " << chunkNr << endl;
            response->result = false;
        }

        oFile.close();
        response->lastChunk = lastChunk;
    }
    else
    {
        std::cerr << "Output file opening failed." << fileName << endl;
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    response->result = result;
    return SOAP_OK;
}

int covise::WSCOVISEService::createNewDir(covise::_covise__createNewDir *request, covise::_covise__createNewDirResponse *response)
{

    (void)response;

    std::string &path = request->path;
    std::string &newDir = request->newDir;

    bool result = true;

    std::string userCsDir = getUserCsDir();

    if (userCsDir.empty())
    {
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    // upload file
    std::string newDirPath = userCsDir + path + newDir;

    if (!QDir(QString::fromStdString(newDirPath)).exists())
    {
        result = QDir().mkpath(QString::fromStdString(newDirPath));
    }

    response->result = result;

    return SOAP_OK;
}

bool removeDir(const QString &dirName)
{
    bool result = true;
    QDir dir(dirName);

    if (dir.exists(dirName))
    {
        Q_FOREACH (QFileInfo info, dir.entryInfoList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden | QDir::AllDirs | QDir::Files, QDir::DirsFirst))
        {
            if (info.isDir())
            {
                result = removeDir(info.absoluteFilePath());
            }
            else
            {
                result = QFile::remove(info.absoluteFilePath());
            }

            if (!result)
            {
                return result;
            }
        }
        result = dir.rmdir(dirName);
    }
    return result;
}

int covise::WSCOVISEService::deleteDir(covise::_covise__deleteDir *request, covise::_covise__deleteDirResponse *response)
{

    (void)response;

    std::string &path = request->path;

    std::string userCsDir = getUserCsDir();

    bool result = true;

    if (userCsDir.empty())
    {
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    const QString csPath = QString::fromStdString(userCsDir + path);

    if (QFileInfo(csPath).isFile())
    {
        result = QFile::remove(csPath);
    }
    else
    {
        result = removeDir(csPath);
    }

    response->result = result;
    return SOAP_OK;
}

// used for cloud storage
int covise::WSCOVISEService::setParameterFromUploadedFile(covise::_covise__setParameterFromUploadedFile *request, covise::_covise__setParameterFromUploadedFileResponse *response)
{
    (void)response;

    std::string &filePath = request->value;

    std::string userCsDir = getUserCsDir();

    if (userCsDir.empty())
    {
        //TODO: return SOAP error
        return SOAP_OK;
    }

    std::string csFilePath = userCsDir + filePath;

    QMutexLocker l(&sequentialLock);

    covise::WSMainHandler::instance()->setParameterFromString(QString::fromStdString(request->moduleID), QString::fromStdString(request->parameter), QString::fromStdString(csFilePath));

    return SOAP_OK;
}

// MTOM file upload
void *mime_write_open(struct soap *soap, const char *id, const char *type, const char *description, enum soap_mime_encoding encoding)
{

    // TODO: how get filename
    FILE *handle = fopen("uploadFileName", "wb");
    // We ignore the MIME content transfer encoding here, but should check
    if (!handle)
    {
        soap->error = SOAP_EOF;
        soap->errnum = errno; // get reason
    }
    return (void *)handle;
}

void mime_write_close(struct soap *soap, void *handle)
{
    fclose((FILE *)handle);
}

int mime_write(struct soap *soap, void *handle, const char *buf, size_t len)
{
    size_t nwritten;
    while (len)
    {
        nwritten = fwrite(buf, 1, len, (FILE *)handle);
        if (!nwritten)
        {
            soap->errnum = errno; // get reason
            return SOAP_EOF;
        }
        len -= nwritten;
        buf += nwritten;
    }
    return SOAP_OK;
}

int covise::WSCOVISEService::uploadFileMtom(covise::_covise__uploadFileMtom *request, covise::_covise__uploadFileMtomResponse *response)
{

    (void)response;

    std::string &path = request->path;
    std::string &fileName = request->fileName;
    covise::covise__FileData fileData = request->fileData;
    //fileData.xop__Include.type = "text/html";
    fileData.xop__Include.type = "application/octet-stream";
    //fileData.xmime5__contentType = "text/html";
    fileData.xmime5__contentType = "application/octet-stream";
    const char *resource = reinterpret_cast<const char *>(fileData.xop__Include.__ptr);
    int resourceSize = fileData.xop__Include.__size;

    bool fileTrunc = request->fileTrunc;
    int fileSize = request->fileSize;

    bool lastChunk = false;
    bool result = true;

    std::string userCsDir = getUserCsDir();

    if (userCsDir.empty())
    {
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    // upload file
    std::string uploadPath = userCsDir + path;

    if (!QDir(QString::fromStdString(uploadPath)).exists())
    {
        result = QDir().mkpath(QString::fromStdString(uploadPath));
    }

    if (!result)
    {
        response->result = result;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    std::string uploadFileName = uploadPath + fileName;

    /*
    struct soap& soap = *((struct soap *)this);

    // TODO: not run
    soap.fmimewriteopen = mime_write_open;
    soap.fmimewriteclose = mime_write_close;
    soap.fmimewrite = mime_write;
    */

    std::ofstream oFile;
    // If FileName already exists, its content is destroyed and the file becomes as new

    if (fileTrunc)
    {
        oFile.open(uploadFileName.c_str(), std::ios::trunc | std::ios::binary);
    }
    else
    {
        oFile.open(uploadFileName.c_str(), std::ios::app | std::ios::binary);
    }

    std::cerr << "file size: " << fileSize << endl;

    if (oFile.is_open())
    {
        // write to outfile
        oFile.write(resource, resourceSize);

        if (((size_t)oFile.tellp()) == fileSize)
        {
            lastChunk = true;
        }

        oFile.close();

        // The fileName will be send to covise in order to set the corresponding parameter
        response->lastChunk = lastChunk;
    }
    else
    {
        std::cerr << "Output file opening failed." << fileName << endl;
        response->result = false;
        //TODO: return SOAP error
        return SOAP_OK;
    }

    response->result = result;
    return SOAP_OK;
}
