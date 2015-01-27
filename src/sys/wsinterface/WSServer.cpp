/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSServer.h"

#include "covise.nsmap"
//SOAP_NMAC struct Namespace * namespaces = covise_namespaces;

#include "WSMessageHandler.h"
#include "WSMainHandler.h"
#include "WSMap.h"
#include "WSModule.h"
#include "WSColormapChoiceParameter.h"
#include "WSCoviseStub.h"

#include <iostream>

#include <QTextStream>
#include <QFile>
#include <QThreadPool>
#include <QDir>

#include <QProcess>
#include <QRegExp>
#include <QImage>
#include <QLinearGradient>
#include <QBuffer>
#include <QPainter>

#include <config/coConfig.h>

using std::cerr;
using std::endl;

QString covise::WSServer::coviseDir;
covise::WSServer *covise::WSServer::instancePtr = 0;

covise::WSServer::WSServer()
{
    this->port = covise::coConfig::getInstance()->getInt("port", "System.WSInterface", 31111);
    this->fget = covise::WSServer::httpGet;
    this->bind_flags = SO_REUSEADDR;
    instancePtr = this;

    QStringList env = QProcess::systemEnvironment();
    QRegExp coviseDirRegExp = QRegExp("^COVISEDIR=(.*)", Qt::CaseInsensitive);
    this->coviseDir = env.filter(coviseDirRegExp).first();
    coviseDirRegExp.indexIn(this->coviseDir);
    this->coviseDir = coviseDirRegExp.cap(1);

    start();
}

covise::WSServer::~WSServer()
{
    terminate();
}

covise::WSServer *covise::WSServer::instance()
{
    return instancePtr;
}

int covise::WSServer::getPort() const
{
    return this->port;
}

void covise::WSServer::run()
{

    QThreadPool threadPool;
    threadPool.setMaxThreadCount(100);

    this->send_timeout = 60; // 60 seconds
    this->recv_timeout = 60; // 60 seconds

    cerr << "WSServer::run() info: binding to port " << this->port << endl;

    SOAP_SOCKET masterSocket, slaveSocket;
    masterSocket = bind(0, this->port, 100);

    //bool isPortChanged = false;

    while (!soap_valid_socket(masterSocket))
    {
        cerr << "WSServer::Internal::run() info: cannot bind to port " << this->port << endl;
        ++port;
        masterSocket = bind(0, this->port, 100);
        //isPortChanged = true;
    }

    cerr << "WSServer::run() info: bound to port " << this->port << endl;

    //if (isPortChanged) acws->portChangedFromInternal(this->port);

    while (true)
    {
        slaveSocket = accept();
        if (!soap_valid_socket(slaveSocket))
        {
            if (this->errnum)
            {
                cerr << "WSServer::run() err: SOAP error:" << endl;
                ::soap_print_fault(this, stderr);
                exit(1); // TODO Handle gracefully
            }
            cerr << "WSServer::run() err: server timed out" << endl;
            exit(1); // TODO Handle gracefully
        }
#if 1 // Multithreaded
        covise::WSServer::Thread *t = new covise::WSServer::Thread(this);
        threadPool.start(t);
#else // Single threaded
        serve();
#endif
    }
}

int covise::WSServer::run(int value)
{
    return WSCOVISEService::run(value);
}

int covise::WSServer::httpGet(soap *soap)
{
    QString path = soap->path;
    if (path.startsWith("/"))
        path.remove(0, 1);
    if (path.contains('?'))
    {
        path.truncate(path.indexOf('?'));
    }

    QString servePath = WSServer::coviseDir + "/share/covise/web";
    //QString servePath = ":/web";
    QFile serveFile(servePath + "/" + path);

    if (path != "clientaccesspolicy.xml" && path != "COVISE.wsdl" && !serveFile.exists() && !path.startsWith("colormap/"))

        return SOAP_GET_METHOD;

    if (path.endsWith(".xml") || path.endsWith("wsdl"))
        soap->http_content = "text/xml"; // HTTP header with text/xml content
    else if (path.endsWith(".html"))
        soap->http_content = "text/html";
    else if (path.endsWith(".js"))
        soap->http_content = "text/javascript";
    else if (path.endsWith(".css"))
        soap->http_content = "text/css";
    else if (path.endsWith(".png"))
        soap->http_content = "image/png";
    else if (path.endsWith(".jpg"))
        soap->http_content = "image/jpeg";

    soap_response(soap, SOAP_FILE);

    // Return the client access policy on request that certain M$ consumers need
    if (path == "clientaccesspolicy.xml")
    {
        strncpy(soap->tmpbuf,
                "<?xml version=\"1.0\" encoding=\"utf-8\"?><access-policy><cross-domain-access><policy><allow-from http-request-headers=\"*\"><domain uri=\"*\"/></allow-from><grant-to><resource path=\"/\" include-subpaths=\"true\"/></grant-to></policy></cross-domain-access></access-policy>",
                1023);
        soap_send_raw(soap, soap->tmpbuf, strlen(soap->tmpbuf));
    }
    // Make WSDL available as /COVISE.wsdl
    else if (path == "COVISE.wsdl")
    {
        QFile f(":/WS/COVISE.wsdl");
        f.open(QIODevice::ReadOnly);

        for (int bytesToSend = f.read(soap->tmpbuf, 1024); bytesToSend > 0; bytesToSend = f.read(soap->tmpbuf, 1024))
        {
            soap_send_raw(soap, soap->tmpbuf, bytesToSend);
        }
    }
    else if (path.startsWith("colormap/") && path.endsWith(".png"))
    {
        QString paramID = path.mid(9);
        paramID.chop(4);
        QStringList components = paramID.split("_");

        if (!(components.size() > 6))
            return SOAP_GET_METHOD;

        QString moduleID = components[0] + "_" + components[1] + "_" + components[2] + "." + components[3] + "." + components[4] + "." + components[5];
        QString parameterID = components[6];
        for (int ctr = 7; ctr < components.size(); ++ctr)
        {
            parameterID += "_" + components[ctr];
        }

        // Check for selection indicator
        components = parameterID.split('/');
        parameterID = components[0];

        covise::WSModule *module = covise::WSMainHandler::instance()->getMap()->getModule(moduleID);

        if (module == 0)
            return SOAP_GET_METHOD;

        covise::WSParameter *parameter = module->getParameter(parameterID);

        if (parameter == 0)
            return SOAP_GET_METHOD;

        QImage colormapImage(256, 48, QImage::Format_ARGB32);
        covise::WSColormap cm;

        if (parameter->getType() == "ColormapChoice")
        {

            if (components.size() < 2)
                return SOAP_GET_METHOD;

            int selection = components[1].toInt();

            covise::WSColormapChoiceParameter *cc = qobject_cast<covise::WSColormapChoiceParameter *>(parameter);

            QList<covise::WSColormap> cms = cc->getValue();

            if (cms.size() <= selection)
                return SOAP_GET_METHOD;

            cm = cms[selection];
        }
        else
        {
            return SOAP_GET_METHOD;
        }

        QPainter painter(&colormapImage);

        // draw checker board
        painter.fillRect(0, 0, colormapImage.width(), colormapImage.height(), QBrush(QImage(":/web/checker.xpm")));

        // loop over all interpolation markers
        for (int ctr = 0; ctr < cm.getPins().size() - 1; ++ctr)
        {
            // set a gradientpoint
            covise::WSColormapPin left = cm.getPins()[ctr];
            covise::WSColormapPin right = cm.getPins()[ctr + 1];

            int xleft = int(left.position * colormapImage.width());
            int xright = int(right.position * colormapImage.width());

            QLinearGradient lgrad(xleft, 0.f, xright, 0.f);

            QColor c1 = QColor::fromRgbF(left.r, left.g, left.b, left.a);
            lgrad.setColorAt(0.f, c1);

            QColor c2 = QColor::fromRgbF(right.r, right.g, right.b, right.a);
            lgrad.setColorAt(1.f, c2);

            painter.fillRect(xleft, 0, (xright - xleft + 1), colormapImage.height(), QBrush(lgrad));
        }

        QByteArray ba;
        QBuffer buffer(&ba);
        buffer.open(QIODevice::ReadWrite);
        colormapImage.save(&buffer, "PNG"); // writes image into ba in PNG format
        buffer.reset();

        for (int bytesToSend = buffer.read(soap->tmpbuf, 1024);
             bytesToSend > 0;
             bytesToSend = buffer.read(soap->tmpbuf, 1024))
        {
            soap_send_raw(soap, soap->tmpbuf, bytesToSend);
        }

        buffer.close();
    }
    else
    {
        serveFile.open(QIODevice::ReadOnly);

        for (int bytesToSend = serveFile.read(soap->tmpbuf, 1024); bytesToSend > 0; bytesToSend = serveFile.read(soap->tmpbuf, 1024))
        {
            soap_send_raw(soap, soap->tmpbuf, bytesToSend);
        }
    }

    soap_end_send(soap);
    return SOAP_OK;
}

covise::WSServer::Thread::Thread(WSCOVISEService *service)
{
    this->service = service->copy();
}

void covise::WSServer::Thread::run()
{
    this->service->serve();
    delete this->service;
}
