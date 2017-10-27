/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <openssl/err.h>
#include <config/CoviseConfig.h>
#include <config/coConfigEntryString.h>

#include <string>
#include <QApplication>
#include <QListWidget>
#include <QInputDialog>
#include <net/covise_socket.h>
#include <net/tokenbuffer.h>
#include <iostream>
#include <sstream>
#include <VRBClientList.h>
#include "string_utils.h"
#include "SSLDaemon.h"

#ifndef WIN32
#define strncpy_s strncpy
#endif

using namespace std;
using namespace ::covise;

SSLDaemon::SSLDaemon(frmMainWindow *window)
{

    mDebugFile = NULL;
    mFile = NULL;
    mSBuf = NULL;
    mWindow = window;
    mRequest = new frmRequestDialog();
    mRequest->setupUi(mRequest);
    mRequest->setDaemon(this);
    mIsAllowed = false;
    this->mLogInternal = NULL;

    mNetworkpollIntervall = coCoviseConfig::getFloat("System.CoviseDaemon.Poll", 0.1f);

    if (coCoviseConfig::isOn("System.CoviseDaemon.Debug", false))
    {
        if (coCoviseConfig::isOn("System.CoviseDaemon.EnableFileDebug", false))
        {
            std::string line = coCoviseConfig::getEntry("System.CoviseDaemon.DebugFile");
            if (!line.empty())
            {
                cerr << "SSLDaemon::SSLDaemon(): Value of debug-file: " << line << "\n";
            }
            else
            {
                cerr << "SSLDaemon::SSLDaemon(): No config entry for debugFile. Using hardcoded!" << std::endl;
                line = "CoviseSSLDaemon.log";
            }

            cerr << "SSLDaemon::SSLDaemon(): Used debug-file: " << line << std::endl;
            mFile = new ofstream();
            mFile->open(line.c_str(), ios::out);
            mSBuf = std::cerr.rdbuf();
            std::cerr.rdbuf(mFile->rdbuf());
        }
        else if (coCoviseConfig::isOn("System.CoviseDaemon.EnableInternalDebug", false))
        {
            cerr << "SSLDaemon::SSLDaemon(): Don't use debug file! Dump to GUI-Listview!" << std::endl;
            cerr << "SSLDaemon::SSLDaemon(): Only use with -g!" << std::endl;
            mLogInternal = new std::stringstream();
            mSBuf = std::cerr.rdbuf();
            std::cerr.rdbuf(mLogInternal->rdbuf());
        }
    }
    else
    {
        cerr << "SSLDaemon::SSLDaemon(): Dump to NULL stream!" << std::endl;
        cerr << "Use config file to enable debugging to console or file!" << std::endl;
        mFile = new ofstream();
#ifdef WIN32
        mFile->open("nul", ios::out);
#else
        mFile->open("/dev/null", ios::out);
#endif
        mSBuf = std::cerr.rdbuf();
        std::cerr.rdbuf(mFile->rdbuf());
    }

    //Creating base Config objects for storing personal SSLDaemon settings
    cerr << "SSLDaemon::SSLDaemon(): Preparing basic setup for personal settings storage!" << std::endl;
    mConfig = new coConfigGroup("SSLDaemon");
    mConfig->addConfig(coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "ssldaemon.xml", "local", true);
    coConfig::getInstance()->addConfig(mConfig);

    coConfigEntryStringList list = coConfig::getInstance()->getScopeList("System.CoviseDaemon.HostACL");
    coConfigEntryStringList::iterator listentry;

    //Get ACL for Hosts
    listentry = list.begin();
    cerr << "SSLDaemon::SSLDaemon(): Size of list =  " << list.count() << std::endl;
    while (listentry != list.end() && (!list.empty()))
    {
        cerr << "SSLDaemon::SSLDaemon(): " << (*listentry).toStdString() << std::endl;
        QString value = coConfig::getInstance()->getString("hostname", QString("System.CoviseDaemon.HostACL.") + (*listentry), "");
        cerr << "SSLDaemon::SSLDaemon(): Hostname = " << value.toStdString() << std::endl;
        mHostList.push_back(value.toStdString());
        listentry++;
    }

    //Get UID whitelist
    list = coConfig::getInstance()->getScopeList("System.CoviseDaemon.AllowedUID");

    //Get ACL for SubjectUIDs
    listentry = list.begin();
    cerr << "SSLDaemon::SSLDaemon(): Size of list =  " << list.count() << std::endl;
    while (listentry != list.end() && (!list.empty()))
    {
        std::string dump = (*listentry).toStdString();
        cerr << "SSLDaemon::SSLDaemon(): " << (*listentry).toStdString() << std::endl;
        //QString value = coConfig::getInstance()->getString("UID",QString("System.CoviseDaemon.AllowedUID.") + (*listentry),"");
        //cerr << "SSLDaemon::SSLDaemon(): Hostname = "<< value.toStdString() << std::endl;
        QString name = coConfig::getInstance()->getString("Name", QString("System.CoviseDaemon.AllowedUID.") + (*listentry), "");
        //mSubjectList.push_back(value.toStdString());
        mSubjectNameList.push_back(name.toStdString());
        listentry++;
    }
    mbCertCheck = coCoviseConfig::isOn("System.CoviseDaemon.EnableCertificateCheck", false);

    cerr << "SSLDaemon::SSLDaemon(): Initialize Controller to NULL..." << std::endl;
    mController = NULL;

    mPort = 31090;
    cerr << "SSLDaemon::SSLDaemon(): Default port " << mPort << "..." << std::endl;
    cerr << "SSLDaemon::SSLDaemon(): Determine TCP-Port from Covise Config..." << std::endl;
    mPort = coCoviseConfig::getInt("port", "System.CoviseDaemon.Server", mPort);

    cerr << "SSLDaemon::SSLDaemon(): TCP-Port is " << mPort << std::endl;
    cerr << "SSLDaemon::SSLDaemon(): Finished constructor!" << std::endl;

    mbConfirmed = false;

#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif
}

std::string SSLDaemon::ToString(int value)
{
    std::stringstream strm;
    strm << value;
    return strm.str();
}

SSLDaemon::~SSLDaemon(void)
{
    cerr << "SSLDaemon::~SSLDaemon(): server connection..." << std::endl;
    if (mSBuf)
    {
        std::cerr.rdbuf(mSBuf);
    }
    cerr << "SSLDaemon::~SSLDaemon(): Clearing memory from objects!" << std::endl;
    if (mFile)
    {
        delete mFile;
        mFile = NULL;
    }
    cerr << "SSLDaemon::~SSLDaemon(): Deleted debugFile name variable!" << std::endl;
    if (mSSLConn)
    {
        delete mSSLConn;
        mSSLConn = NULL;
    }

    if (mRequest)
    {
        cerr << "SSLDaemon::~SSLDaemon(): Deleted User request dialog!" << std::endl;
        delete mRequest;
        mRequest = NULL;
    }

    if (mConfig)
    {
        cerr << "SSLDaemon::~SSLDaemon(): Deleted config objects!" << std::endl;
        coConfig::getInstance()->removeConfig("SSLDaemon");
        delete mConfig;
        mConfig = NULL;
    }

    cerr << "SSLDaemon::~SSLDaemon(): closed Server connection" << std::endl;
    cerr << "SSLDaemon::~SSLDaemon(): Done!" << std::endl;
}

bool SSLDaemon::openServer()
{
    cerr << "SSLDaemon::openServer(): Create new server connection..." << std::endl;
    mSSLConn = new SSLServerConnection(mPort, 0, (sender_type)0, sslPasswdCallback, this);

    //check for valid SimpleServerConnection object

    if (!mSSLConn->getSocket())
    {
        cerr << "SSLDaemon::openServer(): Creation of server failed!" << std::endl;
        cerr << "SSLDaemon::openServer(): Port-Binding failed! Port already bound?" << std::endl;
        return false;
    }

    //   struct linger linger;
    //   linger.l_onoff = 0;
    //   linger.l_linger = 0;
    cerr << "SSLDaemon::openServer(): Set socket options..." << std::endl;

    cerr << "SSLDaemon::openServer(): Set server to listen mode..." << std::endl;
    mSSLConn->listen();
    if (!mSSLConn->is_connected()) // could not open server port
    {
        fprintf(stderr, "SSLDaemon::openServer(): Could not open server port %d\n", mPort);
        delete mSSLConn;
        mSSLConn = NULL;
        return false;
    }
    cerr << "SSLDaemon::openServer(): Add server connection to connection list..." << std::endl;
    mConnections = new ConnectionList();
    mConnections->add(mSSLConn);
    cerr << "SSLDaemon::openServer(): adding" << mSSLConn << std::endl;

    cerr << "SSLDaemon::openServer(): Server opened!" << std::endl;

    return 0;
}

void SSLDaemon::closeServer()
{
    do
    {
        cerr << "SSLDaemon::closeServer(): SSLDaemon still running...!" << std::endl;
    } while (mIsRunning);

    mConnections->remove(mSSLConn);
    cerr << "SSLDaemon::closeServer(): Remove server connection from list!" << std::endl;

    if (mConnections)
    {
        delete mConnections;
        cerr << "SSLDaemon::closeServer(): Deleted connection list!" << std::endl;
        mConnections = NULL;
    }

    if (mSSLConn)
    {
        delete mSSLConn;
        cerr << "SSLDaemon::closeServer(): Deleted SSL server-connection!" << std::endl;
        mSSLConn = NULL;
    }
    cerr << "SSLDaemon::closeServer(): Server closed...!" << std::endl;
}

bool SSLDaemon::run()
{
    cerr << "SSLDaemon::run(): Creating network notifier!" << std::endl;
    /*QSocketNotifier* locNotifier = new QSocketNotifier(mSSLConn->get_id(NULL),QSocketNotifier::Read);
   mNotifier[mSSLConn] = locNotifier;*/
    /*QObject::connect( locNotifier, SIGNAL(activated(int)), this,  SLOT(processMessagesLegacy()));*/
    processMessagesLegacy();
    return true;
}

void SSLDaemon::stop()
{
    std::map<SSLServerConnection *, QSocketNotifier *>::iterator iter;
    cerr << "SSLDaemon::stop(): Destroying network notifier!" << std::endl;
    if (!mNotifier.empty())
    {
        iter = mNotifier.begin();
        while (iter != mNotifier.end())
        {
            iter->second->setEnabled(false);
            iter->second->disconnect(iter->second, SIGNAL(activated(int)), this, SLOT(processMessagesLegacy()));
            delete iter->second;
            iter++;
        }
        mNotifier.clear();
    }
    mIsRunning = false;
}

void SSLDaemon::setWindow(frmMainWindow *window)
{
    mWindow = window;
    mWindow->setPort(mPort);

    //Populate host list with config file entries
    for (unsigned int i = 0; i < mHostList.size(); i++)
    {
        window->getHostList()->addItem(QString(mHostList.at(i).c_str()));
    }

    //Populate subject list with allowed subject names.
    for (unsigned int i = 0; i < mSubjectNameList.size(); i++)
    {
        window->getUserList()->addItem(QString(mSubjectNameList.at(i).c_str()));
    }
}

void SSLDaemon::processMessagesLegacy()
{
    mIsRunning = true;
    Connection *locConn = NULL;
    cerr << "SSLDaemon::processMessagesLegacy():Entered SSLDaemon::processMessages()..." << std::endl;
    cerr << "SSLDaemon::processMessagesLegacy():Waiting for data ..." << std::endl;
    while (mIsRunning)
    {
        if ((locConn = mConnections->check_for_input(mNetworkpollIntervall)))
        {
            //Spawn new SSL Connection
            cerr << "SSLDaemon::processMessagesLegacy():Received data ..." << std::endl;
            SSLConnection *locSSLConn = dynamic_cast<SSLConnection *>(locConn);
            if (locSSLConn)
            {
                if (mSSLConn == locSSLConn)
                {
                    // Connect on server port --> Spawn new connection to client
                    cerr << "SSLDaemon::processMessagesLegacy():Connect on SSL server-port!" << std::endl;
                    SSLServerConnection *newSSLConn = mSSLConn->spawnConnection();

                    //Check whether Host is allowed
                    cerr << "SSLDaemon::processMessagesLegacy(): Checking for valid access!" << std::endl;

                    mCurrentPeer = newSSLConn->getPeerAddress();

                    for (unsigned int i = 0; i < mHostList.size(); i++)
                    {
                        int result = mCurrentPeer.compare(mHostList.at(i));
                        if (result == 0)
                        {
                            mIsAllowed = true;
                            break;
                        }
                    }

                    //Query for unkwon machine connect
                    if (mHostList.empty() || (!mIsAllowed))
                    {
                        cerr << "SSLDaemon::processMessagesLegacy(): Host is not allowed to connect! Add notification and access request for user here!" << std::endl;
                        mIsAllowed = false;
                        //Query for acceptance based message dialog
                        std::string message = " The computer at " + mCurrentPeer;
                        message += " wants to connect to your computer, but is not in your list of allowed computers.";
                        message += " Do you want to allow this?";
                        mRequest->setMessage(message.c_str(), frmRequestDialog::MachineLevel);
                        mRequest->show();
                        do
                        {
                            QApplication::processEvents(QEventLoop::AllEvents);
                        } while (!mbConfirmed);
                        mbConfirmed = false;
                    }

                    if (mIsAllowed)
                    {
                        cerr << "SSLDaemon::processMessagesLegacy(): Connecting peer: " << newSSLConn->getPeerAddress().c_str()
                             << " is in whitelist and allowed to connect!" << std::endl;
                        if (!newSSLConn)
                        {
                            cerr << "SSLDaemon::processMessagesLegacy():Creation of new SSL connection failed!" << std::endl;
                            return;
                        }
                        if (newSSLConn->AttachSSLToSocket(newSSLConn->getSocket()) == 0)
                        {
                            cerr << "SSLDaemon::processMessagesLegacy():SSL-Attach failed!" << std::endl;
                            ERR_print_errors_fp(stderr);
                            return;
                        }

                        cerr << "SSLDaemon::processMessagesLegacy(): Waiting to accept SSL conn!" << std::endl;
                        int err = 0;
                        int retries = 30;
                        int sslerr = 0;
                        do
                        {
                            err = 0;
                            if ((sslerr = newSSLConn->accept()) <= 0)
                            {
#ifdef WIN32
                                if ((err = WSAGetLastError()) != 10035)
                                {
                                    cerr << "SSLDaemon::processMessagesLegacy(): SSL_accept failed with err = " << err << std::endl;
                                    break;
                                }
#endif
                            }
                            else
                            {
                                cerr << "SSLDaemon::processMessagesLegacy(): Last error = " << sslerr << std::endl;
                                break;
                            }
                            cerr << " Retry: #" << retries << std::endl;
#ifdef WIN32
                            Sleep(500);
#else
                            sleep(1);
#endif
                            retries--;
                        } while ((sslerr != 1) && (retries > 0));

                        mIsAllowed = false;

                        //Add certificate evaluation here
                        if ((!this->isClientValid(newSSLConn)) && mbCertCheck)
                        {
                            std::string message;

                            mSubjectUID = newSSLConn->getSSLSubjectName();

                            cerr << "SSLDaemon::processMessagesLegacy(): Subject = !" << mSubjectUID << std::endl;

                            if (mSubjectUID == "")
                            {
                                message = " An unkown user";
                            }
                            else
                            {
                                message = " The user " + mSubjectUID;
                            }
                            message += " wants to issue a command to your COVISE instance, but is not in your list of allowed subjects.";
                            message += " Do you want to allow this?";
                            mRequest->setMessage(message.c_str(), frmRequestDialog::SubjectLevel);
                            mbConfirmed = false;
                            mRequest->show();
                            do
                            {
                                QApplication::processEvents(QEventLoop::AllEvents);
                            } while (!mbConfirmed);
                        }
                        else
                        {
                            // Either no checking active or subject in whitelist
                            mIsAllowed = true;
                        }

                        if (mIsAllowed)
                        {

                            if (sslerr <= 0)
                            {
                                resolveError();
                                cerr << "SSLDaemon::processMessagesLegacy(): " << err << std::endl;
                                cerr << "SSLDaemon::processMessagesLegacy(): SSL_Accept failed!" << std::endl;
                                return;
                            }
                            else
                            {
                                resolveError();
                                checkSSLError(newSSLConn->mSSL, sslerr);
                            }

                            //add new connection to a list of connections for further reference
                            VRBSClient *client = new VRBSClient(newSSLConn, NULL);
                            cerr << "SSLDaemon::processMessagesLegacy(): Client location: " << client->getIP() << std::endl;
                            mConnections->add(newSSLConn);
                        }
                    }
                    else
                    {
                        //Not allowed --> close connection
                        newSSLConn->close();
                    }
                }
                else
                {
                    // Incoming on client connection
                    // Check for access --> IP --> User(Certificate)
                    // if access granted --> issue command of second data packet
                    cerr << "SSLDaemon::processMessagesLegacy():Data on SSL client-connection! Expecting command..." << std::endl;

                    //Skip Certs for now

                    if (locSSLConn != mController)
                    {
                        //Check for shutdown notice

                        //Parse string-based commands
                        parseCommands(locSSLConn);
                    }
                    else
                    {
                        // Controller connection handling
                        //Check for known commands
                        Message request;
                        int result_code = locSSLConn->recv_msg(&request);
                        if (result_code < 0)
                        {
                            //Houston we have a problem
                            cerr << "SSLDaemon::processMessagesLegacy(): Error while receiving message!" << std::endl;
                            this->mConnections->remove(locSSLConn);
                            delete locSSLConn;
                        }
                        else if (result_code == 0)
                        {
                            //Check for graceful abort
                            //int ret = SSL_get_error();
                            //cerr << "SSLERROR: = " << ret << std::endl;
                        }
                        else
                        {
                            // Analyze message
                            handleMessage(request);
                        }
                    }
                }
                cerr << "SSLDaemon::processMessagesLegacy(): Processing events" << std::endl;
            }
        }
        mIsAllowed = false;
        QApplication::processEvents(QEventLoop::AllEvents);
        if (mWindow && mLogInternal)
        {
            updateLog();
        }
    }
    cerr << "... exiting SSLDaemon::processMessagesLegacy()!" << std::endl;
}

void SSLDaemon::updateLog()
{
    QTextEdit *locLog = mWindow->getLog();
    locLog->clear();
    QString messages = QString::fromStdString(this->mLogInternal->str());
    locLog->append(messages);
    locLog->update();
}

void SSLDaemon::handleMessage(Message &msg)
{
    TokenBuffer tb(&msg);
    switch (msg.type)
    {
    case COVISE_MESSAGE_VRB_REQUEST_FILE:
    {
    }
    break;
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    {
        cerr << "SSLDaemon::handleMessage():remove" << msg.conn;
        if (msg.conn == mController)
        {
            mController->close();
            mConnections->remove(mController);
            mController = NULL;
        }

        if (mAG)
        {
            mAG->getSocket()->write("masterLeft", (unsigned int)(strlen("masterLeft")) + 1);
        }
        cerr << "SSLDaemon::handleMessage(): controller left" << std::endl;
    }
    break;
    default:
    {
        cerr << "SSLDaemon::handleMessage(): Unknown request" << std::endl;
    }
    }
}

void SSLDaemon::parseCommands(SSLConnection *conn)
{

    SSLServerConnection *locConn = dynamic_cast<SSLServerConnection *>(conn);
    std::string line = locConn->readLine();

    if (line.compare(std::string("SSLConnectionClosed")) == 0)
    {
        cerr << "SSLDaemon::parseCommands(): SSLConnection closed by Client!" << std::endl;
        //Remove connection
        this->mConnections->remove(conn);
        delete conn;
        return;
    }

    //Branch dependent on Command

    if (line.compare(std::string("ConnectionClosed")) == 0)
    {
        //Remove connection from list of active connections
        this->mConnections->remove(conn);

        //Cast to appropriate SSLConnection and call destructor
        //Implicitly performs a graceful shutdown: ~SSLServerConnection()

        delete conn;
        return;
    }
    if (std::string("quit").compare(line.substr(0, 4)) == 0)
    {
        //Close connection, however RemoteDaemon so far only implemented this
        //as a return statement
        // Most likely shutdown SSLServer
        this->mWindow->handleOnOff(false);
        return;
    }
    if (std::string("check").compare(line.substr(0, 5)) == 0)
    {
        //Determine Controller connection status and send
        //status
        if (mController == NULL)
        {
            locConn->send("masterLeft\n", (unsigned int)strlen("masterLeft") + 1);
        }
        else
        {
            locConn->send("masterRunning\n", (unsigned int)strlen("masterRunning\n") + 1);
        }
        locConn->send("ACK\n", 4);
        return;
    }
    if (std::string("join").compare(line.substr(0, 4)) == 0)
    {
        //Sending join request from AccessGrid service
        //TODO: Rewrite AG service to use SSL for connecting to CoviseDaemon
        //Format: "join <host>:<port>"

        if (mController == NULL)
        {
            startCovise();
            mAG = locConn;
        }
        Message *msg = new Message;
        msg->type = COVISE_MESSAGE_SSLDAEMON;
        msg->data = (char *)line.c_str();
        msg->length = (int)line.size() + 1;
        mController->send_msg(msg);
        msg->data = NULL;
        delete msg;
        locConn->send("ACK\n", 4);
    }
    if (std::string("startCRB").compare(line.substr(0, 8)) == 0 || std::string("startCrb").compare(line.substr(0, 8)) == 0)
    {

        //Split line commands
        std::vector<std::string> strOptList;
        StringToVector(line, strOptList);
        //Spawn project with given arguments
        strOptList.erase(strOptList.begin());
        strOptList.pop_back();
        this->spawnProcesses(strOptList.at(0), strOptList);
        locConn->send("ACK\n", 4);
    }
#ifndef WIN32
    if (std::string("rebootClient").compare(line.substr(0, 12)) == 0)
    {
        //Split line commands
        std::vector<std::string> strOptList;
        StringToVector(line, strOptList);
        strOptList.at(0) = "RemoteRebootSlave";
        //Spawn process
        this->spawnProcesses(strOptList.at(0), strOptList);
        locConn->send("ACK\n", 4);
    }
    if (std::string("cleanCovise").compare(line.substr(0, 11)) == 0)
    {
        //Split line commands
        std::vector<std::string> strOptList;
        strOptList.push_back(std::string("clean_covise"));
        //Spawn process
        this->spawnProcesses(strOptList.at(0), strOptList);
        locConn->send("ACK\n", 4);
    }
#endif
    if (std::string("startCovise").compare(line.substr(0, 11)) == 0)
    {
        cerr << "SSLDaemon::parseCommands(): Command to start covise detected!" << std::endl;
        mAG = locConn;
        this->startCovise();
        locConn->send("ACK\n", 4);
    }
    if (std::string("startOpenCover").compare(line.substr(0, 14)) == 0)
    {
        cerr << "SSLDaemon::parseCommands(): Command to start OpenCover detected!" << std::endl;
        cerr << "SSLDaemon::parseCommands(): Command to call is: " << line.c_str() << std::endl;

        //Split line commands
        std::vector<std::string> strOptList;
        StringToVector(line, strOptList);
        strOptList.at(0) = "opencover";
        strOptList.push_back(strOptList.at(0));
        //Spawn process
        this->spawnProcesses(strOptList.at(0), strOptList);

        locConn->send("ACK\n", 4);
    }
    if (std::string("startFEN").compare(line.substr(0, 8)) == 0)
    {
        cerr << "SSLDaemon::parseCommands(): Command to start FEN detected!" << std::endl;
        cerr << "SSLDaemon::parseCommands(): Command to call is: " << line << std::endl;

        //Split line commands
        std::vector<std::string> strOptList;
        StringToVector(line, strOptList);
        strOptList.erase(strOptList.begin());
        //Spawn Fenfloss
        this->spawnProcesses(strOptList.at(0), strOptList);
    }
    else
    {
        //Unkown command
        cerr << "SSLDaemon::parseCommands(): Command unknown" << std::endl;
        cerr << "SSLDaemon::parseCommands(): Command is: " << line.c_str() << std::endl;
    }
}

void SSLDaemon::startCovise()
{
    int sPort = 0;
    //Create SSL connection to Controller
    //mController = new SimpleServerConnection(&sPort,0,(sender_type)0);
    mController = new SSLServerConnection(&sPort, 0, (sender_type)0, sslPasswdCallback, this);
    mController->listen();

    //Create argumentlist
    std::vector<std::string> optlist;
    optlist.push_back("covise");
    optlist.push_back(std::string("-r"));
    optlist.push_back(ToString(sPort));

    //Spawn process --> start Covise
    this->spawnProcesses(std::string("covise"), optlist);

    // Accept controller connection
    mController->sock_accept();
    mController->AttachSSLToSocket(mController->getSocket());

    int retries = 30;
    int sslerr = 0;
    do
    {
        if ((sslerr = mController->accept()) <= 0)
        {
#ifdef WIN32
            int err = WSAGetLastError();
            if (err != 10035)
            {
                cerr << "SSLDaemon::processMessagesLegacy(): SSL_accept failed with err = " << err << std::endl;
                break;
            }
#endif
        }
        else
        {
            cerr << "SSLDaemon::processMessagesLegacy(): Last error = " << sslerr << std::endl;
            break;
        }
#ifdef WIN32
        Sleep(500);
#else
        sleep(1);
#endif
        retries--;
    } while ((sslerr != 1) && (retries > 0));

    //Add Connection to connection list
    this->mConnections->add(mController); //add new connection;
    cerr << "SSLDaemon::startCovise(): add " << mController;
}

void SSLDaemon::spawnProcesses(std::string cmd, std::vector<std::string> opt)
{
    //Compose argument array
    const char **locOpt = new const char *[opt.size() + 2];
    const char **locOptBegin = locOpt;

    cerr << "SSLDaemon::spawnProcesses(): Command: " << cmd.c_str() << std::endl;

    //Go through list of std::strings and
    //put char-pointers in pointer array
    std::vector<std::string>::iterator itr;

    itr = opt.begin();
    while (itr != opt.end())
    {
        *locOpt = itr->c_str();
        cerr << "SSLDaemon::spawnProcesses(): Argument(): " << itr->c_str() << std::endl;
        itr++;
        locOpt++;
    }

    //NULL-terminating the arg-list
    *locOpt = NULL;

    for (int i = 0; i < opt.size() + 2; i++)
    {
        if (*(locOptBegin + i) != NULL)
        {
            cerr << "SSLDaemon::spawnProcesses(): Argument(" << i << "): " << *(locOptBegin + i) << std::endl;
        }
        else
        {
            cerr << "SSLDaemon::spawnProcesses(): End of List! " << std::endl;
            break;
        }
    }

//Spawn process
#ifdef _WIN32
    spawnvp(P_NOWAIT, cmd.c_str(), (char *const *)locOptBegin);
#else
    int pid = fork();
    if (pid == 0)
    {
        execvp(cmd.c_str(), (char *const *)locOptBegin);
    }
    else if (pid == -1)
    {
        cerr << "SSLDaemon::spawnProcesses(): Couldn't fork! " << std::endl;
    }
    else
    {
        // Needed to prevent zombies
        // if childs terminate
        signal(SIGCHLD, SIG_IGN);
    }
#endif

    //delete argument array
    delete[] locOptBegin;
}

int SSLDaemon::SplitString(const string &input,
                           const string &delimiter, vector<string> &results,
                           bool includeEmpties)
{
    int iPos = 0;
	size_t newPos = -1;
    int sizeS2 = (int)delimiter.size();
    int isize = (int)input.size();

    if ((isize == 0)
        || (sizeS2 == 0))
    {
        return 0;
    }

    vector<int> positions;

    newPos = input.find(delimiter, 0);

    if (newPos < 0)
    {
        return 0;
    }

    int numFound = 0;

    while (newPos >= iPos)
    {
        numFound++;
        positions.push_back((int)newPos);
        iPos = (int)newPos;
        newPos = input.find(delimiter, iPos + sizeS2);
    }

    if (numFound == 0)
    {
        return 0;
    }

    for (int i = 0; i <= (int)positions.size(); ++i)
    {
        string s("");
        if (i == 0)
        {
            s = input.substr(i, positions[i]);
        }
        //Hier krachts --> Jetzt erschtmal verstehen
        int offset = positions[i - 1] + sizeS2;
        if (offset < isize)
        {
            if (i == positions.size())
            {
                s = input.substr(offset);
            }
            else if (i > 0)
            {
                s = input.substr(positions[i - 1] + sizeS2,
                                 positions[i] - positions[i - 1] - sizeS2);
            }
        }
        if (includeEmpties || (s.size() > 0))
        {
            results.push_back(s);
        }
    }
    return numFound;
}

void SSLDaemon::allowPermanent(bool storeGlobal)
{
    std::string uniqueID;
    mIsAllowed = true;
    mbConfirmed = true;
    if (mRequest->getCurrentMode() == frmRequestDialog::MachineLevel)
    {
        cerr << "SSLDaemon::allowPermanent(): Writing host IP = " << mCurrentPeer << std::endl;

        if (storeGlobal)
        {
            coConfig::getInstance()->setValue("hostname", QString(mCurrentPeer.c_str()), "System.CoviseDaemon.HostACL.Host");
        }
        else
        {
            mConfig->setValue("hostname", QString(mCurrentPeer.c_str()), "System.CoviseDaemon.HostACL.Host");
        }
        uniqueID = mCurrentPeer;

        std::string::size_type pos = uniqueID.find(".");
        do
        {
            uniqueID.erase(pos, 1);
            pos = uniqueID.find(".");
        } while (pos != std::string::npos);

        cerr << "SSLDaemon::allowPermanent(): Writing unique ID = " << uniqueID << std::endl;

        if (storeGlobal)
        {
            coConfig::getInstance()->setValue("index", QString(uniqueID.c_str()), "System.CoviseDaemon.HostACL.Host");
        }
        else
        {
            mConfig->setValue("index", QString(uniqueID.c_str()), "System.CoviseDaemon.HostACL.Host");
            mConfig->save();
        }
    }
    else
    {
        cerr << "SSLDaemon::allowPermanent(): Writing subject Name = " << mSubjectUID << std::endl;

        if (storeGlobal)
        {
            coConfig::getInstance()->setValue("Name", QString(mSubjectUID.c_str()), "System.CoviseDaemon.AllowedUID.Subject");
        }
        else
        {
            mConfig->setValue("Name", QString(mSubjectUID.c_str()), "System.CoviseDaemon.AllowedUID.Subject");
        }

        uniqueID = mCurrentPeer;
        std::string::size_type pos = uniqueID.find(".");
        do
        {
            uniqueID.erase(pos, 1);
            pos = uniqueID.find(".");
        } while (pos != std::string::npos);

        cerr << "SSLDaemon::allowPermanent(): Writing unique ID = " << mSubjectUID << std::endl;

        if (storeGlobal)
        {
            coConfig::getInstance()->setValue("index", QString(uniqueID.c_str()), "System.CoviseDaemon.AllowedUID.Subject");
        }
        else
        {
            mConfig->setValue("index", QString(ToString((int)mSubjectNameList.size()).c_str()), "System.CoviseDaemon.AllowedUID.Subject");
            mConfig->save();
        }
    }
}
void SSLDaemon::allow()
{
    mIsAllowed = true;
    mbConfirmed = true;
}
void SSLDaemon::deny()
{
    mIsAllowed = false;
    mbConfirmed = true;
}

bool SSLDaemon::isClientValid(SSLServerConnection *server)
{
    std::string ssl_uid;

    ssl_uid = server->getSSLSubjectName();
    if (ssl_uid == "")
    {
        return false;
    }
    std::vector<std::string>::iterator itrUID;
    itrUID = mSubjectNameList.begin();
    while ((itrUID != mSubjectNameList.end()))
    {
        if (itrUID->find(ssl_uid, 0) != std::string::npos)
        {
            return true;
        }
        itrUID++;
    }
    return false;
}

int SSLDaemon::sslPasswdCallback(char *buf, int size, int rwflag, void *userData)
{
    (void)rwflag;
    SSLDaemon *obj = static_cast<SSLDaemon *>(userData);

    if (obj->mPassword.empty())
    {
        bool ok;
        QString text = QInputDialog::getText(obj->mWindow, tr("Certificate PrivateKey Password"),
                                             tr("Password:"), QLineEdit::Password,
                                             "", &ok);
        if (ok && !text.isEmpty())
        {
            obj->mPassword = text.toStdString();
        }
        else
        {
            return 0;
        }
    }

    if (obj->mPassword.size() > size)
        return 0;

    strncpy(buf, obj->mPassword.c_str(), obj->mPassword.size() /*should be length of buf*/);
    buf[obj->mPassword.size() - 1] = '\0';

    return int(obj->mPassword.size());
}
