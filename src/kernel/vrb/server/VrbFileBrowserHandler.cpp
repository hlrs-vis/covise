#include "VrbFileBrowserHandler.h"

#include <net/tokenbuffer.h>
#include <net/message.h>
#include <util/coTabletUIMessages.h>
#include <util/unixcompat.h>
#include <qtutil/NetHelp.h>
#include <net/covise_connect.h>

#include <QDir>
#include <qtutil/NetHelp.h> 
#include <qtutil/FileSysAccess.h>
#include <QtNetwork/qhostinfo.h>

#include <iostream>
#include <fstream>

using namespace covise;
using namespace vrb;
void vrb::handleFileBrouwserRequest(covise::Message* msg)
{
	int id = 0;
	int recvId = 0;
	int type = 0;
	char* filter = NULL;
	char* path = NULL;
	char* location = NULL;
	QStringList list;

	//Message for tabletui filebrowser request
	TokenBuffer tb(msg);
	tb >> id;
	tb >> recvId;
	tb >> type;

	VRBSClient* receiver = clients.get(msg->conn);

	switch (type)
	{
	case TABLET_REQ_DIRLIST:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Request DirList!" << std::endl;
#endif
		tb >> filter;
		tb >> path;
		tb >> location;

		QString qFilter = filter;
		QString qPath = path;

		NetHelp networkHelper;
		QString localIP = networkHelper.getLocalIP();
		if (strcmp(location, localIP.toStdString().c_str()) == 0)
		{
			//Server-side dir request
			FileSysAccess fileSys;
			list = fileSys.getFileSysList(qFilter, qPath, false);

			//Create return message
			TokenBuffer rtb;
			rtb << TABLET_SET_DIRLIST;
			rtb << id;
			rtb << list.size();
			for (int i = 0; i < list.size(); i++)
			{
				std::string tempStr = list.at(i).toStdString();
				rtb << tempStr.c_str();
			}

			Message m(rtb);
			m.type = COVISE_MESSAGE_VRB_FB_SET;

			//Send message
			receiver->conn->sendMessage(&m);
		}
		else
		{
			//Determine destination's VRBSClient object and
			//route the request to the appropriate client
			RerouteRequest(location, TABLET_SET_DIRLIST, id, recvId, qFilter, qPath);
		}
	}
	break;
	case TABLET_REQ_FILELIST:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Request FileList!" << std::endl;
#endif
		tb >> filter;
		tb >> path;
		tb >> location;

		QString qFilter = filter;
		QString qPath = path;

		// not used bool sendDir = false;
		NetHelp networkHelper;
		QString localIP = networkHelper.getLocalIP();
		if (strcmp(location, localIP.toStdString().c_str()) == 0)
		{

			QString qoldPath = qPath;

			FileSysAccess fileSys;
			list = fileSys.getFileSysList(qFilter, qPath, true);

			if (!(qoldPath == qPath))
			{
				TokenBuffer rtb;
				rtb << TABLET_SET_CURDIR;
				rtb << id;
				std::string tempStr = qPath.toStdString();
				char* pathTemp = (char*)tempStr.c_str();
				rtb << pathTemp;

				Message m(rtb);
				m.type = COVISE_MESSAGE_VRB_FB_SET;

				//Send message
				receiver->conn->sendMessage(&m);
			}

			TokenBuffer tb2;
			//Create new message
			tb2 << TABLET_SET_FILELIST;
			tb2 << id;
			int d1 = list.size();
			tb2 << d1;
			for (int i = 0; i < d1; i++)
			{
				std::string tempStr = list.at(i).toStdString();
				char* temp = (char*)tempStr.c_str();
				tb2 << temp;
			}

			Message m2(tb2);
			m2.type = COVISE_MESSAGE_VRB_FB_SET;

			//Send message
			receiver->conn->sendMessage(&m2);
		}
		else
		{
			//Determine destination's VRBSClient object and
			//route the request to the appropriate client
			RerouteRequest(location, TABLET_SET_FILELIST, id, recvId, qFilter, qPath);
		}
	}
	break;
	case TABLET_REQ_DRIVES:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Request Drives!" << std::endl;
#endif
		tb >> filter;
		tb >> path;
		tb >> location;

		QString qFilter = filter;
		QString qPath = path;

		// not used bool sendDir = false;
		NetHelp networkHelper;
		QString localIP = networkHelper.getLocalIP();
		if (strcmp(location, localIP.toStdString().c_str()) == 0)
		{

			//Determine available drives
			QFileInfoList list = QDir::drives();

			TokenBuffer tb2;
			//Create new message
			tb2 << TABLET_SET_DRIVES;
			tb2 << id;
			tb2 << list.size();
			for (int i = 0; i < list.size(); i++)
			{
				QFileInfo info = list.at(i);
				QString dir = info.absolutePath();
				std::string sdir = dir.toStdString();
				char* temp = (char*)sdir.c_str();
				tb2 << temp;
			}

			Message m2(tb2);
			m2.type = COVISE_MESSAGE_VRB_FB_SET;

			//Send message
			receiver->conn->sendMessage(&m2);
		}
		else
		{
			//Determine destination's VRBSClient object and
			//route the request to the appropriate client
			RerouteRequest(location, TABLET_SET_DRIVES, id, recvId, qFilter, qPath);
		}
	}
	break;
	case TABLET_REQ_CLIENTS:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Request Clients!" << std::endl;
#endif
		QStringList tuiClientList;
		QString locClient;
		QString locClientName;

		//Determine client list connected to VRB
		for (int i = clients.numberOfClients(); i > 0;)
		{
			VRBSClient* locConn = clients.getNthClient(--i);
			locClientName = QString::fromStdString(locConn->userInfo().userName);
			locClient = QString::fromStdString(locConn->userInfo().ipAdress);
			tuiClientList.append(locClient);
		}

		//Now send gathered client list to requesting client
		TokenBuffer tb2;
		//Create new message
		tb2 << TABLET_SET_CLIENTS;
		tb2 << id;
		tb2 << tuiClientList.size() + 1;
		for (int i = 0; i < tuiClientList.size(); i++)
		{
			std::string tempStr = tuiClientList.at(i).toStdString();
			char* temp = (char*)tempStr.c_str();
			tb2 << temp;
		}

		NetHelp networkHelper;
		QString serverIP = networkHelper.getLocalIP();
		tb2 << serverIP.toStdString().c_str();

		QHostInfo host;
		QString hostName;
		std::string strHostName;

		for (int i = 0; i < tuiClientList.size(); i++)
		{
			QString address(tuiClientList.at(i));
			hostName = networkHelper.GetNamefromAddr(&address);
			strHostName = hostName.toStdString();
			const char* temp = strHostName.c_str();
			tb2 << temp;
		}

		//tb2 << (gethostbyaddr(this->getLocalIP().toAscii().data(),4,AF_INET))->h_name;
		hostName = networkHelper.GetNamefromAddr(&serverIP);
		hostName = hostName + "(VRB-Server)";
		strHostName = hostName.toStdString();
		tb2 << strHostName.c_str();

		Message m2(tb2);
		m2.type = COVISE_MESSAGE_VRB_FB_SET;

		//Send message
		receiver->conn->sendMessage(&m2);

		break;
	}
	case TABLET_FB_FILE_SEL:
	{
		tb >> location;
		tb >> path;

#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Request File!" << std::endl;
		std::cerr << "File: " << path << std::endl;
#endif

		// not used bool sendDir = false;
		NetHelp networkHelper;
		QString localIP = networkHelper.getLocalIP();
		if (strcmp(location, localIP.toStdString().c_str()) == 0)
		{

			//Open local file and put in binary buffer
			QString qpath(path);

			QDir dir;

			QStringList list = qpath.split(dir.separator());
			QString file = list.last();

			int bytes = 0;
			std::ifstream vrbFile;
			//Currently opens files in
			vrbFile.open(path, std::ifstream::binary);
			if (vrbFile.fail())
			{
				//TODO: Evaluate and send NoSuccess message
				TokenBuffer tb2;
				tb2 << TABLET_SET_FILE_NOSUCCESS;
				tb2 << id;
				tb2 << "Open failed!";
				std::cerr << "Opening of file for submission failed!" << std::endl;
				Message m(tb2);
				m.type = COVISE_MESSAGE_VRB_FB_SET;
				//Send message
				receiver->conn->sendMessage(&m);
				return;
			}
			vrbFile.seekg(0, std::ios::end);
			bytes = vrbFile.tellg();
			char* data = new char[bytes];
			vrbFile.seekg(std::ios::beg);
			vrbFile.read(data, bytes);
			if (vrbFile.fail())
			{
				//TODO: Evaluate and send NoSuccess message
				TokenBuffer tb2;
				tb2 << TABLET_SET_FILE_NOSUCCESS;
				tb2 << id;
				tb2 << "Read failed!";
				std::cerr << "Reading of file for submission failed!" << std::endl;
				Message m(tb2);
				m.type = COVISE_MESSAGE_VRB_FB_SET;
				//Send message
				receiver->conn->sendMessage(&m);
				return;
			}
			vrbFile.close();

#ifdef MB_DEBUG
			std::cerr << "Start file send!" << std::endl;
#endif

			TokenBuffer tb2; //Create new message
			tb2 << TABLET_SET_FILE;
			tb2 << id;
			std::string sfile = file.toStdString();
			tb2 << sfile.c_str();
			tb2 << bytes;
			tb2.addBinary(data, bytes);

			delete[] data;

			Message m(tb2);
			m.type = COVISE_MESSAGE_VRB_FB_SET;

			//Send message
			receiver->conn->sendMessage(&m);
#ifdef MB_DEBUG
			std::cerr << "End file send!" << std::endl;
#endif
		}
		else
		{
			//Determine destination's VRBSClient object and
			//route the request to the appropriate client
#ifdef MB_DEBUG
			std::cerr << "Entered reroute of request to OC Client!" << std::endl;
#endif

			QString qFilter = filter;
			QString qPath = path;
			RerouteRequest(location, TABLET_FB_FILE_SEL, id, recvId, qFilter, QString(path));
#ifdef MB_DEBUG
			std::cerr << "Finished reroute of request to OC Client!" << std::endl;
#endif
		}
		break;
	}
	case TABLET_REQ_GLOBALLOAD:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Request GlobalLoad!" << std::endl;
#endif
		char* curl = NULL;
		tb >> curl;

		TokenBuffer tb2;

		//Create new message
		tb2 << TABLET_SET_GLOBALLOAD;
		tb2 << id;
		tb2 << curl;

		Message m2(tb2);
		m2.type = COVISE_MESSAGE_VRB_FB_SET;

		//Send message
		for (int i = clients.numberOfClients(); i > 0;)
		{
			VRBSClient* locConn = clients.getNthClient(--i);
			if (locConn->conn != msg->conn)
			{
				locConn->conn->sendMessage(&m2);
			}
		}
		break;
	}
	case TABLET_REQ_HOMEDIR:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Request HOMEDIR!" << std::endl;
#endif
		tb >> location;

		// not used bool sendDir = false;
		NetHelp networkHelper;
		QString localIP = networkHelper.getLocalIP();
		if (strcmp(location, localIP.toStdString().c_str()) == 0)
		{

			FileSysAccess file;
			QStringList dirList = file.getLocalHomeDir();

			TokenBuffer ltb;
			ltb << TABLET_SET_DIRLIST;
			ltb << id;
			ltb << dirList.size();

			for (int i = 0; i < dirList.size(); i++)
			{
				QString qentry = dirList.at(i);
				std::string sentry = qentry.toStdString();
				ltb << sentry.c_str();
			}

			Message m2(ltb);
			m2.type = COVISE_MESSAGE_VRB_FB_SET;

			//Send message
			receiver->conn->sendMessage(&m2);

			ltb.reset();
			ltb << TABLET_SET_CURDIR;
			ltb << id;
			std::string shome = file.getLocalHomeDirStr().toStdString();
			ltb << shome.c_str();

			Message m3(ltb);
			m3.type = COVISE_MESSAGE_VRB_FB_SET;
			receiver->conn->sendMessage(&m3);
		}
		else
		{
			// TODO: HomeDir retrieval from OC client
			//Determine destination's VRBSClient object and
			//route the request to the appropriate client
			RerouteRequest(location, TABLET_REQ_HOMEDIR, id, recvId, QString(filter), QString(path));
		}
		break;
	}

	case TABLET_REQ_HOMEFILES:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Request HOMEFILES!" << std::endl;
#endif
		tb >> filter;
		tb >> location;

		// not used bool sendDir = false;
		NetHelp networkHelper;
		QString localIP = networkHelper.getLocalIP();
		if (strcmp(location, localIP.toStdString().c_str()) == 0)
		{

			FileSysAccess file;
			QStringList fileList = file.getLocalHomeFiles(filter);

			TokenBuffer ltb;
			ltb << TABLET_SET_FILELIST;
			ltb << id;
			ltb << fileList.size();

			for (int i = 0; i < fileList.size(); i++)
			{
				QString qentry = fileList.at(i);
				std::string sentry = qentry.toStdString();
				ltb << sentry.c_str();
			}

			Message m2(ltb);
			m2.type = COVISE_MESSAGE_VRB_FB_SET;

			//Send message
			receiver->conn->sendMessage(&m2);

			//Send message
			receiver->conn->sendMessage(&m2);
		}
		else
		{
			// TODO: HomeFiles retrieval from OC client
			//Determine destination's VRBSClient object and
			//route the request to the appropriate client
			RerouteRequest(location, TABLET_REQ_HOMEFILES, id, recvId, QString(filter), QString(path));
		}
		break;
	}
	default:
		std::cerr << "Unknown FileBrowser request!" << std::endl;
		break;
	}
}

void vrb::handleFileBrowserRemoteRequest(covise::Message* msg)
{
	int id = 0;
	int receiverId = 0;
	int type = 0;

	//Message for tabletui filebrowser request
	TokenBuffer tb(msg);
	tb >> type;
	tb >> id;
	tb >> receiverId;

	VRBSClient* receiver = clients.get(receiverId);

	switch (type)
	{
	case TABLET_REMSET_FILELIST:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest FileList!" << std::endl;
#endif
		int size = 0;
		char* entry = NULL;

		tb >> size;

		TokenBuffer tb2;
		//Create new message
		tb2 << TABLET_SET_FILELIST;
		tb2 << id;
		tb2 << size;

		for (int i = 0; i < size; i++)
		{
			tb >> entry;
			tb2 << entry;
		}

		Message m2(tb2);
		m2.type = COVISE_MESSAGE_VRB_FB_SET;

		//Send message
		receiver->conn->sendMessage(&m2);
	}
	break;
	case TABLET_REMSET_FILE_NOSUCCESS:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest NoSuccess!" << std::endl;
#endif
		char* comment = NULL;

		tb >> comment;

		TokenBuffer tb2;
		//Create new message
		tb2 << TABLET_SET_FILE_NOSUCCESS;
		tb2 << id;
		tb2 << comment;

		Message m2(tb2);
		m2.type = COVISE_MESSAGE_VRB_FB_SET;

		//Send message
		receiver->conn->sendMessage(&m2);
	}
	break;
	case TABLET_REMSET_DIRLIST:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest DirList!" << std::endl;
#endif
		int size = 0;
		char* entry = NULL;

		tb >> size;

		TokenBuffer tb2;
		//Create new message
		tb2 << TABLET_SET_DIRLIST;
		tb2 << id;
		tb2 << size;

		for (int i = 0; i < size; i++)
		{
			tb >> entry;
			tb2 << entry;
		}

		Message m2(tb2);
		m2.type = COVISE_MESSAGE_VRB_FB_SET;

		//Send message
		receiver->conn->sendMessage(&m2);
	}
	break;
	case TABLET_REMSET_FILE:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest FileRequest!" << std::endl;
#endif
		char* filename = NULL;
		int size = 0;
		tb >> filename;
		tb >> size;

		TokenBuffer rt;
		rt << TABLET_SET_FILE;
		rt << id;
		rt << filename;
		rt << size;
		rt.addBinary(tb.getBinary(size), size);

		Message m(rt);
		m.type = COVISE_MESSAGE_VRB_FB_SET;
		receiver->conn->sendMessage(&m);
	}
	break;
	case TABLET_REMSET_DIRCHANGE:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest DirChange!" << std::endl;
#endif
		char* cpath = NULL;
		tb >> cpath;
		TokenBuffer rtb;
		rtb << TABLET_SET_CURDIR;
		rtb << id;
		rtb << cpath;

		Message m(rtb);
		m.type = COVISE_MESSAGE_VRB_FB_SET;

		//Send message
		receiver->conn->sendMessage(&m);
	}
	break;
	case TABLET_REMSET_DRIVES:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser RemoteRequest DriveList!" << std::endl;
#endif
		int size = 0;
		char* entry = NULL;

		tb >> size;

		TokenBuffer tb2;
		//Create new message
		tb2 << TABLET_SET_DRIVES;
		tb2 << id;
		tb2 << size;

		for (int i = 0; i < size; i++)
		{
			tb >> entry;
			tb2 << entry;
		}

		Message m2(tb2);
		m2.type = COVISE_MESSAGE_VRB_FB_SET;

		//Send message
		receiver->conn->sendMessage(&m2);
	}
	break;
	default:
	{
#ifdef MB_DEBUG
		std::cerr << "::HANDLECLIENT VRB FileBrowser Unknown RemoteRequest!" << std::endl;
#endif
		std::cerr << "unknown VRB FileBrowser RemoteRequest message in vrb" << std::endl;
	}
	}
}



void vrb::RerouteRequest(const char* location, int type, int senderId, int recvVRBId, QString filter, QString path)
{
	VRBSClient* locClient = NULL;

	for (int i = 0; i < clients.numberOfClients(); i++)
	{
		if (location == clients.getNthClient(i)->userInfo().ipAdress)
		{
			locClient = clients.getNthClient(i);
		}
	}

	TokenBuffer tb;
	tb << type;
	tb << senderId;
	tb << recvVRBId;
	std::string spath = path.toStdString();
	tb << spath.c_str();
	std::string sfilter = filter.toStdString();
	tb << sfilter.c_str();

	Message msg(tb);
	msg.type = COVISE_MESSAGE_VRB_FB_REMREQ;

	if (locClient != NULL)
	{
		locClient->conn->sendMessage(&msg);
	}
}
