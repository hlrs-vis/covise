// TuioClientPlugin.cpp

#include "TuioClientPlugin.h"
#include "RREvent.h"

#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>

#include <PluginUtil/PluginMessageTypes.h>

TuioClientPlugin::TuioClientPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, opencover::coVRPlugin()
, tuioClient(NULL)
{
}

TuioClientPlugin::~TuioClientPlugin()
{
}

bool TuioClientPlugin::init()
{
    try
    {
        destination = covise::coCoviseConfig::getEntry("destination", "COVER.Plugin.TuioClientPlugin", "Utouch3D2");

        int port = covise::coCoviseConfig::getInt("udpPort", "COVER.Plugin.TuioClientPlugin", 50096);

        tuioClient = new TUIO::TuioClient(port);
        tuioClient->connect();
        tuioClient->addTuioListener(this);

        std::cout << "TuioClientPlugin: destination: '" << destination.c_str() << "'" << std::endl;
        std::cout << "TuioClientPlugin: port: '" << port << "'" << std::endl;

        return true;
    }
    catch (std::exception &e)
    {
        std::cout << "TuioPlugin::init: EXCEPTION: " << e.what() << std::endl;

        return false;
    }
}

bool TuioClientPlugin::destroy()
{
    try
    {
        tuioClient->removeTuioListener(this);
        tuioClient->disconnect();

        delete tuioClient;
        tuioClient = NULL;

        return true;
    }
    catch (std::exception &e)
    {
        std::cout << "TuioPlugin::destroy: EXCEPTION: " << e.what() << std::endl;

        return false;
    }
}

void TuioClientPlugin::addTuioObject(TUIO::TuioObject * /*tobj*/)
{
    //std::cout << "TuioClientPlugin::addTuioObject: not implemented" << std::endl;
}

void TuioClientPlugin::updateTuioObject(TUIO::TuioObject * /*tobj*/)
{
    //std::cout << "TuioClientPlugin::updateTuioObject: not implemented" << std::endl;
}

void TuioClientPlugin::removeTuioObject(TUIO::TuioObject * /*tobj*/)
{
    //std::cout << "TuioClientPlugin::removeTuioObject: not implemented" << std::endl;
}

void TuioClientPlugin::addTuioCursor(TUIO::TuioCursor *tcur)
{
    rrxevent rev(RREV_TOUCHPRESS, tcur->getX(), tcur->getY(), tcur->getCursorID(), 0);

    opencover::cover->sendMessage(this, destination.c_str(), opencover::PluginMessageTypes::RRZK_rrxevent, sizeof(rrxevent), &rev, true);
}

void TuioClientPlugin::updateTuioCursor(TUIO::TuioCursor *tcur)
{
    rrxevent rev(RREV_TOUCHMOVE, tcur->getX(), tcur->getY(), tcur->getCursorID(), 0);

    opencover::cover->sendMessage(this, destination.c_str(), opencover::PluginMessageTypes::RRZK_rrxevent, sizeof(rrxevent), &rev, true);
}

void TuioClientPlugin::removeTuioCursor(TUIO::TuioCursor *tcur)
{
    rrxevent rev(RREV_TOUCHRELEASE, tcur->getX(), tcur->getY(), tcur->getCursorID(), 0);

    opencover::cover->sendMessage(this, destination.c_str(), opencover::PluginMessageTypes::RRZK_rrxevent, sizeof(rrxevent), &rev, true);
}

void TuioClientPlugin::refresh(TUIO::TuioTime /*ftime*/)
{
}

COVERPLUGIN(TuioClientPlugin)
