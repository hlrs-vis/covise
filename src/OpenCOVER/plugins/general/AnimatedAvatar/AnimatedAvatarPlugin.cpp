#include "AnimatedAvatarPlugin.h"

#include <cover/coVRCommunication.h>
#include <cover/coVRPartner.h>
#include <cover/coVRPluginSupport.h>
#include <cover/ui/FileBrowser.h>
#include <cover/ui/Owner.h>

#include <map>
#include <vector>

#include <osg/MatrixTransform>
#include <osgCal/CoreModel>
#include <osgCal/Model>

using namespace covise;
using namespace opencover;
using namespace ui;



AnimatedAvatarPlugin::AnimatedAvatarPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, Owner(COVER_PLUGIN_NAME, cover->ui)
{
   
}

bool AnimatedAvatarPlugin::update()
{
    addPartner();

    removeAvatar();

    // Avatar-Update durchführen
    for (auto& avatar : m_avatars)
    {
        avatar.second->update();
    }

    return true;
}

void AnimatedAvatarPlugin::removeAvatar()
{
    auto partnerList = coVRPartnerList::instance();
    for (auto avatarIt = m_avatars.begin(); avatarIt != m_avatars.end(); avatarIt++)
    {
        auto &avatar = *avatarIt;
        int partnerId = avatar.first;
        auto partner = std::find_if(partnerList->begin(), partnerList->end(), [partnerId](const std::unique_ptr<coVRPartner> &partner){
            return partner->ID() == partnerId;
        });
        if (partner == partnerList->end())
        {
            avatarIt = m_avatars.erase(avatarIt);
            return;
        }
    }
}

void AnimatedAvatarPlugin::addPartner()
{
    auto partnerList = coVRPartnerList::instance();
    for (const auto& partner : *partnerList)
    {
        int partnerId = partner->ID();
        if(partnerId == coVRCommunication::instance()->getID())
            continue;
        // Avatar bereits vorhanden?
        if (m_avatars.find(partnerId) != m_avatars.end())
        {
            // Avatar bereits vorhanden, überspringen
            continue;
        }
        // CoreModel auswählen
        std::string modelFilename = partner->userInfo().icon;
        auto coreModelIt = m_coreModels.find(modelFilename);
        osg::ref_ptr<osgCal::CoreModel> coreModel; 
        if (coreModelIt == m_coreModels.end())
            coreModel = load_avatar(partnerId); 
        else
            coreModel = coreModelIt->second;
        if(!coreModel)
        {
            std::cerr << "failed to load coremodel " << modelFilename << std::endl;
            return;
        }
        // Avatar erstellen und in m_avatars einfügen
        auto avatar = std::make_shared<AnimatedAvatar>(coreModel, partnerId);
        m_avatars.emplace(partnerId, std::move(avatar));
    }
}

osg::ref_ptr<osgCal::CoreModel> AnimatedAvatarPlugin::load_avatar(int partnerID)
{
    // CoreModel aus Datei laden, falls noch nicht vorhanden

    auto modelFilename = coVRPartnerList::instance()->get(partnerID)->userInfo().icon;
    osg::ref_ptr<osgCal::CoreModel> coreModel = new osgCal::CoreModel();
    try
    {
        coreModel->load(modelFilename);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return nullptr;
    }
    
    m_coreModels[modelFilename] = coreModel;
    return coreModel;
}



COVERPLUGIN(AnimatedAvatarPlugin)