#ifndef COVER_PLUGIN_FBXAVATAR_PLUGIN_H
#define COVER_PLUGIN_FBXAVATAR_PLUGIN_H

#include <cover/coVRPluginSupport.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/FileBrowser.h>
#include <cover/ui/VectorEditField.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Menu.h>

#include "Avatar.h"

class PLUGINEXPORT AvatarPlugin : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:

    AvatarPlugin();
private:
    std::unique_ptr<opencover::ui::FileBrowserConfigValue> m_avatarFile;
    osg::ref_ptr<osg::MatrixTransform>m_transform;//Position des Avatars
    osg::ref_ptr<osg::MatrixTransform>m_sphereTransform;
    std::shared_ptr<config::File>m_config;
    ui::Menu* m_menu = nullptr;
    std::unique_ptr<LoadedAvatar> m_avatar; 
    std::unique_ptr<opencover::coVR3DTransInteractor> m_interactor; 

    void loadAvatar();
    void key(int type, int keySym, int mod) override;
    bool update() override; //wird in jedem Frame aufgerufen
    void preFrame() override;

};

#endif // COVER_PLUGIN_FBXAVATAR_PLUGIN_H
