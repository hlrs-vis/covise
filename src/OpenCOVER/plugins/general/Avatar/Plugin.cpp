
#include "Plugin.h"
#include "Avatar.h"
#include <cover/coVRFileManager.h>

using namespace opencover;

COVERPLUGIN(AvatarPlugin);

AvatarPlugin::AvatarPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("AvatarPlugin", cover->ui)//braucht man für gui
, m_transform(new osg::MatrixTransform)
, m_sphereTransform(new osg::MatrixTransform)
, m_config(config())
, m_menu(new ui::Menu("Avatar",this))
{
    m_avatarFile = std::make_unique<FileBrowserConfigValue>(m_menu, "avatarFile", "", *m_config, "");
    m_avatarFile->ui()->setFilter("*.fbx");
    m_avatarFile->setUpdater([this](){
        loadAvatar();
    });
    loadAvatar();

    m_interactor .reset(new coVR3DTransInteractor(osg::Vec3{-400,0,0}, 10, vrui::coInteraction::InteractionType::ButtonA, "target", "targetInteractor", vrui::coInteraction::InteractionPriority::Medium));
    m_interactor->enableIntersection();
}

void AvatarPlugin::loadAvatar()
{
    m_avatar = std::make_unique<LoadedAvatar>();
    auto path = coVRFileManager::instance()->findOrGetFile(m_avatarFile->getValue());
    if(m_avatar->loadAvatar(path, m_menu))
        m_config->save();
    else
        m_avatar = nullptr;
}

void updateMatrixPosition(osg::MatrixTransform *mt, const osg::Vec3f &pos)
{
    auto m = mt->getMatrix();
    m.setTrans(m.getTrans() + pos);
    mt->setMatrix(m);
}

void AvatarPlugin::key(int type, int keySym, int mod)
{
    std::string key = "unknown";
    if (!(keySym & 0xff00))
    {
        char buf[2] = { static_cast<char>(keySym), '\0' };
        key = buf;
    }
    constexpr float speed = 50;
    osg::Vec3f position;
    if(key == "w")
    {
        position = osg::Vec3f(speed,0,0);
    } else if (key == "a")
    {
        position = osg::Vec3f(0, 0, speed);
    }
    else if (key == "s")
    {
        position = osg::Vec3f(-speed,0,0);
    }
    else if (key == "d")
    {
        position = osg::Vec3f(0, 0, -speed);
    }
    else if (key == "e")
    {
        position = osg::Vec3f(0, speed, 0);
    }
    else if (key == "r")
    {
        position = osg::Vec3f(0, -speed, 0);
    }
    updateMatrixPosition(m_sphereTransform, position);

}

bool AvatarPlugin::update(){

    if(!m_avatar)
        return true;
    osg::Matrix m = osg::Matrix::identity();
    auto pos = m_interactor->getPos();
    m.setTrans(pos);
    m_sphereTransform->setMatrix(m);
    m_avatar->update(pos);
    return true;
}

void AvatarPlugin::preFrame()
{
    m_interactor->preFrame();
}
