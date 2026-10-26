/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2024 HLRS  **
 **                                                                          **
 ** Description: Infobox Plugin                                        **
 **                                                                          **
 **                                                                          **
 ** Author: Uwe Woessner 		                                              **
 **                                                                          **
 ** History:  								                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Infobox.h"

#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/VRSceneGraph.h>
#include <osg/ref_ptr>

#include <string>
#include <vrml97/vrml/VrmlNamespace.h>
#include <cover/ui/Manager.h>
#include <config/CoviseConfig.h>

#include "VrmlNodeInfobox.h"

#include <cover/ui/Owner.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Group.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Label.h>

#include <PluginUtil/PluginMessageTypes.h>
#include <stdio.h>

using namespace vrui;
using namespace opencover;

Infobox *Infobox::s_instance = nullptr;

Infobox *Infobox::instance()
{
    // assert(s_instance != nullptr);
    return s_instance;
}

Infobox::Infobox()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("Infobox", cover->ui)
{
    assert(s_instance == nullptr);
    s_instance = this;

    VrmlNamespace::addBuiltIn(VrmlNode::defineType<VrmlNodeInfobox>());
}

bool Infobox::init()
{

    m_popupHandle = new coPopupHandle("Info");
    m_popupHandle->setScale(0.4);
    m_popupHandle->setPos(-40, 0, -40); // ??

    m_frame = new vrui::coFrame("UI/Frame");
    // m_panel = new coPanel(new vrui::coFlatPanelGeometry(vrui::coUIElement::BLACK));
    m_imageBackground = new vrui::coTexturedBackground(nullptr, nullptr, nullptr, 3, 0, 0, 0);

    float px = 0;
    float py = 100;
    float pz = 0;
    float s = 1;
    vrui::OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;
    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.makeIdentity();
    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);
    //
    m_popupHandle->setTransformMatrix(&matrix);
    // m_popupHandle->setScale(1.0);
    // m_popupHandle->setScale(cover->getSceneSize() / 2500);

    // m_infobox->setScale(10 * cover->getSceneSize() / 2500);
    // m_infobox->setPos(0, 0, 0);

    m_popupHandle->addElement(m_frame);
    m_frame->addElement(m_imageBackground);
    m_popupHandle->setVisible(false);

    return true;
}

// this is called if the plugin is removed at runtime
Infobox::~Infobox()
{
    s_instance = nullptr;

    delete m_popupHandle;
}

void Infobox::setContents(std::string_view imageUrl)
{
    if (imageUrl.empty())
        hide();
    else
        show();

    auto s = coVRFileManager::instance()->getName(std::string(imageUrl).c_str());
    if (s)
    {
        osg::Image *image = osgDB::readImageFile(s);
        std::cout << "Loaded image " << image->s() << "x" << image->t() << "x" << image->r() << " at " << imageUrl << std::endl;

        float width = image->s();
        float height = image->t();

        m_imageBackground->setImage((uint *)image->data(), NULL, NULL, image->getPixelSizeInBits() / 8, image->s(), image->t(), image->r());
        m_imageBackground->setTexSize(width, height);
        m_imageBackground->setSize(width, height, 0);
        m_imageBackground->setMinWidth(width);
        m_imageBackground->setMinHeight(height);
        m_imageBackground->setUpdated(true);
    }
    else
    {
        std::cout << "[Infobox.imageUrl] File not found: " << imageUrl << std::endl;
    }
    // TODO: do not leak image, clean up
}

void Infobox::show()
{
    m_popupHandle->setVisible(true);
}
void Infobox::hide()
{
    m_popupHandle->setVisible(false);
}

COVERPLUGIN(Infobox)
