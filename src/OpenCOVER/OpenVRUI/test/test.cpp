/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "test.h"

#define WINW 800
#define WINH 600

#include <iostream>
#include <string>

#include <osg/Geode>
#include <osg/Math>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osgProducer/Viewer>

#include <Producer/Camera>

using namespace std;
using namespace osgProducer;
using namespace osg;

#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coDefaultButtonGeometry.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/coSlider.h>

#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

#include <OpenVRUI/osg/OSGVruiColoredBackground.h>
#include <OpenVRUI/osg/OSGVruiDefaultButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiFlatButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiFlatPanelGeometry.h>
#include <OpenVRUI/osg/OSGVruiFrame.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/osg/OSGVruiPanelGeometry.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiLabel.h>
#include <OpenVRUI/osg/OSGVruiToggleButtonGeometry.h>
#include <OpenVRUI/osg/OSGVruiSlider.h>
#include <OpenVRUI/osg/OSGVruiNull.h>

VRUITest::VRUITest()
{

    //cerr << "VRUITest::<init> info: creating " << endl;

    the = this;

    //cerr << "VRUITest::<init> info: assigned interface " << endl;

    Producer::RenderSurface *surface = new Producer::RenderSurface;
    Producer::Camera *camera = new Producer::Camera;
    Producer::CameraConfig *cameraConfig = new Producer::CameraConfig;

    surface->setScreenNum(0);
    surface->setWindowName("OSG Test");
    surface->setWindowRectangle(0, 0, WINW, WINH);

    camera->setRenderSurface(surface);
    camera->setProjectionRectangle(0, 0, WINW, WINH);
    //   camera->setViewByLookat(Producer::Vec3(0.0f, 0.0f, -6.0f),
    //                           Producer::Vec3(0.0f, 0.0f, 0.0f),
    //                           Producer::Vec3(0.0f, 1.0f, 0.0f));

    cameraConfig->addCamera("Main Camera", camera);

    //cerr << "VRUITest::<init> info: OSG setup complete" << endl;

    Group *group = new Group();
    groupNode = new OSGVruiNode(group);

    OSGVruiTransformNode *transformNode;

    //cerr << "VRUITest::<init> info: made Group node" << endl;

    //frame = new coFrame();
    //cerr << "VRUITest::<init> info: made coFrame" << endl;

    //panel = new coPanel(new coFlatPanelGeometry(coUIElement::GREY));
    //cerr << "VRUITest::<init> info: made coPanel" << endl;

    //coFrame * frame2 = new coFrame();
    //button = new coPushButton(new coFlatButtonGeometry(), new coButtonActor());
    //cerr << "VRUITest::<init> info: made coButton" << endl;

    //OSGVruiTransformNode * panelTransform = dynamic_cast<OSGVruiTransformNode*>(panel->getDCS());
    //OSGVruiTransformNode * frameTransform = dynamic_cast<OSGVruiTransformNode*>(frame->getDCS());
    //OSGVruiTransformNode * buttonTransform = dynamic_cast<OSGVruiTransformNode*>(button->getDCS());

    //vruiTransform->setTranslation(100, 0, 0);

    //  group->addChild(vruiTransform->getNodePtr());

    //   OSGVruiTransformNode * vruiTransform2 = dynamic_cast<OSGVruiTransformNode*>(button->getDCS());
    //   vruiTransform2->setTranslation(100, 0, 100);
    //   group->addChild(vruiTransform2->getNodePtr());

    //   coColoredBackground * back = new coColoredBackground(coUIElement::GREY, coUIElement::DARK_YELLOW, coUIElement::RED);
    //   back->setMinWidth(400);
    //   back->setMinHeight(600);
    //OSGVruiTransformNode * backTransform = dynamic_cast<OSGVruiTransformNode*>(back->getDCS());

    //frame->setSize(60, 40, 0);
    //frame->addElement(back);

    coRowMenu *menu = new coRowMenu("Test", 0);
    coSliderMenuItem *sliderMenuItem = new coSliderMenuItem("Test Slider", 0.0f, 1.0f, 0.5f);
    coButtonMenuItem *buttonMenuItem = new coButtonMenuItem("Test Button");
    coSubMenuItem *subMenuItem = new coSubMenuItem("Test SubMenu");
    coCheckboxMenuItem *checkMenuItem1 = new coCheckboxMenuItem("Test Checkbox 1", false, 0);

    menu->add(sliderMenuItem);
    menu->add(buttonMenuItem);
    menu->add(checkMenuItem1);
    menu->insert(subMenuItem, 2);

    coRowMenu *subMenu = new coRowMenu("Test SubMenu Title", menu);
    coCheckboxMenuItem *checkMenuItem2 = new coCheckboxMenuItem("Test Checkbox 2", true, 0);
    subMenu->add(checkMenuItem2);
    subMenuItem->setAttachment(coUIElement::LEFT);
    subMenuItem->setMenu(subMenu);
    subMenuItem->openSubmenu();

    //coLabel * label = new coLabel();
    //label->setString("Test");

    transformNode = dynamic_cast<OSGVruiTransformNode *>(menu->getDCS());

    group->addChild(transformNode->getNodePtr());

    Viewer viewer(cameraConfig);

    viewer.setUpViewer(Viewer::TRACKBALL_MANIPULATOR | Viewer::SKY_LIGHT_SOURCE | Viewer::HEAD_LIGHT_SOURCE | Viewer::ESCAPE_SETS_DONE);
    viewer.setSceneData(group);

    //viewer.getApplicationUsage()->write(std::cout);

    viewer.realize();

    while (!viewer.done())
    {
        viewer.sync();
        viewer.update();
        viewer.frame();
    }

    viewer.sync();
}

VRUITest::~VRUITest()
{
}

vruiNode *VRUITest::getMenuGroup()
{
    return groupNode;
}

vruiUIElementProvider *VRUITest::createUIElementProvider(coUIElement *element)
{

    string name(element->getClassName());

    if (name == "coFrame")
    {
        coFrame *frame = dynamic_cast<coFrame *>(element);
        if (frame)
        {
            cerr << "VRUITest::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiFrame(frame);
        }
    }

    if (name == "coBackground")
    {
        coBackground *back = dynamic_cast<coBackground *>(element);
        if (back)
        {
            cerr << "VRUITest::createUIElementProvider info: creating (null) provider for " << name << endl;
            return new OSGVruiNull(back);
        }
    }

    if (name == "coColoredBackground")
    {
        coColoredBackground *back = dynamic_cast<coColoredBackground *>(element);
        if (back)
        {
            cerr << "VRUITest::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiColoredBackground(back);
        }
    }

    if (name == "coLabel")
    {
        coLabel *label = dynamic_cast<coLabel *>(element);
        if (label)
        {
            cerr << "VRUITest::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiLabel(label);
        }
    }

    if (name == "coSlider")
    {
        coSlider *slider = dynamic_cast<coSlider *>(element);
        if (slider)
        {
            cerr << "VRUITest::createUIElementProvider info: creating provider for " << name << endl;
            return new OSGVruiSlider(slider);
        }
    }

    cerr << "VRUITest::createUIElementProvider err: " << name << ": should not be here" << endl;
    return 0;
}

vruiButtonProvider *VRUITest::createButtonProvider(coButtonGeometry *element)
{

    string name(element->getClassName());

    if (name == "coDefaultButtonGeometry")
    {
        coDefaultButtonGeometry *button = dynamic_cast<coDefaultButtonGeometry *>(element);
        if (button)
        {
            cerr << "VRUITest::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiDefaultButtonGeometry(button);
        }
    }

    if (name == "coFlatButtonGeometry")
    {
        coFlatButtonGeometry *button = dynamic_cast<coFlatButtonGeometry *>(element);
        if (button)
        {
            cerr << "VRUITest::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiFlatButtonGeometry(button);
        }
    }

    if (name == "coToggleButtonGeometry")
    {
        coToggleButtonGeometry *button = dynamic_cast<coToggleButtonGeometry *>(element);
        if (button)
        {
            cerr << "VRUITest::createButtonProvider info: creating provider for " << name << endl;
            return new OSGVruiToggleButtonGeometry(button);
        }
    }

    cerr << "VRUITest::createUIElementProvider err: " << name << ": should not be here" << endl;
    return 0;
}

vruiPanelGeometryProvider *VRUITest::createPanelGeometryProvider(coPanelGeometry *element)
{

    string name(element->getClassName());

    if (name == "coPanelGeometry")
    {

        coPanelGeometry *panel = dynamic_cast<coPanelGeometry *>(element);
        if (panel)
        {
            cerr << "VRUITest::createPanelGeometryProvider info: creating provider for " << name << endl;
            return new OSGVruiPanelGeometry(panel);
        }
    }

    if (name == "coFlatPanelGeometry")
    {
        coFlatPanelGeometry *panel = dynamic_cast<coFlatPanelGeometry *>(element);
        if (panel)
        {
            cerr << "VRUITest::createPanelGeometryProvider info: creating provider for " << name << endl;
            return new OSGVruiFlatPanelGeometry(panel);
        }
    }

    cerr << "VRUITest::createUIElementProvider err: " << name << ": should not be here" << endl;
    return 0;
}

vruiTransformNode *VRUITest::createTransformNode()
{

    MatrixTransform *transform = new MatrixTransform();

    return new OSGVruiTransformNode(transform);
}

vruiMatrix *VRUITest::createMatrix()
{
    return new OSGVruiMatrix();
}

int main(int, char **)
{

    VRUITest test;

    return 0;
}
