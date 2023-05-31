/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*********************************************************************************\
 **                                                            (C)2001 HLRS      **
 **                                                                              **
 ** Description: Template Plugin (Building coordinatesystems (global and local)) **
 **                                                                              **
 **                                                                              **
 ** Author: A.Gottlieb                                                           **
 **                                                                              **
 ** History:                   **
 ** Okt-08  v1                                   **
 **                                                                              **
 **                                                                              **
\*********************************************************************************/

#include "CoordSystems.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <osg/CullStack>
#include <iostream>

CoordSystems *CoordSystems::plugin = NULL;
coRowMenu *CoordSystems::coords_menu = NULL;

namespace vrui
{
class coSubMenuItem;
}

osg::ref_ptr<osg::MatrixTransform> CoordAxis::axisNode = 0;
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
CoordSystems::CoordSystems()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}
//------------------------------------------------------------------------------------------------------------------------------
bool CoordSystems::init()
{
    fprintf(stderr, "CoordSystems::CoordSystems\n");
    if (plugin)
    {
        fprintf(stderr, "already have an instance of CoordSystems\n");
    }
    plugin = this;

    cover_menu = cover->getMenu();;
    if (cover_menu)
    {
        button = new coSubMenuItem("Coordsys");
        coord_menu = new coRowMenu("Coords");

        coords = new coSubMenuItem("Coords");
        coords_menu = new coRowMenu("Coords");
        coords->setMenu(coords_menu);
        coords->setMenuListener(this);

        menu_showCoord = new coCheckboxMenuItem("Show global Coordinatesystem", false);
        menu_showCoord->setMenuListener(this);
        coords_menu->add(menu_showCoord);

        menue_showSelected = new coCheckboxMenuItem("Show Coordinatesystem of selected Part", false);
        menue_showSelected->setMenuListener(this);
        coords_menu->add(menue_showSelected);

        Scale = new coSubMenuItem("Scale");
        Scale_menu = new coRowMenu("Scale");
        Scale->setMenu(Scale_menu);
        scale_factor_Group = new coCheckboxGroup(false);
        scale_factor[1] = new coCheckboxMenuItem("0.001", false, scale_factor_Group);
        scale_factor[1]->setMenuListener(this);
        Scale_menu->add(scale_factor[1]);
        scale_factor[2] = new coCheckboxMenuItem("0.01", false, scale_factor_Group);
        scale_factor[2]->setMenuListener(this);
        Scale_menu->add(scale_factor[2]);
        scale_factor[3] = new coCheckboxMenuItem("0.1", false, scale_factor_Group);
        scale_factor[3]->setMenuListener(this);
        Scale_menu->add(scale_factor[3]);
        scale_factor[4] = new coCheckboxMenuItem("1", false, scale_factor_Group);
        scale_factor[4]->setMenuListener(this);
        Scale_menu->add(scale_factor[4]);
        scale_factor[5] = new coCheckboxMenuItem("10", false, scale_factor_Group);
        scale_factor[5]->setMenuListener(this);
        Scale_menu->add(scale_factor[5]);
        scale_factor[6] = new coCheckboxMenuItem("100", false, scale_factor_Group);
        scale_factor[6]->setMenuListener(this);
        Scale_menu->add(scale_factor[6]);
        scale_factor[7] = new coCheckboxMenuItem("1000", false, scale_factor_Group);
        scale_factor[7]->setMenuListener(this);
        Scale_menu->add(scale_factor[7]);
        scale_factor_Group->setState(scale_factor[6], true);

        coord_menu->add(coords);
        button->setMenu(coords_menu);
        cover_menu->add(button);
        coords_menu->add(Scale);
    }
    globalCoordSys = new CoordAxis();

    coVRSelectionManager::instance()->addListener(this);
    return true;
}
//------------------------------------------------------------------------------------------------------------------------------
// this is called if the plugin is removed at runtime
CoordSystems::~CoordSystems()
{
    //fprintf ( stderr,"CoordSystems::~CoordSystems\n" );
    delete Scale_menu;
    delete Scale;
    delete menu_showCoord;
    delete menue_showSelected;
    delete coord_menu;
    delete coords;
    delete button;
    delete coords_menu;
    globalCoordSys->RemoveFromScenegraph();
    delete globalCoordSys;
}
//------------------------------------------------------------------------------------------------------------------------------
void
CoordSystems::preFrame()
{
}
//------------------------------------------------------------------------------------------------------------------------------
void CoordSystems::menuEvent(coMenuItem *item)
{

    if (item == menu_showCoord)
    {

        coCheckboxMenuItem *m = dynamic_cast<coCheckboxMenuItem *>(item);
        if (m && m->getState())
        {
            globalCoordSys->AddToScenegraph(cover->getObjectsRoot());
            // globalCoordSys->printMatrix();
            for (int i = 0; i <= 6; i++)
            {
                if (scale_factor[i + 1]->getState())
                {
                    globalCoordSys->setScaling(ScFactors[i]);
                    globalCoordSys->setLabelValue(ScFactors[i]);
                    //globalCoordSys->printMatrix();
                }
            }
        }
        else
        {
            globalCoordSys->RemoveFromScenegraph();
            globalCoordSys->makeIdentity();
        }
    }

    if (item == menue_showSelected)
    {
        refresh_coords();
    }

    for (int i = 0; i <= 6; i++)
    {

        if (item == scale_factor[i + 1])
        {
            refresh_coords();
        }
    }
}
//------------------------------------------------------------------------------------------------------------------------------
bool CoordSystems::selectionChanged()
{
    //std::cerr << "CoordSystems::selectionChanged info: called" << std::endl;
    refresh_coords();
    return true;
}
//------------------------------------------------------------------------------------------------------------------------------
bool CoordSystems::pickedObjChanged()
{
    return true;
}
//------------------------------------------------------------------------------------------------------------------------------
bool CoordSystems::refresh_coords()
{
    for (std::list<CoordAxis *>::iterator k = CoordList.begin(); k != CoordList.end(); ++k)
    {
        (*k)->RemoveFromScenegraph();
        delete *k;
    }
    CoordList.clear();

    coCheckboxMenuItem *m = dynamic_cast<coCheckboxMenuItem *>(menue_showSelected);

    //std::cerr << "CoordSystems::refresh_coords info: " << m << "  " << m->getState() << std::endl;

    if (m && m->getState())
    {
        std::list<osg::ref_ptr<osg::Node> > selectedNodeList = coVRSelectionManager::instance()->getSelectionList();
        std::list<osg::ref_ptr<osg::Group> > selectedParentList = coVRSelectionManager::instance()->getSelectedParentList();
        if (selectedNodeList.size() != 0)
        {
            for (std::list<osg::ref_ptr<osg::Group> >::iterator iter = selectedParentList.begin();
                 iter != selectedParentList.end(); ++iter)
            {
                if ((*iter) != cover->getObjectsRoot())
                {
                    CoordAxis *localCoord = new CoordAxis();
                    CoordList.push_back(localCoord);

                    localCoord->AddToScenegraph((*iter).get());
                    for (int i = 0; i <= 6; i++)
                    {
                        if (scale_factor[i + 1]->getState())
                        {
                            localCoord->setScaling(ScFactors[i]);
                            localCoord->setLabelValue(ScFactors[i]);
                        }
                    }
                }
            }
        }
    }
    coCheckboxMenuItem *j = dynamic_cast<coCheckboxMenuItem *>(menu_showCoord);
    if (j && j->getState())
    {
        for (int i = 0; i <= 6; i++)
        {
            if (scale_factor[i + 1]->getState())
            {
                globalCoordSys->makeIdentity();
                globalCoordSys->setScaling(ScFactors[i]);
                globalCoordSys->setLabelValue(ScFactors[i]);
            }
        }
    }
    return true;
}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
CoordAxis::CoordAxis()
{
    bool isGlobal = false;
    this->parent = 0;
    HeadNode = new osg::MatrixTransform;
    HeadNode->setName("Axissystem");
    //coVRSelectionManager::markAsHelperNode ( HeadNode.get() );

    if (axisNode.get() == NULL)
    {
        axisNode = new osg::MatrixTransform;
        axisNode->setName("AxisGeometry");
        axisNode->addChild(coVRFileManager::instance()->loadIcon("Axis"));
        coVRLabel *XLabel;
        coVRLabel *YLabel;
        coVRLabel *ZLabel;
        XLabel = new coVRLabel("X", 10.0, 10.0, osg::Vec4(1, 1, 1, 1), osg::Vec4(0.1, 0.1, 0.1, 1));
        XLabel->reAttachTo(axisNode.get());
        XLabel->setPosition(osg::Vec3(100, 0, 0));

        YLabel = new coVRLabel("Y", 10.0, 10.0, osg::Vec4(1, 1, 1, 1), osg::Vec4(0.1, 0.1, 0.1, 1));
        YLabel->reAttachTo(axisNode.get());
        YLabel->setPosition(osg::Vec3(0, 100, 0));

        ZLabel = new coVRLabel("Z", 10.0, 10.0, osg::Vec4(1, 1, 1, 1), osg::Vec4(0.1, 0.1, 0.1, 1));
        ZLabel->reAttachTo(axisNode.get());
        ZLabel->setPosition(osg::Vec3(0, 0, 100));
        isGlobal = true;
    }
    HeadNode->addChild(axisNode.get());
    if (isGlobal)
    {
        ScLabel = new coVRLabel("not setted", 10.0, 50.0, osg::Vec4(1, 1, 1, 1), osg::Vec4(1, 0.1, 0.1, 1));
    }
    else
    {
        ScLabel = new coVRLabel("not setted", 10.0, 50.0, osg::Vec4(1, 1, 1, 1), osg::Vec4(0.1, 0.1, 0.1, 1));
    }
    ScLabel->reAttachTo(HeadNode.get());
    ScLabel->setRotMode(coBillboard::POINT_ROT_EYE); //options: AXIAL_ROT , POINT_ROT_WORLD , POINT_ROT_EYE
    ScLabel->setPosition(osg::Vec3(0, 0, 0));
}
CoordAxis::~CoordAxis()
{
}
//------------------------------------------------------------------------------------------------------------------------------
void CoordAxis::AddToScenegraph(osg::Group *parent)
{
    if (this->parent == 0 && parent != 0)
    {
        this->parent = parent;
        parent->addChild(HeadNode.get());
    }
}
//------------------------------------------------------------------------------------------------------------------------------
void CoordAxis::RemoveFromScenegraph()
{
    if (this->parent.valid())
    {
        this->parent->removeChild(HeadNode.get());
        this->parent = 0;
    }
}
//------------------------------------------------------------------------------------------------------------------------------
void CoordAxis::setScaling(float scalefactor)
{
    osg::Matrix dcsMat;
    osg::Matrix startBaseMat;
    osg::Node *currentNode;

    currentNode = dynamic_cast<osg::MatrixTransform *>(HeadNode.get());
    startBaseMat.makeIdentity();
    while (currentNode != NULL)
    {
        if (dynamic_cast<osg::MatrixTransform *>(currentNode))
        {
            dcsMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
            startBaseMat.postMult(dcsMat);
        }
        if (currentNode->getNumParents() > 0 && currentNode->getParent(0) != cover->getObjectsRoot())
            currentNode = currentNode->getParent(0);
        else
            currentNode = NULL;
    }
    osg::Matrix CompleteMat = startBaseMat;

    osg::Vec3d vect = CompleteMat.getScale();
    vect.set(1. / vect.x(), 1. / vect.y(), 1. / vect.z());
    vect.set(vect * (scalefactor));
    vect.set(vect / (100)); //cause of the size of the original axis-geometry
    HeadNode->setMatrix(osg::Matrix::scale(vect));
}
//------------------------------------------------------------------------------------------------------------------------------
void CoordAxis::makeIdentity()
{
    osg::Matrix tmp;
    tmp.makeIdentity();
    HeadNode->setMatrix(tmp);
}
//------------------------------------------------------------------------------------------------------------------------------
void CoordAxis::setLabelValue(float scalefactor)
{
    char buf[50];
    sprintf(buf, "%.3f", scalefactor);
    ScLabel->setString(buf);
}
//------------------------------------------------------------------------------------------------------------------------------
void CoordAxis::printMatrix()
{
    osg::Matrix ma = HeadNode->getMatrix();
    cout << "/----------------------- " << endl;
    cout << ma(0, 0) << " " << ma(0, 1) << " " << ma(0, 2) << " " << ma(0, 3) << endl;
    cout << ma(1, 0) << " " << ma(1, 1) << " " << ma(1, 2) << " " << ma(1, 3) << endl;
    cout << ma(2, 0) << " " << ma(2, 1) << " " << ma(2, 2) << " " << ma(2, 3) << endl;
    cout << ma(3, 0) << " " << ma(3, 1) << " " << ma(3, 2) << " " << ma(3, 3) << endl;
    cout << "/-----------------------  " << endl;
}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
COVERPLUGIN(CoordSystems)
