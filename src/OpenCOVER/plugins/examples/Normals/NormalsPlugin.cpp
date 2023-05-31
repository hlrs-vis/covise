/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2005 ZAIK  **
 **                                                                          **
 ** Description: Normals Plugin  (draw normals)                              **
 **                                                                          **
 ** Author: Martin Aumueller (aumueller@uni-koeln.de)                        **
 **                                                                          **
 \****************************************************************************/

#include "NormalsPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>
#include "Normals.h"
#include <osgGA/GUIEventAdapter>

NormalsPlugin::NormalsPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //fprintf(stderr,"NormalsPlugin::NormalsPlugin\n");
    normalsState = NORM_NONE;
}

bool NormalsPlugin::init()
{
    tuiTab = new coTUITab("Normals", coVRTui::instance()->mainFolder->getID());
    tuiTab->setPos(0, 0);

    faceNorm = new coTUIToggleButton("Face Normals", tuiTab->getID());
    vertNorm = new coTUIToggleButton("Vertex Normals", tuiTab->getID());
    faceNorm->setPos(0, 0);
    faceNorm->setEventListener(this);
    vertNorm->setPos(1, 0);
    vertNorm->setEventListener(this);

    scaleValue = 1.f;
    scaleLabel = new coTUILabel("Scale factor (log10)", tuiTab->getID());
    scaleLabel->setPos(0, 1);
    scaleSlider = new coTUIFloatSlider("Scale", tuiTab->getID());
    scaleSlider->setPos(1, 1);
    scaleSlider->setEventListener(this);
    scaleSlider->setMin(-5.f);
    scaleSlider->setMax(5.f);
    scaleSlider->setValue(log(scaleValue) / log(10.f));

    update = new coTUIButton("Update", tuiTab->getID());
    update->setPos(2, 2);

    applyState();

    return true;
}

// this is called if the plugin is removed at runtime
NormalsPlugin::~NormalsPlugin()
{
    //fprintf(stderr,"NormalsPlugin::~NormalsPlugin\n");
    cover->getObjectsRoot()->removeChild(vertexNormals.get());
    cover->getObjectsRoot()->removeChild(faceNormals.get());

    delete faceNorm;
    delete vertNorm;
    delete update;
    delete scaleLabel;
    delete scaleSlider;
    delete tuiTab;
}

void NormalsPlugin::tabletEvent(coTUIElement *tUIItem)
{
    bool changed = false;

    int s = normalsState;
    if (tUIItem == faceNorm)
    {
        changed = true;
        if (faceNorm->getState())
        {
            s |= NORM_FACE;
        }
        else
        {
            s &= ~NORM_FACE;
        }
    }
    if (tUIItem == vertNorm)
    {
        changed = true;
        if (vertNorm->getState())
        {
            s |= NORM_VERTEX;
        }
        else
        {
            s &= ~NORM_VERTEX;
        }
    }
    normalsState = NormalsState(s);

    if (tUIItem == scaleSlider)
    {
        changed = true;
        scaleValue = powf(10.f, scaleSlider->getValue());
    }

    if (changed)
        applyState();
}

void NormalsPlugin::applyState()
{

    std::cerr << "showing normals: ";
    switch (normalsState)
    {
    case NORM_NONE:
        std::cerr << "none";
        break;
    case NORM_VERTEX:
        std::cerr << "vertex";
        break;
    case NORM_FACE:
        std::cerr << "face";
        break;
    case NORM_ALL:
        std::cerr << "face + vertex";
        break;
    default:
        std::cerr << "INVALID";
        break;
    }
    std::cerr << std::endl;

    bool faceState = normalsState & NORM_FACE;
    if (faceNorm->getState() != faceState)
        faceNorm->setState(faceState);
    bool vertState = normalsState & NORM_VERTEX;
    if (vertNorm->getState() != vertState)
        vertNorm->setState(vertState);

    cover->getObjectsRoot()->removeChild(vertexNormals.get());
    cover->getObjectsRoot()->removeChild(faceNormals.get());
    switch (normalsState)
    {
    default:
        cerr << "normalsState = " << normalsState << " not handled" << endl;
        break;
    case NORM_NONE:
        //cerr << "normalsState = NORM_NONE" << endl;
        break;
    case NORM_VERTEX:
        //cerr << "normalsState = NORM_VERTEX" << endl;
        vertexNormals = new osgUtil::VertexNormals(cover->getObjectsRoot(), scaleValue);
        cover->getObjectsRoot()->addChild(vertexNormals.get());
        break;
    case NORM_FACE:
        //cerr << "normalsState = NORM_FACE" << endl;
        faceNormals = new osgUtil::SurfaceNormals(cover->getObjectsRoot(), scaleValue);
        cover->getObjectsRoot()->addChild(faceNormals.get());
        break;
    case NORM_ALL:
        //cerr << "normalsState = NORM_ALL" << endl;
        vertexNormals = new osgUtil::VertexNormals(cover->getObjectsRoot(), scaleValue);
        faceNormals = new osgUtil::SurfaceNormals(cover->getObjectsRoot(), scaleValue);
        cover->getObjectsRoot()->addChild(vertexNormals.get());
        cover->getObjectsRoot()->addChild(faceNormals.get());
        break;
    }
}

void NormalsPlugin::keyEvent(int type, int keySym, int mod)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN
        && (mod & osgGA::GUIEventAdapter::MODKEY_ALT)
        && keySym == 14)
    {
        // ALT+N pressed
        if (normalsState == NORM_ALL)
        {
            normalsState = NORM_NONE;
        }
        else
            normalsState = NormalsState(int(normalsState) + 1);
    }
    else
    {
        return;
    }

    applyState();
}

COVERPLUGIN(NormalsPlugin)
