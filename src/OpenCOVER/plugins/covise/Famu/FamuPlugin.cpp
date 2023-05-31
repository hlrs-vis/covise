/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>

#include <osg/MatrixTransform>
#include <osg/Geode>

#include "FamuPlugin.h"
#include <cover/RenderObject.h>
#include <cover/coInteractor.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coRowMenuItem.h>
#include <OpenVRUI/coLabelMenuItem.h>

#include <cstring>
#include <string>

using namespace osg;
using std::string;

char *FamuPlugin::currentObjectName = NULL;

void FamuPlugin::newInteractor(const RenderObject *cont, coInteractor *inter)
{

    //std::cout<<"The Cover recieve a interator!!!";

    if (cover->debugLevel(1))
        fprintf(stderr, "\n--- coVRNewInteractor containerName=[%s]\n", cont->getName());

    if (strcmp(inter->getPluginName(), "Famu") == 0)
    {
        // this example can only handle one interactor
        // --> if a new Famu Interactor arrives, it will be regarded
        if (currentObjectName && string(cont->getName()).find(currentObjectName) != 0)
        {
            remove(currentObjectName);
        }

        add(inter);

        //store the basename ModuleName_OUT_PortNo --> erasing index
        string contName(cont->getName());
        int stripIndex = contName.rfind("_");
        contName.erase(stripIndex, contName.length());

        currentObjectName = new char[strlen(contName.c_str()) + 1];
        strcpy(currentObjectName, contName.c_str());
    }
}

void coVRRemoveObject(const char *contName, int r)
{

    if (cover->debugLevel(1))
        fprintf(stderr, "\n--- coVRRemoveObject containerName=[%s]\n", contName);

    if (FamuPlugin::currentObjectName && contName)
    {
        // if object to be deleted is the interactor object then it has to be regarded
        if (string(contName).find(FamuPlugin::currentObjectName) == 0)
        {
            // replace is handeled in add
            if (!r)
            {
                remove(contName);
            }
        }
    }
}

//-----------------------------------------------------------------------------

FamuPlugin::FamuPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    new FamuPlugin\n");

    firsttime = true;

    FamuSubmenu = NULL;
    pinboardButton = NULL;

    // get the parameter names (this is hardcoded)
    bottomLeftParaName = "BottomLeftPoint";
    bottomRightParaName = "BottomRightPoint";
    topLeftParaName = "TopLeftPoint";
    topRightParaName = "TopRightPoint";
    scaleFactorParaName = "ScaleFactor";
    XYParaName = "RotateDegXY";
    YZParaName = "RotateDegYZ";
    ZXParaName = "RotateDegXZ";
    resetParaName = "ResetElectrode";
    moveIsolParaName = "MoveIsolator";
    scaleIsolParaName = "ScaleIsolator";
}

FamuPlugin::~FamuPlugin()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    delete FamuPlugin\n");

    if (pinboardButton)
        delete pinboardButton;
    deleteSubmenu();

    inter->decRefCount();
    inter = NULL;
}

void
FamuPlugin::createSubmenu()
{

    // create the submenu and attach it to the pinboard button
    FamuSubmenu = new coRowMenu("Move Eletrode");
    pinboardButton->setMenu(FamuSubmenu);

    //creat the xCheckbox,ycheckbox and zcheckbox
    moveLabel = new coLabelMenuItem("Move Ctrl.Points");
    FamuSubmenu->add(moveLabel);
    firstCheckbox = new coCheckboxMenuItem("First Point", false, NULL);
    secondCheckbox = new coCheckboxMenuItem("Second Point", false, NULL);
    thirdCheckbox = new coCheckboxMenuItem("Third Point", false, NULL);
    fourthCheckbox = new coCheckboxMenuItem("Fourth Point", false, NULL);

    FamuSubmenu->add(firstCheckbox);
    FamuSubmenu->add(secondCheckbox);
    FamuSubmenu->add(thirdCheckbox);
    FamuSubmenu->add(fourthCheckbox);

    //creat the moveDist slider menuitem
    xDistSlider = new coSliderMenuItem("Move X", -10.0, 10.0, 0.0);
    yDistSlider = new coSliderMenuItem("Move Y", -10.0, 10.0, 0.0);
    zDistSlider = new coSliderMenuItem("Move Z", -10.0, 10.0, 0.0);
    FamuSubmenu->add(xDistSlider);
    FamuSubmenu->add(yDistSlider);
    FamuSubmenu->add(zDistSlider);
    //creat the scaleFactor slider
    scaleLabel = new coLabelMenuItem("Scale the Electrode");
    scaleSlider = new coSliderMenuItem("Factor", 0.1, 5.0, 1.0);
    FamuSubmenu->add(scaleLabel);
    FamuSubmenu->add(scaleSlider);

    //creat the rotate slider
    rotateLabel = new coLabelMenuItem("Rotate the Eletrode");
    XYSlider = new coSliderMenuItem("XY(Deg)", 0.0, 180.0, 0.0);
    YZSlider = new coSliderMenuItem("YZ(Deg)", 0.0, 180.0, 0.0);
    ZXSlider = new coSliderMenuItem("ZX(Deg)", 0.0, 180.0, 0.0);
    FamuSubmenu->add(rotateLabel);
    FamuSubmenu->add(XYSlider);
    FamuSubmenu->add(YZSlider);
    FamuSubmenu->add(ZXSlider);

    //creat the reset checkbox
    resetCheckbox = new coCheckboxMenuItem("Reset Eletrode", false, NULL);
    FamuSubmenu->add(resetCheckbox);
    //creat the menuitem for the Isolator
    isolLabel = new coLabelMenuItem("Transform the Isolator");
    isolCheckbox = new coCheckboxMenuItem("Transform", false, NULL);
    xMoveIsolSlider = new coSliderMenuItem("Move X", -5.0, 5.0, 0.0);
    yMoveIsolSlider = new coSliderMenuItem("Move Y", -5.0, 5.0, 0.0);
    zMoveIsolSlider = new coSliderMenuItem("Move Z", -5.0, 5.0, 0.0);
    scaleIsolSlider = new coSliderMenuItem("Scale", 1.0, 5.0, 1.0);
    FamuSubmenu->add(isolLabel);
    FamuSubmenu->add(isolCheckbox);
    FamuSubmenu->add(xMoveIsolSlider);
    FamuSubmenu->add(yMoveIsolSlider);
    FamuSubmenu->add(zMoveIsolSlider);
    FamuSubmenu->add(scaleIsolSlider);

    //creat the execute button
    exeButton = new coButtonMenuItem("Execute");
    exeButton->setMenuListener(this);
    FamuSubmenu->add(exeButton);
}

void
FamuPlugin::deleteSubmenu()
{

    if (FamuSubmenu)
    {
        delete exeButton;

        delete isolLabel;
        delete isolCheckbox;
        delete xMoveIsolSlider;
        delete yMoveIsolSlider;
        delete zMoveIsolSlider;
        delete scaleIsolSlider;

        delete resetCheckbox;

        delete XYSlider;
        delete YZSlider;
        delete ZXSlider;
        delete rotateLabel;

        delete scaleSlider;
        delete scaleLabel;

        delete xDistSlider;
        delete yDistSlider;
        delete zDistSlider;

        delete firstCheckbox;
        delete secondCheckbox;
        delete thirdCheckbox;
        delete fourthCheckbox;
        delete moveLabel;

        delete FamuSubmenu;
    }
}

void
FamuPlugin::add(coInteractor *in)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\n    FamuPlugin::add\n");

    // if the interactor arrives firsttime
    if (firsttime)
    {
        firsttime = false;

        if (cover->debugLevel(3))
            fprintf(stderr, "firsttime\n");

        // create the button for the pinboard
        pinboardButton = new coSubMenuItem("Famu ...");

        // create submenu
        createSubmenu();

        // add the button to the pinboard
        cover->getMenu()->add(pinboardButton);

        // save the interactor for feedback
        inter = in;
        inter->incRefCount();
    }
    else
    {
        //do some reset works here
        exeButton->setLabel("Execute");

        xDistSlider->setValue(0.0);
        yDistSlider->setValue(0.0);
        zDistSlider->setValue(0.0);
        scaleSlider->setValue(1.0);
        XYSlider->setValue(0.0);
        YZSlider->setValue(0.0);
        ZXSlider->setValue(0.0);
        resetCheckbox->setState(false);
    }
}

void
FamuPlugin::remove(const char * /*objName*/)
// remove is only called by coVRAddObject or coVRRemoveObject
// objName does not need to be regarded (already done)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    FamuPlugin::remove\n");

    cover->getMenu()->remove(pinboardButton); // braucht man das ?
    if (pinboardButton)
        delete pinboardButton;

    deleteSubmenu();

    inter->decRefCount();

    firsttime = true;

    delete[] FamuPlugin::currentObjectName;
    FamuPlugin::currentObjectName = NULL;
}

void FamuPlugin::preFrame()
{
}
void FamuPlugin::postFrame()
{
}

void FamuPlugin::menuEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "FamuPlugin::menuEvent for %s\n", menuItem->getName());

    float xDist, yDist, zDist;
    float *currCoords;

    xDist = xDistSlider->getValue();
    yDist = yDistSlider->getValue();
    zDist = zDistSlider->getValue();
    int n = 0;
    bool executable = false;

    if (firstCheckbox->getState())
    {
        inter->getFloatVectorParam(bottomLeftParaName, n, currCoords);
        currCoords[0] += xDist;
        currCoords[1] += yDist;
        currCoords[2] += zDist;
        inter->setVectorParam(bottomLeftParaName, 3, currCoords);
        executable = true;
    }

    if (secondCheckbox->getState())
    {
        inter->getFloatVectorParam(bottomRightParaName, n, currCoords);
        currCoords[0] += xDist;
        currCoords[1] += yDist;
        currCoords[2] += zDist;
        inter->setVectorParam(bottomRightParaName, 3, currCoords);
        executable = true;
    }

    if (thirdCheckbox->getState())
    {
        inter->getFloatVectorParam(topLeftParaName, n, currCoords);
        currCoords[0] += xDist;
        currCoords[1] += yDist;
        currCoords[2] += zDist;
        inter->setVectorParam(topLeftParaName, 3, currCoords);
        executable = true;
    }

    if (fourthCheckbox->getState())
    {
        inter->getFloatVectorParam(topRightParaName, n, currCoords);
        currCoords[0] += xDist;
        currCoords[1] += yDist;
        currCoords[2] += zDist;
        inter->setVectorParam(topRightParaName, 3, currCoords);
        executable = true;
    }
    if (scaleSlider->getValue() != 1)
    {
        inter->setSliderParam(scaleFactorParaName, 0.1, 5.0, scaleSlider->getValue());
        executable = true;
    }
    if (XYSlider->getValue() != 0)
    {
        inter->setSliderParam(XYParaName, 0.0, 180.0, XYSlider->getValue());
        executable = true;
    }
    if (YZSlider->getValue() != 0)
    {
        inter->setSliderParam(YZParaName, 0.0, 180.0, YZSlider->getValue());
        executable = true;
    }
    if (ZXSlider->getValue() != 0)
    {
        inter->setSliderParam(ZXParaName, 0.0, 180.0, ZXSlider->getValue());
        executable = true;
    }

    if (resetCheckbox->getState())
    {

        inter->setBooleanParam(resetParaName, 1);
        executable = true;
    }
    if (isolCheckbox->getState())
    {
        executable = true;

        inter->setVectorParam(moveIsolParaName, xMoveIsolSlider->getValue(), yMoveIsolSlider->getValue(), zMoveIsolSlider->getValue());

        inter->setVectorParam(scaleIsolParaName, 1.0, scaleIsolSlider->getValue(), scaleIsolSlider->getValue());
    }
    if (executable)
    {
        exeButton->setLabel("Thinking...");

        inter->executeModule();
    }
}

void FamuPlugin::menuReleaseEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "FamuPlugin::menuReleaseEvent for %s\n", menuItem->getName());
}

COVERPLUGIN(FamuPlugin)
