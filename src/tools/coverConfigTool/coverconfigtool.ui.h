/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** ui.h extension file, included from the uic-generated form implementation.
 **
 ** If you wish to add, delete or rename functions or slots use
 ** Qt Designer which will update this file, preserving your code. Create an
i** init() function in place of a constructor, and a destroy() function in
** place of a destructor.
*****************************************************************************/
//#include <iostream>
#include "covise.h"
#include <math.h>

//#include <qpoint.h>
//#include <qpointarray.h>
#include <qvalidator.h>
#include <qstring.h>
#include <qmap.h>
#include <qlistbox.h>
#include <qpixmap.h>
#include <qimage.h>
#include <qmessagebox.h>
#include <qfiledialog.h>

#include "projectionarea.h"
#include "computeprojvalues.h"
#include "tracking.h"
#include "host.h"
#include "pipe.h"
#include "window.h"
#include "channel.h"
#include "configfileio.h"
#include "xmlfileio.h"

static ProjectionAreaMap projMap;

HostMap hostMap;

HostMap::Iterator actualHost = 0;

double oldOverlap;
double oldWidth;
double oldHeight;

int flipStateFront = 0;
int flipStateBack = 0;
int flipStateLeft = 0;
int flipStateRight = 0;
int flipStateTop = 0;
int flipStateBottom = 0;

bool imagesInitialized = false;

// flip states of the projection areas
// angles: h, p, r according to rotating around the
// z, x and y axes.
// Note: x points right, y points into the screen, z points up!
// in COVER the order of applying the rotations is rph!

const double flipStateFront0[3] = { 0.0, 0.0, 0.0 };
const double flipStateFront1[3] = { 0.0, 0.0, 90.0 };
const double flipStateFront2[3] = { 0.0, 0.0, 180.0 };
const double flipStateFront3[3] = { 0.0, 0.0, 270.0 };
const double flipStateBack0[3] = { 180.0, 0.0, 0.0 };
const double flipStateBack1[3] = { 180.0, 0.0, 90.0 };
const double flipStateBack2[3] = { 180.0, 0.0, 180.0 };
const double flipStateBack3[3] = { 180.0, 0.0, 270.0 };
const double flipStateLeft0[3] = { 90.0, 0.0, 0.0 };
const double flipStateLeft1[3] = { 90.0, 0.0, 90.0 };
const double flipStateLeft2[3] = { 90.0, 0.0, 180.0 };
const double flipStateLeft3[3] = { 90.0, 0.0, 270.0 };
const double flipStateRight0[3] = { -90.0, 0.0, 0.0 };
const double flipStateRight1[3] = { -90.0, 0.0, 90.0 };
const double flipStateRight2[3] = { -90.0, 0.0, 180.0 };
const double flipStateRight3[3] = { -90.0, 0.0, 270.0 };
const double flipStateTop0[3] = { 0.0, 90.0, 0.0 };
const double flipStateTop1[3] = { 0.0, 90.0, 90.0 };
const double flipStateTop2[3] = { 0.0, 90.0, 180.0 };
const double flipStateTop3[3] = { 0.0, 90.0, 270.0 };
const double flipStateBottom0[3] = { 0.0, -90.0, 0.0 };
const double flipStateBottom1[3] = { 0.0, -90.0, 90.0 };
const double flipStateBottom2[3] = { 0.0, -90.0, 180.0 };
const double flipStateBottom3[3] = { 0.0, -90.0, 270.0 };

/********************************************************************************
 ** init function(s)
 ********************************************************************************/
void coverConfigTool::init()
{
    // add validators to entry fields
    projWidthEdit->setValidator(new QIntValidator(projWidthEdit));
    projHeightEdit->setValidator(new QIntValidator(projHeightEdit));

    projOriginXEdit->setValidator(new QDoubleValidator(projOriginXEdit));
    projOriginYEdit->setValidator(new QDoubleValidator(projOriginYEdit));
    projOriginZEdit->setValidator(new QDoubleValidator(projOriginZEdit));

    projRotation_h_Edit->setValidator(new QDoubleValidator(projRotation_h_Edit));
    projRotation_p_Edit->setValidator(new QDoubleValidator(projRotation_p_Edit));
    projRotation_r_Edit->setValidator(new QDoubleValidator(projRotation_r_Edit));

    newProjWidthEdit->setValidator(new QIntValidator(newProjWidthEdit));
    newProjHeightEdit->setValidator(new QIntValidator(newProjHeightEdit));
    newProjOverlapEdit->setValidator(new QDoubleValidator(newProjOverlapEdit));

    newProjOriginXEdit->setValidator(new QDoubleValidator(newProjOriginXEdit));
    newProjOriginYEdit->setValidator(new QDoubleValidator(newProjOriginYEdit));
    newProjOriginZEdit->setValidator(new QDoubleValidator(newProjOriginZEdit));

    newProjRot_h_Edit->setValidator(new QDoubleValidator(newProjRot_h_Edit));
    newProjRot_p_Edit->setValidator(new QDoubleValidator(newProjRot_p_Edit));
    newProjRot_r_Edit->setValidator(new QDoubleValidator(newProjRot_r_Edit));

    winOriginXEdit->setValidator(new QIntValidator(winOriginXEdit));
    winOriginYEdit->setValidator(new QIntValidator(winOriginYEdit));

    winWidthEdit->setValidator(new QIntValidator(winWidthEdit));
    winHeightEdit->setValidator(new QIntValidator(winHeightEdit));

    leftChannelCornerEdit->setValidator(new QDoubleValidator(leftChannelCornerEdit));
    rightChannelCornerEdit->setValidator(new QDoubleValidator(rightChannelCornerEdit));
    topChannelCornerEdit->setValidator(new QDoubleValidator(topChannelCornerEdit));
    bottomChannelCornerEdit->setValidator(new QDoubleValidator(bottomChannelCornerEdit));

    pipeIndexEdit->setValidator(new QIntValidator(pipeIndexEdit));
    hardPipeEdit->setValidator(new QIntValidator(hardPipeEdit));

    viewerPosXEdit->setValidator(new QIntValidator(viewerPosXEdit));
    viewerPosYEdit->setValidator(new QIntValidator(viewerPosYEdit));
    viewerPosZEdit->setValidator(new QIntValidator(viewerPosZEdit));

    floorHeightEdit->setValidator(new QIntValidator(floorHeightEdit));
    stepSizeEdit->setValidator(new QIntValidator(stepSizeEdit));
    menuPosXEdit->setValidator(new QIntValidator(menuPosXEdit));
    menuPosYEdit->setValidator(new QIntValidator(menuPosYEdit));
    menuPosZEdit->setValidator(new QIntValidator(menuPosZEdit));
    menuOrientHEdit->setValidator(new QDoubleValidator(menuOrientHEdit));
    menuOrientPEdit->setValidator(new QDoubleValidator(menuOrientPEdit));
    menuOrientREdit->setValidator(new QDoubleValidator(menuOrientREdit));
    sceneSizeEdit->setValidator(new QIntValidator(sceneSizeEdit));

    // Add first projection area. This is FRONT0 per default.
    addProjectionArea();

    // initialize widgetStacks
    projWidgetStack->raiseWidget(0);
    configurationWidgetStack->raiseWidget(0);
    overlapAngleWidgetStack->raiseWidget(0);

    // initialize the images of the push-buttons and set image of the
    // flipProjAreaPushButton to front0.
    initializeImages();
    flipProjAreaButton->setPixmap(*front0);

    // enable/disable buttons on pages
    setHelpEnabled(LoadSettings, false);
    setHelpEnabled(Geometry, false);
    setHelpEnabled(Stereo, false);
    setHelpEnabled(HostConfiguration, false);
    setHelpEnabled(GeneralSettings, false);
    setHelpEnabled(TrackerConfig, false);
    setHelpEnabled(trackerOrientation, false);
    setHelpEnabled(SaveSettings, false);
    setFinishEnabled(SaveSettings, true);
    setNextEnabled(HostConfiguration, false);

    // set controlHost and masterHost bool-variables...-
    controlHostSet = false;
    masterHostSet = false;

    numChannels = 0;

    initializeHostWidgets();

    generalSettings = CoverGeneral();
    initializeGeneralSettings();

    tracking = Tracking();
    initializeTracking();

    checkChannelsMatchProjectors();

    trackerOrientationChangedSlot();
}

/*------------------------------------------------------------------------------
 ** initializes images
-------------------------------------------------------------------------------*/
void coverConfigTool::initializeImages()
{

    front0 = new QPixmap(QImage::fromMimeSource("front1.png").smoothScale(90, 90));
    front1 = new QPixmap(QImage::fromMimeSource("front2.png").smoothScale(90, 90));
    front2 = new QPixmap(QImage::fromMimeSource("front3.png").smoothScale(90, 90));
    front3 = new QPixmap(QImage::fromMimeSource("front4.png").smoothScale(90, 90));

    back0 = new QPixmap(QImage::fromMimeSource("back1.png").smoothScale(90, 90));
    back1 = new QPixmap(QImage::fromMimeSource("back2.png").smoothScale(90, 90));
    back2 = new QPixmap(QImage::fromMimeSource("back3.png").smoothScale(90, 90));
    back3 = new QPixmap(QImage::fromMimeSource("back4.png").smoothScale(90, 90));

    left0 = new QPixmap(QImage::fromMimeSource("left1.png").smoothScale(90, 90));
    left1 = new QPixmap(QImage::fromMimeSource("left2.png").smoothScale(90, 90));
    left2 = new QPixmap(QImage::fromMimeSource("left3.png").smoothScale(90, 90));
    left3 = new QPixmap(QImage::fromMimeSource("left4.png").smoothScale(90, 90));

    right0 = new QPixmap(QImage::fromMimeSource("right1.png").smoothScale(90, 90));
    right1 = new QPixmap(QImage::fromMimeSource("right2.png").smoothScale(90, 90));
    right2 = new QPixmap(QImage::fromMimeSource("right3.png").smoothScale(90, 90));
    right3 = new QPixmap(QImage::fromMimeSource("right4.png").smoothScale(90, 90));

    top0 = new QPixmap(QImage::fromMimeSource("top1.png").smoothScale(90, 90));
    top1 = new QPixmap(QImage::fromMimeSource("top2.png").smoothScale(90, 90));
    top2 = new QPixmap(QImage::fromMimeSource("top3.png").smoothScale(90, 90));
    top3 = new QPixmap(QImage::fromMimeSource("top4.png").smoothScale(90, 90));

    bottom0 = new QPixmap(QImage::fromMimeSource("bottom1.png").smoothScale(90, 90));
    bottom1 = new QPixmap(QImage::fromMimeSource("bottom2.png").smoothScale(90, 90));
    bottom2 = new QPixmap(QImage::fromMimeSource("bottom3.png").smoothScale(90, 90));
    bottom3 = new QPixmap(QImage::fromMimeSource("bottom4.png").smoothScale(90, 90));

    fobImage = new QPixmap(QImage::fromMimeSource("fob_image_cropped_400_400.png").smoothScale(271, 271));
    motionstarImage = new QPixmap(QImage::fromMimeSource("motionstar_400_400.png").smoothScale(271, 271));
    polhemusImage = new QPixmap(QImage::fromMimeSource("polhemus_image_271_271.png").smoothScale(271, 271));
    polhemus2Image = new QPixmap(QImage::fromMimeSource("polhemus2_image_271_271.png").smoothScale(271, 271));
    no_image = new QPixmap(QImage::fromMimeSource("no_image.png").smoothScale(271, 271));

    imagesInitialized = true;
}

/*------------------------------------------------------------------------------
 ** initializes host widgets
-------------------------------------------------------------------------------*/
void coverConfigTool::initializeHostWidgets()
{
    if (hostListView->childCount() == 0)
    {
        // disable widgets...
        hostNameEdit->clear();
        masterInterfaceEdit->clear();
        hostNameEdit->setEnabled(false);
        masterInterfaceEdit->setEnabled(false);
        masterHostCheckbox->setEnabled(false);
        controlHostCheckbox->setEnabled(false);
        masterInterfaceLabel->setEnabled(false);
        monoViewCombobox->setEnabled(false);
        trackingSystemCombobox->setEnabled(false);
        monoViewLabel->setEnabled(false);
        trackingSystemLabel->setEnabled(false);
    }
}

/*------------------------------------------------------------------------------
 ** initializes flip states
-------------------------------------------------------------------------------*/
void initializeFlipStates()
{
    flipStateFront = 0;
    flipStateBack = 0;
    flipStateLeft = 0;
    flipStateRight = 0;
    flipStateTop = 0;
    flipStateBottom = 0;
}

/*------------------------------------------------------------------------------
 ** initializes overlap
-------------------------------------------------------------------------------*/
void coverConfigTool::initializeOverlap()
{
    oldOverlap = 0.0;
    newProjOverlapEdit->setText("0");
}

/*------------------------------------------------------------------------------
 ** initializes general settings
-------------------------------------------------------------------------------*/
void coverConfigTool::initializeGeneralSettings()
{
    CoverGeneral genSets2 = generalSettings;

    viewerPosXEdit->setText(QString().setNum(genSets2.getViewerPosX()));
    viewerPosYEdit->setText(QString().setNum(genSets2.getViewerPosY()));
    viewerPosZEdit->setText(QString().setNum(genSets2.getViewerPosZ()));

    floorHeightEdit->setText(QString().setNum(genSets2.getFloorHeight()));
    stepSizeEdit->setText(QString().setNum(genSets2.getStepSize()));

    menuPosXEdit->setText(QString().setNum(genSets2.getMenuPosX()));
    menuPosYEdit->setText(QString().setNum(genSets2.getMenuPosY()));
    menuPosZEdit->setText(QString().setNum(genSets2.getMenuPosZ()));

    menuOrientHEdit->setText(QString().setNum(genSets2.getMenuOrient_h()));
    menuOrientPEdit->setText(QString().setNum(genSets2.getMenuOrient_p()));
    menuOrientREdit->setText(QString().setNum(genSets2.getMenuOrient_r()));

    menuSizeEdit->setText(QString().setNum(genSets2.getMenuSize()));

    sceneSizeEdit->setText(QString().setNum(genSets2.getSceneSize()));

    if (genSets2.getStereoMode() == "active")
    {
        stereoModeCombobox->setCurrentItem(0);
        stereoModeChangedSlot(0);
    }
    else
    {
        stereoModeCombobox->setCurrentItem(1);
        stereoModeChangedSlot(1);
    }

    //MultiPC
    syncModeCombobox->setCurrentItem((int)genSets2.getSyncMode());
    syncProcessCombobox->setCurrentItem((int)genSets2.getSyncProcess());
    serialDeviceEdit->setText(genSets2.getSerialDevice());

    if (hostMap.count() > 0)
        multiPCGroup->setEnabled(true);
    else
        multiPCGroup->setEnabled(false);
}

/*------------------------------------------------------------------------------
 ** initializes tracking configuration
-------------------------------------------------------------------------------*/
void coverConfigTool::initializeTracking()
{
    Tracking tr2 = tracking;

    noConnectedSensorsEdit->setText(QString().setNum(tr2.getNoSensors()));
    headSensorIDEdit->setText(QString().setNum(tr2.getAdrHeadSensor()));
    handSensorIDEdit->setText(QString().setNum(tr2.getAdrHandSensor()));

    transmitterOffsetXEdit->setText(QString().setNum(tr2.getTransmitterOffsetX()));
    transmitterOffsetYEdit->setText(QString().setNum(tr2.getTransmitterOffsetY()));
    transmitterOffsetZEdit->setText(QString().setNum(tr2.getTransmitterOffsetZ()));

    transmitterOrientHEdit->setText(QString().setNum(tr2.getTransmitterOrientH()));
    transmitterOrientPEdit->setText(QString().setNum(tr2.getTransmitterOrientP()));
    transmitterOrientREdit->setText(QString().setNum(tr2.getTransmitterOrientR()));

    headSensorXEdit->setText(QString().setNum(tr2.getHeadSensorOffsetX()));
    headSensorYEdit->setText(QString().setNum(tr2.getHeadSensorOffsetY()));
    headSensorZEdit->setText(QString().setNum(tr2.getHeadSensorOffsetZ()));

    headSensorOrientHEdit->setText(QString().setNum(tr2.getHeadSensorOrientH()));
    headSensorOrientPEdit->setText(QString().setNum(tr2.getHeadSensorOrientP()));
    headSensorOrientREdit->setText(QString().setNum(tr2.getHeadSensorOrientR()));

    handSensorOffsetXEdit->setText(QString().setNum(tr2.getHandSensorOffsetX()));
    handSensorOffsetYEdit->setText(QString().setNum(tr2.getHandSensorOffsetY()));
    handSensorOffsetZEdit->setText(QString().setNum(tr2.getHandSensorOffsetZ()));

    handSensorOrientHEdit->setText(QString().setNum(tr2.getHandSensorOrientH()));
    handSensorOrientPEdit->setText(QString().setNum(tr2.getHandSensorOrientP()));
    handSensorOrientREdit->setText(QString().setNum(tr2.getHandSensorOrientR()));

    transmitterOrientComboboxX->setCurrentItem(tr2.getXDir());
    transmitterOrientComboboxY->setCurrentItem(tr2.getYDir());
    transmitterOrientComboboxZ->setCurrentItem(tr2.getZDir());

    trackingSystemCombobox2->setCurrentItem(tr2.getTrackerType());
    trackerOrientationChangedSlot();

    // change values on tracker options page
    fieldCorrectionXEdit->setText(QString().setNum(
        tr2.getLinearMagneticFieldCorrectionX()));
    fieldCorrectionYEdit->setText(QString().setNum(
        tr2.getLinearMagneticFieldCorrectionY()));
    fieldCorrectionZEdit->setText(QString().setNum(
        tr2.getLinearMagneticFieldCorrectionZ()));

    interpolationFileEdit->setText(tr2.getInterpolationFile());
    debugTrackingCombobox->setCurrentItem(tr2.getDebugTracking());
    debugButtonsCombobox->setCurrentItem(tr2.getDebugButtons());
    debugStationEdit->setText(QString().setNum(tr2.getDebugStation()));
}

/********************************************************************************
 ** functions of page "Geometry"
 ********************************************************************************/

/*------------------------------------------------------------------------------
 ** projectionListBoxItemChangedSlot(QListBoxItem * item):
 **   is called, whenever the actual projectionArea in the projectionListBox
 **   is changed.
 **   sets the line-edits to the values of the selected projection area.
-------------------------------------------------------------------------------*/
void coverConfigTool::projectionListBoxItemChangedSlot(QListBoxItem *item)
{
    // find the item in the projMap
    if (projectionListbox->count() != 0)
    { // not empty
        if (projMap.find(projectionListbox->currentText()) != projMap.end())
        {
            ProjectionArea p = projMap[item->text()];
            double r = p.getRotation_r();

            // display values of projection area
            QString valueString;

            //width
            valueString.setNum(p.getWidth());
            projWidthEdit->setText(valueString);

            //height
            valueString.setNum(p.getHeight());
            projHeightEdit->setText(valueString);

            //origin
            valueString = valueString.setNum(p.getOriginX());
            projOriginXEdit->setText(valueString);

            valueString = valueString.setNum(p.getOriginY());
            projOriginYEdit->setText(valueString);

            valueString = valueString.setNum(p.getOriginZ());
            projOriginZEdit->setText(valueString);

            //rotation
            valueString = valueString.setNum(p.getRotation_h());
            projRotation_h_Edit->setText(valueString);

            valueString = valueString.setNum(p.getRotation_p());
            projRotation_p_Edit->setText(valueString);

            valueString = valueString.setNum(p.getRotation_r());
            projRotation_r_Edit->setText(valueString);

            // change icon on push button

            initializeFlipStates();
            if (imagesInitialized)
            {
                switch (p.getType())
                {
                case FRONT:
                    if (r == flipStateFront0[2])
                    {
                        flipProjAreaButton->setPixmap(*front0);
                        flipStateFront = 0;
                    }
                    else if (r == flipStateFront1[2])
                    {
                        flipProjAreaButton->setPixmap(*front1);
                        flipStateFront = 1;
                    }
                    else if (r == flipStateFront2[2])
                    {
                        flipProjAreaButton->setPixmap(*front2);
                        flipStateFront = 2;
                    }
                    else if (r == flipStateFront3[2])
                    {
                        flipProjAreaButton->setPixmap(*front3);
                        flipStateFront = 3;
                    }
                    else
                        flipProjAreaButton->setPixmap(QPixmap());
                    break;
                case BACK:
                    if (p.getRotation_r() == 0.0)
                    {
                        flipProjAreaButton->setPixmap(*back0);
                        flipStateBack = 0;
                    }
                    else if (p.getRotation_r() == 90.0)
                    {
                        flipProjAreaButton->setPixmap(*back1);
                        flipStateBack = 1;
                    }
                    else if (p.getRotation_r() == 180.0)
                    {
                        flipProjAreaButton->setPixmap(*back2);
                        flipStateBack = 2;
                    }
                    else if (p.getRotation_r() == 270.0)
                    {
                        flipProjAreaButton->setPixmap(*back3);
                        flipStateBack = 3;
                    }
                    else
                        flipProjAreaButton->setPixmap(QPixmap());

                    break;
                case LEFT:
                    if (p.getRotation_r() == 0.0)
                    {
                        flipProjAreaButton->setPixmap(*left0);
                        flipStateLeft = 0;
                    }
                    else if (p.getRotation_r() == 90.0)
                    {
                        flipProjAreaButton->setPixmap(*left1);
                        flipStateLeft = 1;
                    }
                    else if (p.getRotation_r() == 180.0)
                    {
                        flipProjAreaButton->setPixmap(*left2);
                        flipStateLeft = 2;
                    }
                    else if (p.getRotation_r() == 270.0)
                    {
                        flipProjAreaButton->setPixmap(*left3);
                        flipStateLeft = 3;
                    }
                    else
                        flipProjAreaButton->setPixmap(QPixmap());
                    break;
                case RIGHT:
                    if (p.getRotation_r() == 0.0)
                    {
                        flipProjAreaButton->setPixmap(*right0);
                        flipStateRight = 0;
                    }
                    else if (p.getRotation_r() == 90.0)
                    {
                        flipProjAreaButton->setPixmap(*right1);
                        flipStateRight = 1;
                    }
                    else if (p.getRotation_r() == 180.0)
                    {
                        flipProjAreaButton->setPixmap(*right2);
                        flipStateRight = 2;
                    }
                    else if (p.getRotation_r() == 270.0)
                    {
                        flipProjAreaButton->setPixmap(*right3);
                        flipStateRight = 3;
                    }
                    else
                        flipProjAreaButton->setPixmap(QPixmap());
                    break;
                case TOP:
                    if (p.getRotation_r() == 0.0)
                    {
                        flipProjAreaButton->setPixmap(*top0);
                        flipStateTop = 0;
                    }
                    else if (p.getRotation_r() == 90.0)
                    {
                        flipProjAreaButton->setPixmap(*top1);
                        flipStateTop = 1;
                    }
                    else if (p.getRotation_r() == 180.0)
                    {
                        flipProjAreaButton->setPixmap(*top2);
                        flipStateTop = 2;
                    }
                    else if (p.getRotation_r() == 270.0)
                    {
                        flipProjAreaButton->setPixmap(*top3);
                        flipStateTop = 3;
                    }
                    else
                        flipProjAreaButton->setPixmap(QPixmap());
                    break;
                case BOTTOM:
                    if (p.getRotation_r() == 0.0)
                    {
                        flipProjAreaButton->setPixmap(*bottom0);
                        flipStateBottom = 0;
                    }
                    else if (p.getRotation_r() == 90.0)
                    {
                        flipProjAreaButton->setPixmap(*bottom1);
                        flipStateBottom = 1;
                    }
                    else if (p.getRotation_r() == 180.0)
                    {
                        flipProjAreaButton->setPixmap(*bottom2);
                        flipStateBottom = 2;
                    }
                    else if (p.getRotation_r() == 270.0)
                    {
                        flipProjAreaButton->setPixmap(*bottom3);
                        flipStateBottom = 3;
                    }
                    else
                        flipProjAreaButton->setPixmap(QPixmap());
                    break;
                }
            }
        }
        else
        {
            cout << "Selected Item of ProjListBox not found!" << endl;
        }

        // initialize flipStates of projection areas
    }
}

/*------------------------------------------------------------------------------
 ** projectionValuesChanged():
 **   It is called, whenever one of the actual projection values is changed
 **   It saves projection values to the corresponding projection area.
-------------------------------------------------------------------------------*/
void coverConfigTool::projectionValuesChanged()
{
    QString key = projectionListbox->currentText();
    ProjectionArea p = projMap[key];
    p.setWidth(projWidthEdit->text().toInt());
    p.setHeight(projHeightEdit->text().toInt());
    p.setOrigin(projOriginXEdit->text().toDouble(),
                projOriginYEdit->text().toDouble(),
                projOriginZEdit->text().toDouble());
    p.setRotation(projRotation_h_Edit->text().toDouble(),
                  projRotation_p_Edit->text().toDouble(),
                  projRotation_r_Edit->text().toDouble());
    projMap[key] = p;
}

/*------------------------------------------------------------------------------
 ** addProjectionArea():
 **   is called, whenever the addProjection-Button is clicked().
 **   It adds a new projection area with default values and a name corresponding
 **   to the chosen type of projection. (FRONT0, FRONT1,... BACK0, BACK1, ..)
-------------------------------------------------------------------------------*/
void coverConfigTool::addProjectionArea()
{
    if (projTypeCombobox->currentItem() != -1)
    {
        QString baseString;
        QString searchString;
        QString numString;
        bool notfound = false;
        int num = 0;

        baseString = projTypeCombobox->currentText();
        searchString = baseString;
        numString.setNum(num);
        searchString.append(numString);

        while (!notfound)
        {
            if (projMap.find(searchString) != projMap.end())
            {
                num++;
                numString.setNum(num);
                searchString = baseString;
                searchString.append(numString);
            }
            else
            {
                notfound = true;
                cout << "Name of new created projection area: " << searchString << endl;
            }
        }
        ProjectionArea p = ProjectionArea();
        p.setName(searchString);
        p.setWidth(newProjWidthEdit->text().toInt());
        p.setHeight(newProjHeightEdit->text().toInt());
        p.setOrigin(newProjOriginXEdit->text().toDouble(),
                    newProjOriginYEdit->text().toDouble(),
                    newProjOriginZEdit->text().toDouble());
        p.setRotation(newProjRot_h_Edit->text().toDouble(),
                      newProjRot_p_Edit->text().toDouble(),
                      newProjRot_r_Edit->text().toDouble());

        // add the correct projection type
        // switch statement not possible here, because the combobox only
        // contains such items which are reasonable
        if (projTypeCombobox->currentText().contains("FRONT") != 0)
        {
            p.setType(FRONT);
            if (imagesInitialized)
                flipProjAreaButton->setPixmap(*front0);
        }
        else if (projTypeCombobox->currentText().contains("BACK") != 0)
        {
            p.setType(BACK);
            if (imagesInitialized)
                flipProjAreaButton->setPixmap(*back0);
        }
        else if (projTypeCombobox->currentText().contains("LEFT") != 0)
        {
            p.setType(LEFT);
            if (imagesInitialized)
                flipProjAreaButton->setPixmap(*left0);
        }
        else if (projTypeCombobox->currentText().contains("RIGHT") != 0)
        {
            p.setType(RIGHT);
            if (imagesInitialized)
                flipProjAreaButton->setPixmap(*right0);
        }
        else if (projTypeCombobox->currentText().contains("TOP") != 0)
        {
            p.setType(TOP);
            if (imagesInitialized)
                flipProjAreaButton->setPixmap(*top0);
        }
        else if (projTypeCombobox->currentText().contains("BOTTOM") != 0)
        {
            p.setType(BOTTOM);
            if (imagesInitialized)
                flipProjAreaButton->setPixmap(*bottom0);
        }

        projMap[searchString] = p;
        projectionListbox->insertItem(searchString);
        projectionListbox->setCurrentItem(projectionListbox->findItem(searchString));

        // tell other elements of new projection area
        projCombobox->insertItem(searchString);
        projCombobox2->insertItem(searchString);

        // raise propoerties-widget of the projWidgetStack
        projWidgetStack->raiseWidget(0);

        // initialize flipStates...
        initializeFlipStates();

        // enable next-button
        setNextEnabled(Geometry, true);

        // compute number of projectors needed
        stereoModeChangedSlot(stereoModeCombobox->currentItem());
    } // end if
}

/*------------------------------------------------------------------------------
 ** deleteProjection():
 **   is called, whenever the user clicks the deleteProjection-Button
 **   It deletes the projection from wherever it occurs (listbox, projMap, comboboxes).
-------------------------------------------------------------------------------*/
void coverConfigTool::deleteProjectionArea()
{

    // delete projection area from projMap
    projMap.remove(projectionListbox->currentText());

    // delete projection area from first projCombobox1 and projCombobox2
    for (int i = 0; i < projCombobox->count(); i++)
    {
        if (projCombobox->text(i) == projectionListbox->currentText())
        {
            projCombobox->removeItem(i);
            break;
        }
    }

    for (int j = 0; j < projCombobox2->count(); j++)
    {
        if (projCombobox2->text(j) == projectionListbox->currentText())
        {
            projCombobox2->removeItem(j);
            break;
        }
    }

    // delete projection area from projListbox
    projectionListbox->removeItem(projectionListbox->currentItem());
    if (projectionListbox->currentItem() != -1) // at least 2 elements
    {
        projectionListbox->setSelected(projectionListbox->currentItem(), true);
    }

    // disable next button if this was the last projection area
    if (projMap.isEmpty())
        this->setNextEnabled(Geometry, false);

    // compute number of projectors needed
    stereoModeChangedSlot(stereoModeCombobox->currentItem());
}

/*------------------------------------------------------------------------------
 ** setNewProjValues:
 **   Sets the projValues of the selected projection area to h, p, r.
-------------------------------------------------------------------------------*/
void coverConfigTool::setNewProjRotValues(const double *angles)
{
    QString valueString;
    valueString.setNum(angles[0]);
    newProjRot_h_Edit->setText(valueString);
    valueString.setNum(angles[1]);
    newProjRot_p_Edit->setText(valueString);
    valueString.setNum(angles[2]);
    newProjRot_r_Edit->setText(valueString);
}

/*------------------------------------------------------------------------------
 ** setProjValues:
 **   Sets the projValues of the selected projection area to h, p, r.
-------------------------------------------------------------------------------*/
void coverConfigTool::setProjRotValues(const double *angles)
{
    QString valueString;
    valueString.setNum(angles[0]);
    projRotation_h_Edit->setText(valueString);
    valueString.setNum(angles[1]);
    projRotation_p_Edit->setText(valueString);
    valueString.setNum(angles[2]);
    projRotation_r_Edit->setText(valueString);
}

/*------------------------------------------------------------------------------
 ** flipProjArea:
 **   is called when the user clicks the flipProjArea-Button.
 **   It flips the projection area an changes its rotation values.
-------------------------------------------------------------------------------*/
void coverConfigTool::flipProjArea()
{

    // find the item in the projMap
    if (projectionListbox->count() != 0)
    { // not empty
        if (projMap.find(projectionListbox->currentText()) != projMap.end())
        {
            ProjectionArea p = projMap[projectionListbox->currentText()];
            ProjType pType = p.getType();
            //cout<<"type: "<<pType<<endl;
            switch (pType)
            {
            case FRONT: // FRONT
                switch (flipStateFront)
                {
                case 0:
                    flipStateFront++;
                    setProjRotValues(&flipStateFront1[0]);
                    flipProjAreaButton->setPixmap(*front1);
                    break;
                case 1:
                    flipStateFront++;
                    setProjRotValues(&flipStateFront2[0]);
                    flipProjAreaButton->setPixmap(*front2);
                    break;
                case 2:
                    flipStateFront++;
                    setProjRotValues(&flipStateFront3[0]);
                    flipProjAreaButton->setPixmap(*front3);
                    break;
                case 3:
                    flipStateFront = 0;
                    setProjRotValues(&flipStateFront0[0]);
                    flipProjAreaButton->setPixmap(*front0);
                    break;
                }
                break;
            case BACK: //BACK
                switch (flipStateBack)
                {
                case 0:
                    flipStateBack++;
                    setProjRotValues(&flipStateBack1[0]);
                    flipProjAreaButton->setPixmap(*back1);
                    break;
                case 1:
                    flipStateBack++;
                    setProjRotValues(&flipStateBack2[0]);
                    flipProjAreaButton->setPixmap(*back2);
                    break;
                case 2:
                    flipStateBack++;
                    setProjRotValues(&flipStateBack3[0]);
                    flipProjAreaButton->setPixmap(*back3);
                    break;
                case 3:
                    flipStateBack = 0;
                    setProjRotValues(&flipStateBack0[0]);
                    flipProjAreaButton->setPixmap(*back0);
                    break;
                }
                break;
            case LEFT: // LEFT
                switch (flipStateLeft)
                {
                case 0:
                    flipStateLeft++;
                    setProjRotValues(&flipStateLeft1[0]);
                    flipProjAreaButton->setPixmap(*left1);
                    break;
                case 1:
                    flipStateLeft++;
                    setProjRotValues(&flipStateLeft2[0]);
                    flipProjAreaButton->setPixmap(*left2);
                    break;
                case 2:
                    flipStateLeft++;
                    setProjRotValues(&flipStateLeft3[0]);
                    flipProjAreaButton->setPixmap(*left3);
                    break;
                case 3:
                    flipStateLeft = 0;
                    setProjRotValues(&flipStateLeft0[0]);
                    flipProjAreaButton->setPixmap(*left0);
                    break;
                }
                break;
            case RIGHT: //RIGHT
                switch (flipStateRight)
                {
                case 0:
                    flipStateRight++;
                    setProjRotValues(&flipStateRight1[0]);
                    flipProjAreaButton->setPixmap(*right1);
                    break;
                case 1:
                    flipStateRight++;
                    setProjRotValues(&flipStateRight2[0]);
                    flipProjAreaButton->setPixmap(*right2);
                    break;
                case 2:
                    flipStateRight++;
                    setProjRotValues(&flipStateRight3[0]);
                    flipProjAreaButton->setPixmap(*right3);
                    break;
                case 3:
                    flipStateRight = 0;
                    setProjRotValues(&flipStateRight0[0]);
                    flipProjAreaButton->setPixmap(*right0);
                    break;
                }
                break;
            case TOP: //TOP
                switch (flipStateTop)
                {
                case 0:
                    flipStateTop++;
                    setProjRotValues(&flipStateTop1[0]);
                    flipProjAreaButton->setPixmap(*top1);
                    break;
                case 1:
                    flipStateTop++;
                    setProjRotValues(&flipStateTop2[0]);
                    flipProjAreaButton->setPixmap(*top2);
                    break;
                case 2:
                    flipStateTop++;
                    setProjRotValues(&flipStateTop3[0]);
                    flipProjAreaButton->setPixmap(*top3);
                    break;
                case 3:
                    flipStateTop = 0;
                    setProjRotValues(&flipStateTop0[0]);
                    flipProjAreaButton->setPixmap(*top0);
                    break;
                }
                break;
            case BOTTOM: //BOTTOM
                switch (flipStateBottom)
                {
                case 0:
                    flipStateBottom++;
                    setProjRotValues(&flipStateBottom1[0]);
                    flipProjAreaButton->setPixmap(*bottom1);
                    break;
                case 1:
                    flipStateBottom++;
                    setProjRotValues(&flipStateBottom2[0]);
                    flipProjAreaButton->setPixmap(*bottom2);
                    break;
                case 2:
                    flipStateBottom++;
                    setProjRotValues(&flipStateBottom3[0]);
                    flipProjAreaButton->setPixmap(*bottom3);
                    break;
                case 3:
                    flipStateBottom = 0;
                    setProjRotValues(&flipStateBottom0[0]);
                    flipProjAreaButton->setPixmap(*bottom0);
                    break;
                }
                break;
            }
        }
        else
        {
            cout << "Selected Item of ProjListBox not found!" << endl;
        }
    }
}

/*------------------------------------------------------------------------------
 ** flipNewProjArea:
 **   is called when the user clicks the flipNewProjArea-Button.
 **   It flips the new projection area and changes its rotation values.
-------------------------------------------------------------------------------*/
void coverConfigTool::flipNewProjArea()
{
    if (projTypeCombobox->currentText() == "FRONT")
    {
        switch (flipStateFront)
        {
        case 0:
            flipStateFront++;
            setNewProjRotValues(&flipStateFront1[0]);
            flipNewProjAreaButton->setPixmap(*front1);
            break;
        case 1:
            flipStateFront++;
            setNewProjRotValues(&flipStateFront2[0]);
            flipNewProjAreaButton->setPixmap(*front2);

            break;
        case 2:
            flipStateFront++;
            setNewProjRotValues(&flipStateFront3[0]);
            flipNewProjAreaButton->setPixmap(*front3);
            break;
        case 3:
            flipStateFront = 0;
            setNewProjRotValues(&flipStateFront0[0]);
            flipNewProjAreaButton->setPixmap(*front0);
            break;
        }
    }
    else if (projTypeCombobox->currentText() == "BACK")
    {
        switch (flipStateBack)
        {
        case 0:
            flipStateBack++;
            setNewProjRotValues(&flipStateBack1[0]);
            flipNewProjAreaButton->setPixmap(*back1);
            break;
        case 1:
            flipStateBack++;
            setNewProjRotValues(&flipStateBack2[0]);
            flipNewProjAreaButton->setPixmap(*back2);
            break;
        case 2:
            flipStateBack++;
            setNewProjRotValues(&flipStateBack3[0]);
            flipNewProjAreaButton->setPixmap(*back3);
            break;
        case 3:
            flipStateBack = 0;
            setNewProjRotValues(&flipStateBack0[0]);
            flipNewProjAreaButton->setPixmap(*back0);
            break;
        }
    }
    else if (projTypeCombobox->currentText() == "LEFT")
    {
        switch (flipStateLeft)
        {
        case 0:
            flipStateLeft++;
            setNewProjRotValues(&flipStateLeft1[0]);
            flipNewProjAreaButton->setPixmap(*left1);
            break;
        case 1:
            flipStateLeft++;
            setNewProjRotValues(&flipStateLeft2[0]);
            flipNewProjAreaButton->setPixmap(*left2);
            break;
        case 2:
            flipStateLeft++;
            setNewProjRotValues(&flipStateLeft3[0]);
            flipNewProjAreaButton->setPixmap(*left3);
            break;
        case 3:
            flipStateLeft = 0;
            setNewProjRotValues(&flipStateLeft0[0]);
            flipNewProjAreaButton->setPixmap(*left0);
            break;
        }
    }
    else if (projTypeCombobox->currentText() == "RIGHT")
    {
        switch (flipStateRight)
        {
        case 0:
            flipStateRight++;
            setNewProjRotValues(&flipStateRight1[0]);
            flipNewProjAreaButton->setPixmap(*right1);
            break;
        case 1:
            flipStateRight++;
            setNewProjRotValues(&flipStateRight2[0]);
            flipNewProjAreaButton->setPixmap(*right2);
            break;
        case 2:
            flipStateRight++;
            setNewProjRotValues(&flipStateRight3[0]);
            flipNewProjAreaButton->setPixmap(*right3);
            break;
        case 3:
            flipStateRight = 0;
            setNewProjRotValues(&flipStateRight0[0]);
            flipNewProjAreaButton->setPixmap(*right0);
            break;
        }
    }
    else if (projTypeCombobox->currentText() == "TOP")
    {
        switch (flipStateTop)
        {
        case 0:
            flipStateTop++;
            setNewProjRotValues(&flipStateTop1[0]);
            flipNewProjAreaButton->setPixmap(*top1);
            break;
        case 1:
            flipStateTop++;
            setNewProjRotValues(&flipStateTop2[0]);
            flipNewProjAreaButton->setPixmap(*top2);
            break;
        case 2:
            flipStateTop++;
            setNewProjRotValues(&flipStateTop3[0]);
            flipNewProjAreaButton->setPixmap(*top3);
            break;
        case 3:
            flipStateTop = 0;
            setNewProjRotValues(&flipStateTop0[0]);
            flipNewProjAreaButton->setPixmap(*top0);
            break;
        }
    }
    else if (projTypeCombobox->currentText() == "BOTTOM")
    {
        switch (flipStateBottom)
        {
        case 0:
            flipStateBottom++;
            setNewProjRotValues(&flipStateBottom1[0]);
            flipNewProjAreaButton->setPixmap(*bottom1);
            break;
        case 1:
            flipStateBottom++;
            setNewProjRotValues(&flipStateBottom2[0]);
            flipNewProjAreaButton->setPixmap(*bottom2);
            break;
        case 2:
            flipStateBottom++;
            setNewProjRotValues(&flipStateBottom3[0]);
            flipNewProjAreaButton->setPixmap(*bottom3);
            break;
        case 3:
            flipStateBottom = 0;
            setNewProjRotValues(&flipStateBottom0[0]);
            flipNewProjAreaButton->setPixmap(*bottom0);
            break;
        }
    }
}

/*------------------------------------------------------------------------------
 ** projectionListBoxItemChangedSlot():
 **   is called, whenever the user clicks th addNewProjectionButton.
-------------------------------------------------------------------------------*/
void coverConfigTool::addNewProjection()
{
    projWidgetStack->raiseWidget(1);
    oldOverlap = 0.0;
    newProjOverlapEdit->setText("0");
}

/*------------------------------------------------------------------------------
 ** setNewProjValues(int width, int height, double newOrigin[3]):
 ** inserts the new projection values to the line edits.
-------------------------------------------------------------------------------*/
void coverConfigTool::setNewProjValues(int width, int height, double *origin)
{
    QString valueString;

    valueString.setNum(width);
    newProjWidthEdit->setText(valueString);

    valueString.setNum(height);
    newProjHeightEdit->setText(valueString);

    valueString.setNum(origin[0]);
    newProjOriginXEdit->setText(valueString);
    valueString.setNum(origin[1]);
    newProjOriginYEdit->setText(valueString);
    valueString.setNum(origin[2]);
    newProjOriginZEdit->setText(valueString);

    // remember width and height for later computing of new origin
    oldWidth = (double)width;
    oldHeight = (double)height;
}

/*------------------------------------------------------------------------------
 ** computeNewProjValuesSlot():
 **   is called, whenever the whichSide-Combobox, the projCombobox or the
 **   projTypeCombobox change their values.
 **   It creates an object of the ComputeProjValues class which computes
 **   as much values as possible.
-------------------------------------------------------------------------------*/
void coverConfigTool::computeNewProjValuesSlot()
{
    int newWidth = 0;
    int newHeight = 0;
    double newOrigin[3] = { 0, 0, 0 };

    if (projCombobox->currentItem() != -1) // at least one area already exists
    {
        ComputeProjValues comp = ComputeProjValues();
        comp.computeProjDimensions(&projMap,
                                   projCombobox->currentText(),
                                   projTypeCombobox->currentText(),
                                   newProjWhichSideCombobox->currentItem());
        newWidth = comp.getNewWidth();
        newHeight = comp.getNewHeight();
        newOrigin[0] = comp.getNewOriginX();
        newOrigin[1] = comp.getNewOriginY();
        newOrigin[2] = comp.getNewOriginZ();
        setNewProjValues(newWidth, newHeight, &newOrigin[0]);

        if (projCombobox->currentText().contains(projTypeCombobox->currentText()))
        {

            // the types are equal
            overlapAngleWidgetStack->raiseWidget(0);
            //overlap can be adjusted
        }
        else
        {
            // angle between the two areas can be adjusted
            overlapAngleWidgetStack->raiseWidget(1);
            newProjOverlapEdit->setText("0");
        }
    } // end if
    else
    {
        newWidth = 0;
        newHeight = 0;
        newOrigin[0] = 0;
        newOrigin[1] = 0;
        newOrigin[2] = 0;
        setNewProjValues(newWidth, newHeight, &newOrigin[0]);
    }
}

/*------------------------------------------------------------------------------
 ** computeNewRotValues():
 **   is called, whenever the projType, onWhichSide or projCombobox of the
 **   new projection area is changed.
 **   It sets the angles of the rotation according to the slelected projection
 **   type.
-------------------------------------------------------------------------------*/
void coverConfigTool::computeNewRotValues()
{
    if (projTypeCombobox->currentText().contains("FRONT") != 0)
    {
        setNewProjRotValues(&flipStateFront0[0]);
    }
    else if (projTypeCombobox->currentText().contains("BACK") != 0)
    {
        setNewProjRotValues(&flipStateBack0[0]);
    }
    else if (projTypeCombobox->currentText().contains("LEFT") != 0)
    {
        setNewProjRotValues(&flipStateLeft0[0]);
    }
    else if (projTypeCombobox->currentText().contains("RIGHT") != 0)
    {
        setNewProjRotValues(&flipStateRight0[0]);
    }
    else if (projTypeCombobox->currentText().contains("TOP") != 0)
    {
        setNewProjRotValues(&flipStateTop0[0]);
    }
    else if (projTypeCombobox->currentText().contains("BOTTOM") != 0)
    {
        setNewProjRotValues(&flipStateBottom0[0]);
    }
    else
    {
        setNewProjRotValues(&flipStateFront0[0]); // everything 0
    }
}

/*------------------------------------------------------------------------------
 ** newProjTypeChangedSlot():
 **   is called, whenever the projType of the new projection area is changed.
 **   It sets the angles of the rotation according to the slelected projection
 **   type.
-------------------------------------------------------------------------------*/
void coverConfigTool::newProjTypeChangedSlot()
{
    // initialize flipState variables
    initializeFlipStates();
    initializeOverlap();
    computeNewProjValuesSlot();

    // change icon on push button
    if (projTypeCombobox->currentText().contains("FRONT") != 0)
    {
        if (imagesInitialized)
            flipNewProjAreaButton->setPixmap(*front0);
    }
    else if (projTypeCombobox->currentText().contains("BACK") != 0)
    {
        if (imagesInitialized)
            flipNewProjAreaButton->setPixmap(*back0);
    }
    else if (projTypeCombobox->currentText().contains("LEFT") != 0)
    {
        if (imagesInitialized)
            flipNewProjAreaButton->setPixmap(*left0);
    }
    else if (projTypeCombobox->currentText().contains("RIGHT") != 0)
    {
        if (imagesInitialized)
            flipNewProjAreaButton->setPixmap(*right0);
    }
    else if (projTypeCombobox->currentText().contains("TOP") != 0)
    {
        if (imagesInitialized)
            flipNewProjAreaButton->setPixmap(*top0);
    }
    else if (projTypeCombobox->currentText().contains("BOTTOM") != 0)
    {
        if (imagesInitialized)
            flipNewProjAreaButton->setPixmap(*bottom0);
    }
}

/*------------------------------------------------------------------------------
 ** completeProjType:
 **   is called, when the user changes the "on which Side"-Combobox.
 **   It should change the projTypeCombobox, so that only entries are selectable,
 **   which make sense.
-------------------------------------------------------------------------------*/
void coverConfigTool::completeProjType()
{
    QListBox *projTypeListbox = projTypeCombobox->listBox();

    if (projectionListbox->currentItem() != -1) // at least one area already exists
    {
        // enable newProjWhichSideCombobox
        newProjWhichSideCombobox->setEnabled(true);
        projCombobox->setEnabled(true);
        switch (newProjWhichSideCombobox->currentItem())
        {
        case 0: // left
            if (projCombobox->currentText().contains("FRONT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("FRONT");
                projTypeListbox->insertItem("LEFT");
            }
            else if (projCombobox->currentText().contains("BACK") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("BACK");
                projTypeListbox->insertItem("RIGHT");
            }
            else if (projCombobox->currentText().contains("LEFT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("LEFT");
                projTypeListbox->insertItem("BACK");
            }
            else if (projCombobox->currentText().contains("RIGHT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("RIGHT");
                projTypeListbox->insertItem("FRONT");
            }
            else if (projCombobox->currentText().contains("TOP") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("TOP");
                projTypeListbox->insertItem("LEFT");
            }
            else if (projCombobox->currentText().contains("BOTTOM") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("BOTTOM");
                projTypeListbox->insertItem("LEFT");
            }
            break;
        case 1: // right
            if (projCombobox->currentText().contains("FRONT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("FRONT");
                projTypeListbox->insertItem("RIGHT");
            }
            else if (projCombobox->currentText().contains("BACK") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("BACK");
                projTypeListbox->insertItem("LEFT");
            }
            else if (projCombobox->currentText().contains("LEFT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("LEFT");
                projTypeListbox->insertItem("FRONT");
            }
            else if (projCombobox->currentText().contains("RIGHT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("RIGHT");
                projTypeListbox->insertItem("BACK");
            }
            else if (projCombobox->currentText().contains("TOP") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("TOP");
                projTypeListbox->insertItem("RIGHT");
            }
            else if (projCombobox->currentText().contains("BOTTOM") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("BOTTOM");
                projTypeListbox->insertItem("RIGHT");
            }
            break;
        case 2: // above
            if (projCombobox->currentText().contains("FRONT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("FRONT");
                projTypeListbox->insertItem("TOP");
            }
            else if (projCombobox->currentText().contains("BACK") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("BACK");
                projTypeListbox->insertItem("TOP");
            }
            else if (projCombobox->currentText().contains("LEFT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("LEFT");
                projTypeListbox->insertItem("TOP");
            }
            else if (projCombobox->currentText().contains("RIGHT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("RIGHT");
                projTypeListbox->insertItem("TOP");
            }
            else if (projCombobox->currentText().contains("TOP") != 0)
            {
                projTypeListbox->clear();
                cout << "This combination is not allowed!" << endl;
            }
            else if (projCombobox->currentText().contains("BOTTOM") != 0)
            {
                projTypeListbox->clear();
                // we don't allow that!
            }
            break;
        case 3: // under
            if (projCombobox->currentText().contains("FRONT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("FRONT");
                projTypeListbox->insertItem("BOTTOM");
            }
            else if (projCombobox->currentText().contains("BACK") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("BACK");
                projTypeListbox->insertItem("BOTTOM");
            }
            else if (projCombobox->currentText().contains("LEFT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("LEFT");
                projTypeListbox->insertItem("BOTTOM");
            }
            else if (projCombobox->currentText().contains("RIGHT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("RIGHT");
                projTypeListbox->insertItem("BOTTOM");
            }
            else if (projCombobox->currentText().contains("TOP") != 0)
            {
                projTypeListbox->clear();
                // we don't allow that!
            }
            else if (projCombobox->currentText().contains("BOTTOM") != 0)
            {
                projTypeListbox->clear();
                cout << "This combination is not allowed!" << endl;
            }
            break;
        case 4: // opposite
            if (projCombobox->currentText().contains("FRONT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("BACK");
            }
            else if (projCombobox->currentText().contains("BACK") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("FRONT");
            }
            else if (projCombobox->currentText().contains("LEFT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("RIGHT");
            }
            else if (projCombobox->currentText().contains("RIGHT") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("LEFT");
            }
            else if (projCombobox->currentText().contains("TOP") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("BOTTOM");
            }
            else if (projCombobox->currentText().contains("BOTTOM") != 0)
            {
                projTypeListbox->clear();
                projTypeListbox->insertItem("TOP");
            }
            break;
        } // end switch
        projTypeCombobox->setCurrentItem(0);
        //cout<<"projTypeCombobox->currentText(): "<<projTypeCombobox->currentText()<<endl;
        projTypeCombobox->update();

        // compute the values for the new projection area
        computeNewProjValuesSlot();

        // call the projTypeChangedSlot
        newProjTypeChangedSlot();
    }
    else // no projection area exists, user can add any type he wants.
    {
        projTypeListbox->clear();
        projTypeListbox->insertItem("FRONT");
        projTypeListbox->insertItem("BACK");
        projTypeListbox->insertItem("LEFT");
        projTypeListbox->insertItem("RIGHT");
        projTypeListbox->insertItem("TOP");
        projTypeListbox->insertItem("BOTTOM");

        // disable newProjWhichSideCombobox
        newProjWhichSideCombobox->setEnabled(false);
        projCombobox->setEnabled(false);
    } // end if
}

/*------------------------------------------------------------------------------
 ** overlapChanged()
 **   is called, when the user edits the overlap.
 **   It computes the origin of the new projection area
-------------------------------------------------------------------------------*/
void coverConfigTool::overlapChanged()
{
    if (projCombobox->currentItem() != -1) // at least one area already exists
    {
        // find the item in the projMap
        if (projMap.find(projCombobox->currentText()) != projMap.end())
        {
            ProjectionArea p = projMap[projCombobox->currentText()];

            double overlap = newProjOverlapEdit->text().toDouble();
            double newOriginX = newProjOriginXEdit->text().toDouble();
            double newOriginY = newProjOriginYEdit->text().toDouble();
            double newOriginZ = newProjOriginZEdit->text().toDouble();

            int oldOverlapTermW = (int)floor((oldOverlap / 100.0) * (double)p.getWidth());
            int oldOverlapTermH = (int)floor((oldOverlap / 100.0) * (double)p.getHeight());

            int overlapTermW = (int)floor((overlap / 100.0) * (double)p.getWidth())
                               - oldOverlapTermW;
            int overlapTermH = (int)floor((overlap / 100.0) * (double)p.getHeight())
                               - oldOverlapTermH;

            QString valueString;

            // create a new ComputeProjValues and let it compute the new origin.
            ComputeProjValues comp = ComputeProjValues();
            comp.computeProjOverlap(p, &newOriginX, &newOriginY, &newOriginZ,
                                    projTypeCombobox->currentText(),
                                    newProjWhichSideCombobox->currentItem(),
                                    overlapTermW,
                                    overlapTermH);

            // insert new values to the entry fields
            valueString.setNum(newOriginX);
            newProjOriginXEdit->setText(valueString);
            valueString.setNum(newOriginY);
            newProjOriginYEdit->setText(valueString);
            valueString.setNum(newOriginZ);
            newProjOriginZEdit->setText(valueString);

            oldOverlap = overlap;
        }
    }
}

/*------------------------------------------------------------------------------
 ** newProjWidthHeightChangedSlot:
 **   is called, whenever the user changes the width or height of the new
 **   projection area.
 **   It computes the new origin according to the new widht and height and then
 **   recomputes the overlap.
-------------------------------------------------------------------------------*/
void coverConfigTool::newProjWidthHeightChangedSlot()
{
    if (projCombobox->currentItem() != -1) // at least one area already exists
    {
        // find the item in the projMap
        if (projMap.find(projCombobox->currentText()) != projMap.end())
        {
            // first of all set the overlap to zero and remember the old overlap
            QString overlapString = newProjOverlapEdit->text();
            newProjOverlapEdit->setText("0");

            double originX = newProjOriginXEdit->text().toDouble();
            double originY = newProjOriginYEdit->text().toDouble();
            double originZ = newProjOriginZEdit->text().toDouble();

            ProjectionArea p = projMap[projCombobox->currentText()];
            // compute difference of new width and height to old
            // note: diff is negative if the new value is smaller than the old
            //       diff is positive if the new value is greater than the old
            double diffWidth = newProjWidthEdit->text().toDouble() - oldWidth;
            double diffHeight = newProjHeightEdit->text().toDouble() - oldHeight;

            // then compute the new origin with zero-overlap
            ComputeProjValues comp = ComputeProjValues();
            comp.computeNewOrigin(p, &originX, &originY, &originZ,
                                  projTypeCombobox->currentText(),
                                  newProjWhichSideCombobox->currentItem(),
                                  diffWidth, diffHeight);
            QString valueString;

            valueString.setNum(originX);
            newProjOriginXEdit->setText(valueString);
            valueString.setNum(originY);
            newProjOriginYEdit->setText(valueString);
            valueString.setNum(originZ);
            newProjOriginZEdit->setText(valueString);

            oldWidth = newProjWidthEdit->text().toDouble();
            oldHeight = newProjHeightEdit->text().toDouble();

            // at last reset the overlap
            newProjOverlapEdit->setText(overlapString);
        }
    }
}

/*------------------------------------------------------------------------------
 ** backToProjPropertiesSlot:
 **   is called, when the user clicks the back button at the newProjArea-Page.
 **   It raises the "Projection properties"-page of the widget stack.
-------------------------------------------------------------------------------*/
void coverConfigTool::backToProjPropertiesSlot()
{
    projWidgetStack->raiseWidget(0);
}

/********************************************************************************
 ** functions of page "Stereo"
 ********************************************************************************/

/*------------------------------------------------------------------------------
 ** stereoModeChangedSlot(int)
 **   is called, whenever the stereo mode is changed.
 **   Stereo mode can either be active or passive. In acitve mode, one
 **   the pipe itself generates the stereo image and only one projector is
 **   needed for each projection area. In passive mode, two projectors per
 **   projection area generate the images for the left and the right eye.
 **   The number of projectors (= channels) needed for the projections is
 **   stored in the member variable numProjectors.
-------------------------------------------------------------------------------*/
void coverConfigTool::stereoModeChangedSlot(int mode)
{
    if (mode == 0) // active stereo
    {
        numProjectors = (int)projMap.count();

        //monoViewCombobox->setEnabled(false);
        //monoViewLabel->setEnabled(false);
    }
    else // passive mode
    {
        numProjectors = (int)projMap.count() * 2;

        // eneble projector combobox on channel widget

        //monoViewCombobox->setEnabled(true);
        //monoViewLabel->setEnabled(true);
    }
    cout << "NumProjectors: " << numProjectors << endl;

    if (checkChannelsMatchProjectors())
        setNextEnabled(HostConfiguration, true);
    else
        setNextEnabled(HostConfiguration, false);

    generalSettings.setStereoMode(stereoModeCombobox->currentText());
}

/********************************************************************************
 ** functions of page "Host Configuration"
 ********************************************************************************/

/*------------------------------------------------------------------------------
 ** controlHostChangedSlot()
 **   is called, whenever the control host checkbox changes its value.
 **   It enables/unables the other widgets of the page, because a control host
 **   doesn't render.
-------------------------------------------------------------------------------*/
void coverConfigTool::controlHostChangedSlot()
{
    if (controlHostCheckbox->isChecked())
    {
        masterInterfaceEdit->clear();
        masterHostCheckbox->setChecked(false);
        masterHostCheckbox->setEnabled(false);
        masterInterfaceEdit->setEnabled(false);
        masterInterfaceLabel->setEnabled(false);
        trackingSystemLabel->setEnabled(false);
        trackingSystemCombobox->setEnabled(false);
        monoViewLabel->setEnabled(false);
        monoViewCombobox->setEnabled(false);
        controlHostSet = true;
    }
    else
    {
        if (!masterHostSet)
        {
            masterInterfaceEdit->setEnabled(false);
            masterInterfaceLabel->setEnabled(false);
            masterHostCheckbox->setEnabled(true);
        }

        /*if (actualHost.data().isMasterHost())
      {
          trackingSystemLabel->setEnabled(true);
          trackingSystemCombobox->setEnabled(true);
      }
      else
      {
          trackingSystemLabel->setEnabled(false);
          trackingSystemCombobox->setEnabled(false);
      }*/

        if (stereoModeCombobox->currentItem() == 1)
        {
            // passive stereo
            monoViewCombobox->setEnabled(true);
            monoViewLabel->setEnabled(true);
        }
    }
}

/*------------------------------------------------------------------------------
 ** masterHostChangedSlot()
 **   is called, whenever the master host checkbox changes its value.
 **   It enables/unables the masterInterfaceEdit and the controlHostCheckbox.
-------------------------------------------------------------------------------*/
void coverConfigTool::masterHostChangedSlot()
{
    if (masterHostCheckbox->isChecked())
    {
        masterInterfaceEdit->setEnabled(true);
        masterInterfaceLabel->setEnabled(true);
        controlHostCheckbox->setChecked(false);
        controlHostCheckbox->setEnabled(false);
        trackingSystemLabel->setEnabled(true);
        trackingSystemCombobox->setEnabled(true);
        masterHostSet = true;
    }
    else
    {
        masterInterfaceEdit->clear();
        masterInterfaceEdit->setEnabled(false);
        masterInterfaceLabel->setEnabled(false);
        trackingSystemLabel->setEnabled(false);
        trackingSystemCombobox->setEnabled(false);
        trackingSystemCombobox->setCurrentItem((int)TrackingSystemType(NONE));
        if (!controlHostSet)
            controlHostCheckbox->setEnabled(true);
    }
}

/*------------------------------------------------------------------------------
 ** hostViewClicked:
-------------------------------------------------------------------------------*/
void coverConfigTool::hostListViewClicked(QListViewItem *item)
{

    QListViewItem *windowItem = 0;
    QListViewItem *pipeItem = 0;
    QListViewItem *hostItem = 0;

    if (item != 0)
    {
        configurationWidgetStack->raiseWidget(item->depth());
        switch (item->depth())
        {
        case 0: // host
            // find host in hostMap
            if (hostMap.find(item->text(0)) != hostMap.end())
            {
                Host h = hostMap[item->text(0)];
                actualHost = hostMap.find(item->text(0));
                //initializeHostWidgets();
                hostNameEdit->setText(h.getName());
                masterInterfaceEdit->clear();
                // NONE
                monoViewCombobox->setCurrentItem(3);
                // NONE
                trackingSystemCombobox->setCurrentItem((int)TrackingSystemType(NONE));
                if (h.isControlHost())
                {
                    /*if (masterHostSet)
                  {
                  masterHostCheckbox->setEnabled(false);
                  masterInterfaceLabel->setEnabled(false);
                  masterInterfaceEdit->setEnabled(false);
                  }*/

                    // set master host widgets unchecked and disabled.
                    masterHostCheckbox->setEnabled(false);
                    masterHostCheckbox->setChecked(false);
                    masterInterfaceLabel->setEnabled(false);
                    masterInterfaceEdit->setEnabled(false);

                    controlHostCheckbox->setChecked(true);
                    controlHostCheckbox->setEnabled(true);

                    // disable monoview combobox
                    // NONE
                    monoViewCombobox->setCurrentItem(3);
                    monoViewCombobox->setEnabled(false);
                    monoViewLabel->setEnabled(false);

                    // disable trackging system comobobox
                    trackingSystemCombobox->setEnabled(false);
                    trackingSystemLabel->setEnabled(false);

                    // disable addPipeToHost-Button
                    addPipeToHostButton->setEnabled(false);
                }
                else if (h.isMasterHost())
                {
                    controlHostCheckbox->setChecked(false);
                    controlHostCheckbox->setEnabled(false);
                    masterHostCheckbox->setChecked(true);
                    masterHostCheckbox->setEnabled(true);
                    masterInterfaceEdit->setEnabled(true);
                    masterInterfaceLabel->setEnabled(true);
                    masterInterfaceEdit->setText(h.getMasterInterface());
                    // enable addPipeToHost-Button
                    addPipeToHostButton->setEnabled(true);

                    // passive stereo
                    if (stereoModeCombobox->currentItem() == 1)
                    {
                        monoViewCombobox->setEnabled(true);
                        monoViewLabel->setEnabled(true);
                        if (h.getMonoView() == "LEFT")
                            monoViewCombobox->setCurrentItem(0);
                        else if (h.getMonoView() == "RIGHT")
                            monoViewCombobox->setCurrentItem(1);
                        else if (h.getMonoView() == "MIDDLE")
                            monoViewCombobox->setCurrentItem(2);
                        else // NONE
                            monoViewCombobox->setCurrentItem(3);
                    }
                    else // active stereo
                    {
                        // NONE
                        monoViewCombobox->setCurrentItem(3);
                        monoViewCombobox->setEnabled(false);
                        monoViewLabel->setEnabled(false);
                    }

                    // enable tracking system widgets
                    trackingSystemCombobox->setEnabled(true);
                    trackingSystemLabel->setEnabled(true);
                    trackingSystemCombobox->setCurrentItem((int)h.getTrackingSystem());
                    //trackerLabel->setText(h.getTrackingString());
                }
                else // h is wether control nor master host
                {
                    masterHostCheckbox->setChecked(false);
                    controlHostCheckbox->setChecked(false);
                    masterInterfaceEdit->setEnabled(false);
                    masterInterfaceLabel->setEnabled(false);
                    if (masterHostSet)
                    {
                        masterHostCheckbox->setEnabled(false);
                    }
                    else
                    {
                        masterHostCheckbox->setEnabled(true);
                    }
                    if (controlHostSet)
                    {
                        controlHostCheckbox->setEnabled(false);
                    }
                    else
                    {
                        controlHostCheckbox->setEnabled(true);
                    }
                    // passive stereo
                    if (stereoModeCombobox->currentItem() == 1)
                    {
                        monoViewLabel->setEnabled(true);
                        monoViewCombobox->setEnabled(true);
                        if (h.getMonoView() == "LEFT")
                            monoViewCombobox->setCurrentItem(0);
                        else if (h.getMonoView() == "RIGHT")
                            monoViewCombobox->setCurrentItem(1);
                        else if (h.getMonoView() == "MIDDLE")
                            monoViewCombobox->setCurrentItem(2);
                    }
                    else // active stereo
                    {
                        // NONE
                        monoViewCombobox->setCurrentItem(3);
                        monoViewLabel->setEnabled(false);
                        monoViewCombobox->setEnabled(false);
                    }

                    // disable  tracking system combobox because this is a slave host
                    trackingSystemCombobox->setEnabled(false);
                    trackingSystemLabel->setEnabled(false);
                    // enable addPipeToHost-Button
                    addPipeToHostButton->setEnabled(true);
                }
            }
            break;
        case 1: // pipe
            // get parent host
            hostItem = item->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(item->text(0)) != pm->end())
                {
                    //pipe exists in pipe map of host
                    Pipe p = (*pm)[item->text(0)];
                    pipeIndexEdit->setText(QString().setNum(p.getIndex()));
                    hardPipeEdit->setText(QString().setNum(p.getHardPipe()));
                    displayEdit->setText(p.getDisplay());
                }
            }
            break;
        case 2: // window
            // get parent host and pipe
            pipeItem = item->parent();
            hostItem = item->parent()->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(pipeItem->text(0)) != pm->end())
                {
                    //pipe exists in pipe map of host
                    Pipe p = (*pm)[pipeItem->text(0)];
                    WindowMap *wm = p.getWindowMap();
                    if (wm->find(item->text(0)) != wm->end())
                    {
                        Window w = (*wm)[item->text(0)];
                        windowNameEdit->setText(w.getName());
                        windowIndexEdit->setText(QString().setNum(w.getIndex()));
                        winOriginXEdit->setText(QString().setNum(w.getOriginX()));
                        winOriginYEdit->setText(QString().setNum(w.getOriginY()));
                        winWidthEdit->setText(QString().setNum(w.getWidth()));
                        winHeightEdit->setText(QString().setNum(w.getHeight()));
                    }
                }
            }
            break;
        case 3: // channel
            // get parent host, pipe and window
            windowItem = item->parent();
            pipeItem = item->parent()->parent();
            hostItem = item->parent()->parent()->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(pipeItem->text(0)) != pm->end())
                {
                    //pipe exists in pipe map of host
                    Pipe p = (*pm)[pipeItem->text(0)];
                    WindowMap *wm = p.getWindowMap();
                    if (wm->find(windowItem->text(0)) != wm->end())
                    {
                        // window exists in windowMap of pipe
                        Window w = (*wm)[windowItem->text(0)];
                        ChannelMap *cm = w.getChannelMap();
                        if (cm->find(item->text(0)) != cm->end())
                        {
                            // channel exists in channelMap of window
                            Channel c = (*cm)[item->text(0)];
                            Channel c2 = c;

                            // fill entry fields
                            channelNameEdit->setText(c2.getName());
                            channelIndexEdit->setText(QString().setNum(c2.getIndex()));
                            leftChannelCornerEdit->setText(QString().setNum(c2.getLeft()));
                            rightChannelCornerEdit->setText(QString().setNum(c2.getRight()));
                            bottomChannelCornerEdit->setText(QString().setNum(c2.getBottom()));
                            topChannelCornerEdit->setText(QString().setNum(c2.getTop()));

                            // set correct projection area in combobox
                            QListBox *pListbox = projCombobox2->listBox();

                            //cout<<"hostListViewClick Channel - c2.getProjectionArea()->getName(): "<<c2.getProjectionArea()->getName()<<endl;

                            if (c2.getProjectionArea() != 0)
                            {
                                QListBoxItem *it = pListbox->findItem(
                                    c2.getProjectionArea()->getName());
                                if (it != 0)
                                {
                                    projCombobox2->setCurrentItem(pListbox->index(it));
                                    // we got to tell the widget manually, that the
                                    // projCombobox2 has changed values
                                    channelValuesChangedSlot();
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
    }
}

/*------------------------------------------------------------------------------
 ** newHostSlot:
 ** 	is called, when the user presses the "new Host" Button
 ** 	cleans the entry fields
-------------------------------------------------------------------------------*/
void coverConfigTool::newHostSlot()
{
    //make shure we stored changes of another host name
    if (hostListView->currentItem() != 0)
    {
        if (hostListView->currentItem()->text(0) != hostNameEdit->text())
            hostValuesChangedSlot();
    }

    QString baseString;
    QString hostName;
    QString numString;
    bool notfound = false;
    int num = 0;

    baseString = "Host";
    hostName = baseString;
    numString.setNum(num);
    hostName.append(numString);

    while (!notfound)
    {
        if (hostMap.find(hostName) != hostMap.end())
        {
            // host name does exist yet
            num++;
            numString.setNum(num);
            hostName = baseString;
            hostName.append(numString);
        }
        else
        {
            notfound = true;
            cout << "Name of new created host: " << hostName << endl;
        }
    }
    Host h = Host();
    h.setName(hostName);

    hostMap[hostName] = h;
    actualHost = hostMap.find(hostName);

    // if this is the first host, set it to master by default.
    /*cout<<"hostMap.count(): "<<hostMap.count()<<endl;
   if( (int) hostMap.count() == 1)
   {
   // first host -> set as master host
   h.setMasterHost(true);
   masterHostCheckbox->setChecked(true);
   masterHostSet = true;
   }*/
    QListViewItem *hostItem = new QListViewItem(hostListView, 0);
    hostItem->setText(0, hostName);
    hostListView->setCurrentItem(hostItem);
    hostListView->setSelected(hostItem, true);

    // enable/disable widgets...

    // enable hostNameEdit
    hostNameEdit->setEnabled(true);
    hostNameEdit->setText(hostName);
    hostNameLabel->setEnabled(true);

    // disable trackingSystemCombobox by default
    trackingSystemCombobox->setEnabled(false);
    trackingSystemLabel->setEnabled(false);

    // uncheck and clear checkboxes / edits
    controlHostCheckbox->setChecked(false);
    masterHostCheckbox->setChecked(false);
    masterInterfaceEdit->clear();
    monoViewCombobox->setCurrentItem(3); // NONE
    trackingSystemCombobox->setCurrentItem((int)TrackingSystemType(NONE));

    // look if we got to enable master and control checkboxes
    masterInterfaceEdit->setEnabled(false);
    masterInterfaceLabel->setEnabled(false);
    if (masterHostSet)
    {
        masterHostCheckbox->setEnabled(false);
    }
    else
    {
        masterHostCheckbox->setEnabled(true);
    }
    if (controlHostSet)
    {
        controlHostCheckbox->setEnabled(false);
    }
    else
    {
        controlHostCheckbox->setEnabled(true);
    }

    // look if we got to enable mono view
    if (stereoModeCombobox->currentItem() == 1) // passive stereo
    {
        monoViewLabel->setEnabled(true);
        monoViewCombobox->setEnabled(true);
    }
    else
    {
        monoViewLabel->setEnabled(false);
        monoViewCombobox->setEnabled(false);
    }

    // enable addPipeToHost Button
    addPipeToHostButton->setEnabled(true);

    //eventually enable MultiPC group
    if (hostMap.count() > 1)
        multiPCGroup->setEnabled(true);
    else
        multiPCGroup->setEnabled(false);
}

/*------------------------------------------------------------------------------
 ** hostValuesChangedSlot:
 **	is called whenever one of the  host values change.
-------------------------------------------------------------------------------*/
void coverConfigTool::hostValuesChangedSlot()
{
    if (!hostNameEdit->text().isEmpty())
    {
        QString newHostName = hostNameEdit->text();
        QString oldHostName = hostListView->currentItem()->text(0);

        if ((oldHostName != newHostName) && (hostMap.find(newHostName) != hostMap.end()))
        {
            // name exists yet -> error message!
            QMessageBox::warning(
                this,
                tr("Name of host already exists! -- coverConfigTool"),
                tr("A host called %1 already exists. "
                   "Please rename the new host and try again.")
                    .arg(newHostName),
                tr("&Ok"),
                QString::null, 0);
            hostNameEdit->setFocus();
        }
        else
        {
            if (hostMap.find(oldHostName) != hostMap.end())
            {
                // host does exist
                Host h = hostMap[oldHostName];
                actualHost = hostMap.find(oldHostName);

                // check if host is control host
                if (controlHostCheckbox->isChecked())
                {
                    masterHostCheckbox->setEnabled(false);
                    masterInterfaceEdit->setEnabled(false);
                    masterInterfaceLabel->setEnabled(false);
                    trackingSystemLabel->setEnabled(false);
                    trackingSystemCombobox->setEnabled(false);
                    monoViewLabel->setEnabled(false);
                    monoViewCombobox->setEnabled(false);

                    // disable addPipeToHost-Button
                    addPipeToHostButton->setEnabled(false);

                    h.setMasterHost(false);
                    h.setControlHost(true);
                    controlHostSet = true;
                }
                else if (masterHostCheckbox->isChecked())
                {
                    masterInterfaceEdit->setEnabled(true);
                    masterInterfaceLabel->setEnabled(true);
                    controlHostCheckbox->setEnabled(false);
                    trackingSystemLabel->setEnabled(true);
                    trackingSystemCombobox->setEnabled(true);

                    // enable addPipeToHost-Button
                    addPipeToHostButton->setEnabled(true);

                    h.setControlHost(false);
                    h.setMasterHost(true);
                    if (masterInterfaceEdit->text().length() == 0)
                    {
                        h.setMasterInterface(oldHostName);
                        masterInterfaceEdit->setText(oldHostName);
                    }
                    else
                    {
                        h.setMasterInterface(masterInterfaceEdit->text());
                    }
                    h.setTrackingSystem((TrackingSystemType)
                                            trackingSystemCombobox->currentItem());
                    // set things on tracker config page
                    //trackerLabel->setText(h.getTrackingString());

                    // passive Stereo
                    if (stereoModeCombobox->currentItem() == 1)
                    {
                        monoViewLabel->setEnabled(true);
                        monoViewCombobox->setEnabled(true);
                        h.setMonoView(monoViewCombobox->currentText());
                    }
                    else
                    {
                        monoViewLabel->setEnabled(false);
                        monoViewCombobox->setEnabled(false);
                    }
                    masterHostSet = true;
                }
                else
                {
                    // wether control nor master host checkbox is checked
                    // look if host was control or master host before
                    // user changed values
                    if (h.isControlHost())
                    {
                        h.setControlHost(false);
                        controlHostSet = false;
                    }
                    if (h.isMasterHost())
                    {
                        h.setMasterHost(false);
                        masterHostSet = false;
                        trackingSystemCombobox->setEnabled(false);
                        trackingSystemLabel->setEnabled(false);
                    }

                    // enable addPipeToHost-Button
                    addPipeToHostButton->setEnabled(true);

                    masterInterfaceLabel->setEnabled(false);
                    masterInterfaceEdit->setEnabled(false);
                    if (!masterHostSet)
                        masterHostCheckbox->setEnabled(true);
                    else
                        masterHostCheckbox->setEnabled(false);
                    if (!controlHostSet)
                        controlHostCheckbox->setEnabled(true);
                    else
                        controlHostCheckbox->setEnabled(false);

                    h.setTrackingSystem((TrackingSystemType)
                                            trackingSystemCombobox->currentItem());

                    // passive Stereo
                    if (stereoModeCombobox->currentItem() == 1)
                    {
                        monoViewLabel->setEnabled(true);
                        monoViewCombobox->setEnabled(true);
                        h.setMonoView(monoViewCombobox->currentText());
                    }
                    // why set the master host here true???
                    //masterHostSet = true;
                }
                if (oldHostName != newHostName)
                {
                    h.setName(newHostName);
                    hostMap[newHostName] = h;
                    hostMap.remove(oldHostName);
                    hostListView->currentItem()->setText(0, newHostName);
                    hostListView->sort();
                }
                else
                    hostMap[newHostName] = h;
            }
            if (checkChannelsMatchProjectors())
                setNextEnabled(HostConfiguration, true);
            else
                setNextEnabled(HostConfiguration, false);
        }
    }
    else // no host name!
    {
        QMessageBox::warning(
            this,
            tr("No host name! -- coverConfigTool"),
            tr("No host name!\n"
               "Please enter a name for the new host."),
            tr("&Ok"),
            QString::null, 0);
        hostNameEdit->setFocus();
    }
}

/*------------------------------------------------------------------------------
 ** deleteHostSlot:
 **   is called, whenever the user clicks the deleteHost-Button
 **   It deletes the host from the hostListView and the hostMap.
-------------------------------------------------------------------------------*/
void coverConfigTool::deleteHostSlot()
{
    // delete host and all its pipes, windows channels from hostMap
    if (hostListView->currentItem() != 0) // item exists
    {
        if (hostMap.find(hostListView->currentItem()->text(0)) != hostMap.end())
        {
            // host exists...
            Host h = hostMap[hostListView->currentItem()->text(0)];
            if (h.isControlHost())
                controlHostSet = false;
            if (h.isMasterHost())
                masterHostSet = false;

            // get corresponding number of channels of host
            PipeMap *pm = h.getPipeMap();
            PipeMap::Iterator pIt;
            WindowMap *wm;
            WindowMap::Iterator winIt;
            for (pIt = pm->begin(); pIt != pm->end(); ++pIt)
            {
                wm = pIt.data().getWindowMap();
                for (winIt = wm->begin(); winIt != wm->end(); ++winIt)
                {
                    numChannels -= winIt.data().getNumChannels();
                }
            }

            hostMap.remove(hostListView->currentItem()->text(0));
        }

        // delete host from hostListView
        delete hostListView->currentItem();

        if (hostListView->childCount() == 0)
        {
            addPipeToHostButton->setEnabled(false);
            initializeHostWidgets();
        }
        else
            hostListView->setCurrentItem(hostListView->firstChild());
        if (checkChannelsMatchProjectors())
            setNextEnabled(HostConfiguration, true);
        else
            setNextEnabled(HostConfiguration, false);

        //eventually enable MultiPC group
        if (hostMap.count() > 1)
            multiPCGroup->setEnabled(true);
        else
            multiPCGroup->setEnabled(false);
    }
}

/*------------------------------------------------------------------------------
 ** addPipeToHost:
 **   is called, whenever the user clicks the addPipeToHost-Button
 **   It adds a new pipe with default name to the current host.
-------------------------------------------------------------------------------*/
void coverConfigTool::addPipeToHost()
{
    // raise the pipe widget
    configurationWidgetStack->raiseWidget(PipeConfig);

    // get parent host
    QListViewItem *actualItem = hostListView->currentItem();

    if (actualItem != 0) // a host exists
    {
        QListViewItem *hostItem;
        Host h;
        switch (actualItem->depth())
        {
        case 0: // host
            hostItem = actualItem;
            break;
        case 1: // pipe
            hostItem = actualItem->parent();
            break;
        case 2: // window
            hostItem = actualItem->parent()->parent();
            break;
        case 3: // channel
            hostItem = actualItem->parent()->parent()->parent();
            break;
        default:
            hostItem = 0;
        }

        if (hostMap.find(hostItem->text(0)) != hostMap.end())
        {
            // host found in hostMap
            h = hostMap[hostItem->text(0)];

            int numPipes = h.getNumPipes();
            PipeMap *pm = h.getPipeMap();

            // create new pipe
            Pipe p = Pipe();
            QString numString;
            numString.setNum(numPipes);
            //if numPipes is 0 -> Pipe0
            QString pipeName = QString("Pipe").append(numString);

            // look if pipe name already exists
            int i = 0;
            while (pm->find(pipeName) != pm->end())
            {
                numString.setNum(numString.toInt() + i);
                pipeName = QString("Pipe").append(numString);
                i++;
            }

            p.setIndex(numPipes);
            p.setHardPipe(numPipes);

            // add pipe to host
            h.addPipe(pipeName, p);
            hostMap[h.getName()] = h;

            // insert index in pipeIndexEdit and hardPipeEdit
            pipeIndexEdit->setText(numString);
            hardPipeEdit->setText(numString);

            QListViewItem *pipeItem = new QListViewItem(hostItem);
            hostItem->setOpen(true);
            pipeItem->setText(0, pipeName);
            hostListView->setCurrentItem(pipeItem);
            hostListView->setSelected(pipeItem, true);

            // enable addWindow Button
            addWindowButton->setEnabled(true);
        }
    }
}

/*------------------------------------------------------------------------------
 ** pipeValuesChangedSlot()
 **   is called, whenever the user changes the values of a pipe
 **   It saves the new values to the pipeMap of the corresponding host.
-------------------------------------------------------------------------------*/
void coverConfigTool::pipeValuesChangedSlot()
{
    QListViewItem *item = hostListView->currentItem();
    if (item != 0)
    {
        // item exists
        if (item->depth() == 1)
        {
            // we really got a pipe item
            QListViewItem *hostItem = item->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(item->text(0)) != pm->end())
                {
                    // pipe found in PipeMap of host
                    cout << "pipe found in pipe map of host: " << item->text(0) << endl;
                    Pipe p = (*pm)[item->text(0)];
                    cout << "old hardPipe: " << p.getHardPipe() << endl;
                    p.setIndex(pipeIndexEdit->text().toInt());
                    p.setHardPipe(hardPipeEdit->text().toInt());
                    p.setDisplay(displayEdit->text());

                    // save the pipe to the Pipe Map
                    //pm->insert(item->text(0), p);
                    (*pm)[item->text(0)] = p;

                    // save host to host map
                    hostMap[hostItem->text(0)] = h;
                }
            }
        }
    }
}

/*------------------------------------------------------------------------------
 ** deletePipeSlot()
 **   is called, whenever the user deletes a pipe.
 **   It deletes the pipe from the pipeMap of the correspondig host and from
 **   the hostListView
-------------------------------------------------------------------------------*/
void coverConfigTool::deletePipeSlot()
{
    QListViewItem *item = hostListView->currentItem();
    if (item != 0)
    {
        // item exists
        if (item->depth() == 1)
        {
            // we really got a pipe item
            QListViewItem *hostItem = item->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(item->text(0)) != pm->end())
                {
                    // pipe found in PipeMap of host
                    Pipe p = (*pm)[item->text(0)];

                    // decrease numChannels...
                    WindowMap *wm;
                    WindowMap::Iterator winIt;
                    wm = p.getWindowMap();
                    for (winIt = wm->begin(); winIt != wm->end(); ++winIt)
                    {
                        numChannels -= winIt.data().getNumChannels();
                    }

                    // remove pipe
                    pm->remove(item->text(0));
                    hostMap[hostItem->text(0)] = h;
                }
                if (pm->count() == 0)
                {
                    addWindowButton->setEnabled(false);
                }
            }
            // delete pipe from hostListView
            delete item;

            if (hostItem->childCount() == 0)
                hostListView->setCurrentItem(hostItem);

            if (checkChannelsMatchProjectors())
                setNextEnabled(HostConfiguration, true);
            else
                setNextEnabled(HostConfiguration, false);
        }
    }
}

/*------------------------------------------------------------------------------
 ** addWindow:
 **   is called, whenever the user clicks the addWindow-Button
 **   It adds a new windwow with default name to the current host.
-------------------------------------------------------------------------------*/
void coverConfigTool::addWindowSlot()
{
    // raise the window widget
    configurationWidgetStack->raiseWidget(WindowConfig);

    // get parent host
    QListViewItem *actualItem = hostListView->currentItem();

    if (actualItem != 0) // item exists
    {
        QListViewItem *hostItem = 0;
        QListViewItem *pipeItem = 0;
        Host h;
        Pipe p;
        switch (actualItem->depth())
        {
        case 0: // host
            // not allowed
            break;
        case 1: // pipe
            hostItem = actualItem->parent();
            pipeItem = actualItem;
            break;
        case 2: // window
            hostItem = actualItem->parent()->parent();
            pipeItem = actualItem->parent();
            break;
        case 3: // channel
            // not allowed!
            break;
        }

        if (hostMap.find(hostItem->text(0)) != hostMap.end())
        {
            // host found in hostMap
            h = hostMap[hostItem->text(0)];
            PipeMap *pm = h.getPipeMap();

            if (pm->find(pipeItem->text(0)) != pm->end())
            {
                // pipe found in pipeMap
                Pipe p = (*pm)[pipeItem->text(0)];
                WindowMap *wm = p.getWindowMap();
                // the windows shall be numerated consecutively for each host!
                //int numWindows = h.getNumWindows();
                int numWindows = p.getNumWindows();

                // create new window
                Window w = Window();
                QString numString;
                numString.setNum(numWindows);
                QString winName = QString("Window").append(numString);

                // look if win name already exists
                int i = 0;
                while (wm->find(winName) != wm->end())
                {
                    numString.setNum(numString.toInt() + i);
                    winName = QString("Window").append(numString);
                    i++;
                }
                w.setIndex(numWindows);
                w.setName(winName);
                w.setWidth(winDefaultWidthEdit->text().toInt());
                w.setHeight(winDefaultHeightEdit->text().toInt());

                // add win to pipe
                p.addWindow(winName, w);
                (*pm)[pipeItem->text(0)] = p;
                hostMap[h.getName()] = h;

                // add window to the hostListView
                QListViewItem *winItem = new QListViewItem(pipeItem);
                pipeItem->setOpen(true);
                winItem->setText(0, winName);
                hostListView->setCurrentItem(winItem);
                hostListView->setSelected(winItem, true);

                // fill entry fields with default values
                windowIndexEdit->setText(numString);
                windowNameEdit->setText(winName);

                // enable addChannel Button
                addChannelButton->setEnabled(true);
            }
        }
    }
}

/*------------------------------------------------------------------------------
 ** windowValuesChangedSlot()
 **   is called, whenever the user changes the values of a window
 **   It saves the new values to the windowMap of the corresponding pipe.
-------------------------------------------------------------------------------*/
void coverConfigTool::windowValuesChangedSlot()
{
    // get pipe and host
    QListViewItem *item = hostListView->currentItem();
    if (item != 0)
    {
        // item exists
        if (item->depth() == 2)
        {
            // we really got a window item
            QListViewItem *pipeItem = item->parent();
            QListViewItem *hostItem = item->parent()->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(pipeItem->text(0)) != pm->end())
                {
                    // pipe found in PipeMap of host
                    Pipe p = (*pm)[pipeItem->text(0)];
                    WindowMap *wm = p.getWindowMap();

                    if (wm->find(item->text(0)) != wm->end())
                    {
                        //window found in windowMap
                        Window w = (*wm)[item->text(0)];
                        w.setIndex(windowIndexEdit->text().toInt());
                        w.setName(windowNameEdit->text());
                        w.setOriginX(winOriginXEdit->text().toInt());
                        w.setOriginY(winOriginYEdit->text().toInt());
                        w.setWidth(winWidthEdit->text().toInt());
                        w.setHeight(winHeightEdit->text().toInt());

                        if (item->text(0) != w.getName())
                        {
                            (*wm)[w.getName()] = w;
                            wm->remove(item->text(0));

                            // save changed name to hostListView
                            item->setText(0, w.getName());
                        }
                        else
                        {
                            (*wm)[item->text(0)] = w;
                        }

                        // save the changed values to the maps
                        (*pm)[pipeItem->text(0)] = p;
                        hostMap[hostItem->text(0)] = h;
                    }
                }
            }
        }
    }
}

/*------------------------------------------------------------------------------
 ** deleteWindowSlot()
 **   is called, whenever the user clicks the deleteWindow-Button.
 **   It deletes the window from the windowMap of the correspondig pipe.
-------------------------------------------------------------------------------*/
void coverConfigTool::deleteWindowSlot()
{
    QListViewItem *item = hostListView->currentItem();
    if (item != 0)
    {
        // item exists
        if (item->depth() == 2)
        {
            // we really got a window item
            QListViewItem *pipeItem = item->parent();
            QListViewItem *hostItem = item->parent()->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(pipeItem->text(0)) != pm->end())
                {
                    // pipe found in PipeMap of host
                    Pipe p = (*pm)[pipeItem->text(0)];
                    WindowMap *wm = p.getWindowMap();
                    if (wm->find(item->text(0)) != wm->end())
                    {
                        // window found in WindowMap of pipe
                        Window w = (*wm)[item->text(0)];
                        numChannels -= w.getNumChannels();

                        wm->remove(item->text(0));
                        (*pm)[pipeItem->text(0)] = p;
                        hostMap[hostItem->text(0)] = h;
                    }
                    if (pm->count() == 0)
                        addChannelButton->setEnabled(false);
                }
            }
            // delete window from hostListView
            delete item;

            if (hostItem->childCount() == 0)
                hostListView->setCurrentItem(hostItem);

            if (checkChannelsMatchProjectors())
                setNextEnabled(HostConfiguration, true);
            else
                setNextEnabled(HostConfiguration, false);
        }
    }
}

/*------------------------------------------------------------------------------
 ** addChannelSlot:
 **   is called, whenever the user clicks the addChannel-Button
 **   It adds a new channel with default values to the current window.
-------------------------------------------------------------------------------*/
void coverConfigTool::addChannelSlot()
{
    // raise the channel widget
    configurationWidgetStack->raiseWidget(ChannelConfig);

    // get parent host
    QListViewItem *actualItem = hostListView->currentItem();

    if (actualItem != 0) // item exists
    {
        QListViewItem *hostItem = 0;
        QListViewItem *pipeItem = 0;
        QListViewItem *windowItem = 0;
        Host h;
        Pipe p;
        switch (actualItem->depth())
        {
        case 0: // host
            // not allowed!
            break;
        case 1: // pipe
            // not allowed!
            break;
        case 2: // window
            hostItem = actualItem->parent()->parent();
            pipeItem = actualItem->parent();
            windowItem = actualItem;
            break;
        case 3: // channel
            hostItem = actualItem->parent()->parent()->parent();
            pipeItem = actualItem->parent()->parent();
            windowItem = actualItem->parent();
            break;
        }

        if (hostMap.find(hostItem->text(0)) != hostMap.end())
        {
            // host found in hostMap
            h = hostMap[hostItem->text(0)];
            PipeMap *pm = h.getPipeMap();

            if (pm->find(pipeItem->text(0)) != pm->end())
            {
                // pipe found in pipeMap
                Pipe p = (*pm)[pipeItem->text(0)];
                WindowMap *wm = p.getWindowMap();

                if (wm->find(windowItem->text(0)) != wm->end())
                {
                    // window found in WindowMap of pipe
                    Window w = (*wm)[windowItem->text(0)];
                    ChannelMap *cm = w.getChannelMap();
                    // the channels shall be numerated consecutively!
                    //int numChannels = h.getNumChannels();
                    int numChannels = w.getNumChannels();

                    // create new channel
                    Channel c = Channel();
                    QString numString;
                    numString.setNum(numChannels);
                    QString channelName = QString("Channel").append(numString);

                    // look if channel name already exists
                    int i = 0;
                    while (cm->find(channelName) != cm->end())
                    {
                        numString.setNum(numString.toInt() + i);
                        channelName = QString("Channel").append(numString);
                        i++;
                    }
                    c.setIndex(numChannels);
                    c.setName(channelName);

                    // add channel to window and store maps
                    w.addChannel(channelName, c);
                    (*wm)[windowItem->text(0)] = w;
                    (*pm)[pipeItem->text(0)] = p;
                    hostMap[h.getName()] = h;

                    // add channel to hostListView
                    QListViewItem *channelItem = new QListViewItem(windowItem);
                    windowItem->setOpen(true);
                    channelItem->setText(0, channelName);
                    hostListView->setCurrentItem(channelItem);
                    hostListView->setSelected(channelItem, true);

                    // fill entry fields with default values
                    channelNameEdit->setText(channelName);
                    channelIndexEdit->setText(numString);
                    leftChannelCornerEdit->setText("0.0");
                    rightChannelCornerEdit->setText("1.0");
                    bottomChannelCornerEdit->setText("0.0");
                    topChannelCornerEdit->setText("1.0");

                    coverConfigTool::numChannels++;

                    if (checkChannelsMatchProjectors())
                        setNextEnabled(HostConfiguration, true);
                    else
                        setNextEnabled(HostConfiguration, false);
                }
            }
        }
    }
}

/*------------------------------------------------------------------------------
 ** channelValuesChangedSlot()
 **   is called, whenever the user changes the values of a channel
 **   It saves the new values to the channelMap of the corresponding window.
-------------------------------------------------------------------------------*/
void coverConfigTool::channelValuesChangedSlot()
{
    // get pipe, host and window
    QListViewItem *item = hostListView->currentItem();
    if (item != 0)
    {
        // item exists
        if (item->depth() == 3)
        {
            // we really got a channel item
            QListViewItem *windowItem = item->parent();
            QListViewItem *pipeItem = item->parent()->parent();
            QListViewItem *hostItem = item->parent()->parent()->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(pipeItem->text(0)) != pm->end())
                {
                    // pipe found in PipeMap of host
                    Pipe p = (*pm)[pipeItem->text(0)];
                    WindowMap *wm = p.getWindowMap();

                    if (wm->find(windowItem->text(0)) != wm->end())
                    {
                        //window found in windowMap
                        Window w = (*wm)[windowItem->text(0)];
                        ChannelMap *cm = w.getChannelMap();

                        if (cm->find(item->text(0)) != cm->end())
                        {
                            // channel found in channelMap
                            Channel c = (*cm)[item->text(0)];
                            c.setName(channelNameEdit->text());
                            c.setIndex(channelIndexEdit->text().toInt());
                            c.setLeft(leftChannelCornerEdit->text().toDouble());
                            c.setRight(rightChannelCornerEdit->text().toDouble());
                            c.setBottom(bottomChannelCornerEdit->text().toDouble());
                            c.setTop(topChannelCornerEdit->text().toDouble());

                            // set corresponding projection area
                            if (projMap.find(projCombobox2->currentText()) != projMap.end())
                            {
                                c.setProjectionArea(&projMap[projCombobox2->currentText()]);
                                //cout<<"channelValuesChanged - c.getProjArea: "<<c.getProjectionArea()->getName()<<endl;
                            }

                            // new name
                            if (item->text(0) != c.getName())
                            {
                                (*cm)[c.getName()] = c;
                                cm->remove(item->text(0));

                                // save changed name to hostListView
                                item->setText(0, c.getName());
                            }
                            else // name of channel didn't change
                            {
                                (*cm)[item->text(0)] = c;
                            }

                            // save the changed values to the maps
                            (*wm)[windowItem->text(0)] = w;
                            (*pm)[pipeItem->text(0)] = p;
                            hostMap[hostItem->text(0)] = h;

                            if (checkChannelsMatchProjectors())
                                setNextEnabled(HostConfiguration, true);
                            else
                                setNextEnabled(HostConfiguration, false);
                        }
                    }
                }
            }
        }
    }
}

/*------------------------------------------------------------------------------
 ** deleteChannelSlot()
 **   is called, whenever the user clicks the deleteWindow-Button.
 **   It deletes the window from the windowMap of the correspondig pipe.
-------------------------------------------------------------------------------*/
void coverConfigTool::deleteChannelSlot()
{
    QListViewItem *item = hostListView->currentItem();
    if (item != 0)
    {
        // item exists
        if (item->depth() == 3)
        {
            // we really got a channel item
            QListViewItem *windowItem = item->parent();
            QListViewItem *pipeItem = item->parent()->parent();
            QListViewItem *hostItem = item->parent()->parent()->parent();
            if (hostMap.find(hostItem->text(0)) != hostMap.end())
            {
                // host exists in hostMap
                Host h = hostMap[hostItem->text(0)];
                PipeMap *pm = h.getPipeMap();
                if (pm->find(pipeItem->text(0)) != pm->end())
                {
                    // pipe found in PipeMap of host
                    Pipe p = (*pm)[pipeItem->text(0)];
                    WindowMap *wm = p.getWindowMap();
                    if (wm->find(windowItem->text(0)) != wm->end())
                    {
                        // window found in WindowMap of pipe
                        Window w = (*wm)[windowItem->text(0)];
                        ChannelMap *cm = w.getChannelMap();
                        if (cm->find(item->text(0)) != cm->end())
                        {
                            cm->remove(item->text(0));
                            (*wm)[windowItem->text(0)] = w;
                            (*pm)[pipeItem->text(0)] = p;
                            hostMap[hostItem->text(0)] = h;

                            numChannels--;

                            if (checkChannelsMatchProjectors())
                                setNextEnabled(HostConfiguration, true);
                            else
                                setNextEnabled(HostConfiguration, false);
                        }
                    }
                }
            }
            // delete channel from hostListView
            delete item;

            if (hostItem->childCount() == 0)
                hostListView->setCurrentItem(hostItem);

            if (checkChannelsMatchProjectors())
                setNextEnabled(HostConfiguration, true);
            else
                setNextEnabled(HostConfiguration, false);
        }
    }
}

/*------------------------------------------------------------------------------
 ** bool checkChannelsMatchProjectors():
 **	checks if the channels match the projectors.
 **      It only checks numbers, it doesn't check if every projection
 ** 	area has got the correct amount of corresponding channels.
 **    	It also sets the status bar text.
-------------------------------------------------------------------------------*/
bool coverConfigTool::checkChannelsMatchProjectors()
{
    if (!masterHostSet)
    {
        statusText->setText("No master host set!");
        statusText->setPaletteForegroundColor("red");
        return false;
    }
    if (numChannels < numProjectors) // numChannels smaller than numProjectors
    {
        statusText->setText("Not enough channels!");
        statusText->setPaletteForegroundColor("red");
        return false;
    }
    else if (numChannels > numProjectors)
    {
        statusText->setText("Too much channels!");
        statusText->setPaletteForegroundColor("red");
        return false;
    }
    else
    {
        QListBoxItem *it;
        HostMap::Iterator hostIt;
        PipeMap::Iterator pipeIt;
        WindowMap::Iterator winIt;
        ChannelMap::Iterator chIt;
        PipeMap *pm;
        WindowMap *wm;
        ChannelMap *cm;

        int *projArray = new int[projMap.count()];
        // initialize
        for (unsigned int i = 0; i < projMap.count(); i++)
        {
            projArray[i] = 0;
        }

        // count left and right channels and fill up projArrray
        int numLeftChannels = 0;
        int numRightChannels = 0;
        for (hostIt = hostMap.begin(); hostIt != hostMap.end(); hostIt++)
        {
            if (hostIt.data().getMonoView() == "LEFT")
                numLeftChannels++;
            else if (hostIt.data().getMonoView() == "RIGHT")
                numRightChannels++;

            pm = hostIt.data().getPipeMap();
            for (pipeIt = pm->begin(); pipeIt != pm->end(); pipeIt++)
            {
                wm = pipeIt.data().getWindowMap();
                for (winIt = wm->begin(); winIt != wm->end(); winIt++)
                {
                    cm = winIt.data().getChannelMap();
                    for (chIt = cm->begin(); chIt != cm->end(); chIt++)
                    {
                        if (chIt.data().getProjectionArea() != 0)
                        {
                            it = projectionListbox->findItem(
                                chIt.data().getProjectionArea()->getName());
                            if (it != 0)
                            {
                                (projArray[projectionListbox->index(it)])++;
                            }
                        }
                        else
                        {
                            statusText->setText("Projection area of channel not found!");
                            statusText->setPaletteForegroundColor("red");
                            delete projArray;
                            return false;
                        }
                    }
                }
            }
        }
        if (stereoModeCombobox->currentItem() == 1) //passive stereo
        {
            // check if there are the same number of left and right channels
            if (numLeftChannels != numRightChannels)
            {
                statusText->setText("not the same number of left and right monoviews.");
                statusText->setPaletteForegroundColor("red");
                delete projArray;
                return false;
            }
            else
            {
                for (unsigned int i = 0; i < projMap.count(); i++)
                {
                    cout << " projArray[" << i << "]: " << projArray[i] << endl;
                    if (projArray[i] != 2)
                    {
                        statusText->setText("Wrong assignment of channels to projection areas!");
                        statusText->setPaletteForegroundColor("red");
                        delete projArray;
                        return false;
                    }
                }
                statusText->setText("Everything O.k.");
                statusText->setPaletteForegroundColor("black");
                delete projArray;
                return true; // everything's o.k.
            }
        }
        else // active stereo
        {
            // check if every projection area has got a channel
            for (unsigned int i = 0; i < projMap.count(); i++)
            {
                cout << " projArray[" << i << "]: " << projArray[i] << endl;
                if (projArray[i] != 1)
                {
                    statusText->setText("Wrong assignment of channels to projection areas!");
                    statusText->setPaletteForegroundColor("red");
                    delete projArray;
                    return false;
                }
            }
            statusText->setText("Everything O.k.");
            statusText->setPaletteForegroundColor("black");
            delete projArray;
            return true; // everything's o.k.
        }
    }
}

/********************************************************************************
 ** functions of page "General Settings"
 ********************************************************************************/

/*------------------------------------------------------------------------------
 ** generalSettingsChangedSlot()
 **	is called whenever the one of the entry fields of the general settings
 ** 	page changes its values.
-------------------------------------------------------------------------------*/
void coverConfigTool::generalSettingsChangedSlot()
{
    generalSettings.setViewerPosX(viewerPosXEdit->text().toInt());
    generalSettings.setViewerPosY(viewerPosYEdit->text().toInt());
    generalSettings.setViewerPosZ(viewerPosZEdit->text().toInt());

    generalSettings.setFloorHeight(floorHeightEdit->text().toInt());
    generalSettings.setStepSize(stepSizeEdit->text().toInt());

    generalSettings.setMenuPosX(menuPosXEdit->text().toInt());
    generalSettings.setMenuPosY(menuPosYEdit->text().toInt());
    generalSettings.setMenuPosZ(menuPosZEdit->text().toInt());

    generalSettings.setMenuOrient_h(menuOrientHEdit->text().toDouble());
    generalSettings.setMenuOrient_p(menuOrientPEdit->text().toDouble());
    generalSettings.setMenuOrient_r(menuOrientREdit->text().toDouble());

    generalSettings.setMenuSize(menuSizeEdit->text().toDouble());

    generalSettings.setSceneSize(sceneSizeEdit->text().toInt());

    //MultiPC

    if (multiPCGroup->isEnabled())
    {
        generalSettings.setSyncMode((SyncModeType)syncModeCombobox->currentItem());
        generalSettings.setSyncProcess((SyncProcessType)syncProcessCombobox->currentItem());
        generalSettings.setSerialDevice(serialDeviceEdit->text());
        generalSettings.setCommando0(commando0Edit->text());
    }
}

/********************************************************************************
 ** functions of page "TrackerConfig"
 ********************************************************************************/
/*------------------------------------------------------------------------------
 ** trackerConfigurationChangedSlot()
 **	is called whenever one of the entry fields of the tracker configuration
 ** 	page changes its values.
-------------------------------------------------------------------------------*/
void coverConfigTool::trackerConfigurationChangedSlot()
{
    tracking.setNoSensors(noConnectedSensorsEdit->text().toInt());
    tracking.setAdrHeadSensor(headSensorIDEdit->text().toInt());
    tracking.setAdrHandSensor(handSensorIDEdit->text().toInt());

    tracking.setTransmitterOffsetX(transmitterOffsetXEdit->text().toDouble());
    tracking.setTransmitterOffsetY(transmitterOffsetYEdit->text().toDouble());
    tracking.setTransmitterOffsetZ(transmitterOffsetZEdit->text().toDouble());
    tracking.setTransmitterOrientH(transmitterOrientHEdit->text().toDouble());
    tracking.setTransmitterOrientP(transmitterOrientPEdit->text().toDouble());
    tracking.setTransmitterOrientR(transmitterOrientREdit->text().toDouble());

    tracking.setHeadSensorOffsetX(headSensorXEdit->text().toDouble());
    tracking.setHeadSensorOffsetY(headSensorYEdit->text().toDouble());
    tracking.setHeadSensorOffsetZ(headSensorZEdit->text().toDouble());
    tracking.setHeadSensorOrientH(headSensorOrientHEdit->text().toDouble());
    tracking.setHeadSensorOrientP(headSensorOrientPEdit->text().toDouble());
    tracking.setHeadSensorOrientR(headSensorOrientREdit->text().toDouble());

    tracking.setHandSensorOffsetX(handSensorOffsetXEdit->text().toDouble());
    tracking.setHandSensorOffsetY(handSensorOffsetYEdit->text().toDouble());
    tracking.setHandSensorOffsetZ(handSensorOffsetZEdit->text().toDouble());
    tracking.setHandSensorOrientH(handSensorOrientHEdit->text().toDouble());
    tracking.setHandSensorOrientP(handSensorOrientPEdit->text().toDouble());
    tracking.setHandSensorOrientR(handSensorOrientREdit->text().toDouble());

    // save changes made on tracker configuration page...
    tracking.setLinearMagneticFieldCorrection(fieldCorrectionXEdit->text().toDouble(),
                                              fieldCorrectionYEdit->text().toDouble(),
                                              fieldCorrectionZEdit->text().toDouble());
    tracking.setInterpolationFile(interpolationFileEdit->text());
    tracking.setDebugTracking(DebugTrackingType(debugTrackingCombobox->currentItem()));
    tracking.setDebugButtons(debugButtonsCombobox->currentItem());
    tracking.setDebugStation(debugStationEdit->text().toInt());
}

/*------------------------------------------------------------------------------
 ** interpolationFileLoadSlot()
 **	is called whenever the open interpolation file button is pressed.
-------------------------------------------------------------------------------*/
void coverConfigTool::interpolationFileLoadSlot()
{
    QString s = QFileDialog::getOpenFileName(
        ".",
        "Data files (*.data)",
        this,
        "open interpolation file"
        "Choose a file");
    interpolationFileEdit->setText(s);
}

/*------------------------------------------------------------------------------
 ** trackerOrientationChangedSlot()
 **	is called whenever one of the entry fields of the tracker orientation
 ** 	page changes its values.
-------------------------------------------------------------------------------*/
void coverConfigTool::trackerOrientationChangedSlot()
{

    // set tracker image according to the selected tracking system...
    if (trackingSystemCombobox2->currentText() == "POLHEMUS LONGRANGE")
    {
        tracking.setTrackerType(POLHEMUS_LONGRANGE);
        trackerPixmap->setPixmap(*polhemusImage);
        trackerPixmap2->setPixmap(*polhemusImage);
    }
    else if (trackingSystemCombobox2->currentText() == "POLHEMUS")
    {
        tracking.setTrackerType(POLHEMUS2);
        trackerPixmap->setPixmap(*polhemus2Image);
        trackerPixmap2->setPixmap(*polhemus2Image);
    }
    else if (trackingSystemCombobox2->currentText() == "FLOCK OF BIRDS")
    {
        tracking.setTrackerType(FLOCK_OF_BIRDS);
        trackerPixmap->setPixmap(*fobImage);
        trackerPixmap2->setPixmap(*fobImage);
    }
    else if (trackingSystemCombobox2->currentText() == "MOTIONSTAR")
    {
        tracking.setTrackerType(MOTIONSTAR2);
        trackerPixmap->setPixmap(*motionstarImage);
        trackerPixmap2->setPixmap(*motionstarImage);
    }
    else
    {
        tracking.setTrackerType(OTHER);
        trackerPixmap->setPixmap(*no_image);
        trackerPixmap2->setPixmap(*no_image);
    }

    // compute orientation of z-Direction:
    DirectionType xDir = tracking.getDirectionType(
        transmitterOrientComboboxX->currentText());
    DirectionType yDir = tracking.getDirectionType(
        transmitterOrientComboboxY->currentText());
    if (tracking.checkDirections(xDir, yDir))
    {
        //cout<<"computing zDirection now."<<endl;
        DirectionType zDir = tracking.getZDirection(xDir, yDir);
        transmitterOrientComboboxZ->setCurrentText(tracking.getDirectionTypeString(zDir));
        transmitterOrientStatusText->setText("Everything o.k.");
        transmitterOrientStatusText->setPaletteForegroundColor("black");

        double h, p, r;
        tracking.computeOrientation(xDir, yDir, zDir, &h, &p, &r);

        transmitterOrientHEdit->setText(QString().setNum(h));
        transmitterOrientPEdit->setText(QString().setNum(p));
        transmitterOrientREdit->setText(QString().setNum(r));

        tracking.setDirections(xDir, yDir, zDir);
    }
    else
    {
        transmitterOrientStatusText->setText("Configuration not possible!");
        transmitterOrientStatusText->setPaletteForegroundColor("red");
    }
}

/********************************************************************************
 ** functions of page "General Settings"
 ********************************************************************************/

/*------------------------------------------------------------------------------
 ** configFileDialogSaveSlot()
 **	opens a QFileDialog where the user can choose a filename for
 **  	the config file.
-------------------------------------------------------------------------------*/
void coverConfigTool::configFileDialogSaveSlot()
{
    QString s = QFileDialog::getSaveFileName(
        ".",
        "ConfigFiles (*.config)",
        this,
        "save file dialog"
        "Please enter a filename");
    configFileNameEdit->setText(s);
    saveConfigSlot();
}

/*------------------------------------------------------------------------------
 ** saveConfigSlot()
 **	creates a new configFileIO object and stores the settings to the file
 **	with filename entered by the user.
-------------------------------------------------------------------------------*/
void coverConfigTool::saveConfigSlot()
{
    generalSettings.setStereoMode(stereoModeCombobox->currentText());

    if (configFileNameEdit->text().length() > 0) // user choosed a file name
    {
        ConfigFileIO configIO = ConfigFileIO();
        configIO.setHostMap(&hostMap);
        configIO.setProjMap(&projMap);
        configIO.setGeneralSettings(&generalSettings);
        configIO.setTracking(&tracking);

        if (configIO.saveConfigFile(configFileNameEdit->text()))
            cout << "file written succesfullly." << endl;
        else
            cout << "file couldn't be written!" << endl;
    }
}

/*------------------------------------------------------------------------------
 ** xmlFileDialogSaveSlot()
 **	opens a QFileDialog where the user can choose a filename for
 **  	the xml file.
-------------------------------------------------------------------------------*/
void coverConfigTool::xmlFileDialogSaveSlot()
{
    QString s = QFileDialog::getSaveFileName(
        ".",
        "XML (*.xml)",
        this,
        "save file dialog"
        "Please enter a filename");
    xmlFileNameEdit->setText(s);
    saveXMLSlot();
}

/*------------------------------------------------------------------------------
 ** saveXMLSlot()
 **	creates a new xmlFileIO object and stores the settings to the file
 **	with filename entered by the user.
-------------------------------------------------------------------------------*/
void coverConfigTool::saveXMLSlot()
{
    generalSettings.setStereoMode(stereoModeCombobox->currentText());

    if (xmlFileNameEdit->text().length() > 0) // user choosed a file name
    {
        XMLFileIO xmlIO = XMLFileIO();
        xmlIO.setHostMap(&hostMap);
        xmlIO.setProjMap(&projMap);
        xmlIO.setGeneralSettings(&generalSettings);
        xmlIO.setTracking(&tracking);

        if (xmlIO.saveXMLFile(xmlFileNameEdit->text()))
            cout << "file written succesfullly." << endl;
        else
            cout << "file couldn't be written!" << endl;
    }
}

/*------------------------------------------------------------------------------
 ** xmlFileDialogLoadSlot()
 **	opens a QFileDialog where the user can choose an xml file to laod.
-------------------------------------------------------------------------------*/
void coverConfigTool::xmlFileDialogLoadSlot()
{
    QString s = QFileDialog::getOpenFileName(
        ".",
        "XML files (*.xml)",
        this,
        "open xml file"
        "Choose a file");
    xmlLoadFileNameEdit->setText(s);
    loadXMLSlot();
}

/*------------------------------------------------------------------------------
 ** laodXMLSlot
 **	loads the settings from the xml file choosed by the user.
-------------------------------------------------------------------------------*/
void coverConfigTool::loadXMLSlot()
{

    QString message;

    if (xmlLoadFileNameEdit->text().length() > 0) // user choosed a file name
    {
        XMLFileIO xmlIO = XMLFileIO();

        if (xmlIO.loadXMLFile(xmlLoadFileNameEdit->text(),
                              &message))
        {
            // initialize settings
            hostMap.clear();
            generalSettings = CoverGeneral();
            projMap.clear();
            masterHostSet = false;
            controlHostSet = false;
            hostListView->clear();
            projectionListbox->clear();
            projCombobox->clear();
            projCombobox2->clear();

            numProjectors = 0;
            numChannels = 0;

            cout << message << endl;
            xmlLoadStatus->setPaletteForegroundColor("black");
            xmlLoadStatus->setText(message);

            // projection areas
            projMap = (*xmlIO.getProjMap());
            cout << "projMap.count(): " << projMap.count() << endl;
            ProjectionAreaMap::Iterator projIt;
            for (projIt = projMap.begin(); projIt != projMap.end(); ++projIt)
            {
                projectionListbox->insertItem(projIt.data().getName());
                projCombobox->insertItem(projIt.data().getName());
                projCombobox2->insertItem(projIt.data().getName());
                projectionListbox->setCurrentItem(0);
                numProjectors++;
            }

            hostMap = (*xmlIO.getHostMap());
            HostMap::Iterator hostIt;
            if (&hostMap != 0)
            {
                for (hostIt = hostMap.begin(); hostIt != hostMap.end(); ++hostIt)
                {
                    Host h = hostIt.data();
                    QListViewItem *hostItem = new QListViewItem(hostListView, 0);
                    hostItem->setText(0, hostIt.data().getName());
                    hostListView->setCurrentItem(hostItem);
                    hostListView->setSelected(hostItem, true);

                    if (h.isMasterHost())
                        masterHostSet = true;
                    else if (h.isControlHost())
                        controlHostSet = true;

                    PipeMap::Iterator pIt;
                    if (h.getPipeMap() != 0)
                    {
                        for (pIt = h.getPipeMap()->begin();
                             pIt != h.getPipeMap()->end(); ++pIt)
                        {
                            // get pipes
                            Pipe p = pIt.data();
                            QListViewItem *pipeItem = new QListViewItem(hostItem);
                            pipeItem->setOpen(true);
                            pipeItem->setText(0, pIt.key());
                            hostListView->setCurrentItem(pipeItem);
                            hostListView->setSelected(pipeItem, true);
                            WindowMap::Iterator winIt;
                            if (p.getWindowMap() != 0)
                            {
                                for (winIt = p.getWindowMap()->begin();
                                     winIt != p.getWindowMap()->end(); ++winIt)
                                {
                                    Window w = winIt.data();
                                    QListViewItem *winItem = new QListViewItem(pipeItem);
                                    pipeItem->setOpen(true);
                                    winItem->setText(0, w.getName());
                                    hostListView->setCurrentItem(winItem);
                                    hostListView->setSelected(winItem, true);
                                    ChannelMap::Iterator chIt;
                                    if (w.getChannelMap() != 0)
                                    {
                                        for (chIt = w.getChannelMap()->begin();
                                             chIt != w.getChannelMap()->end(); ++chIt)
                                        {
                                            Channel ch = chIt.data();
                                            QListViewItem *channelItem = new QListViewItem(winItem);
                                            winItem->setOpen(true);
                                            channelItem->setText(0, chIt.data().getName());
                                            hostListView->setCurrentItem(channelItem);
                                            hostListView->setSelected(channelItem, true);
                                            numChannels++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            cout << "numChannels: " << numChannels << endl;

            // general settings
            generalSettings = (*xmlIO.getGeneralSettings());
            coverConfigTool::initializeGeneralSettings();

            // tracking
            tracking = (*xmlIO.getTracking());
            coverConfigTool::initializeTracking();
        }
        else
        {
            cout << message << endl;
            xmlLoadStatus->setPaletteForegroundColor("red");
            xmlLoadStatus->setText(message);
        }
    }
}
