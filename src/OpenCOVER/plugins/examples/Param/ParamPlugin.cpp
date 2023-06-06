/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Param Plugin (does nothing)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#define USE_MATH_DEFINES
#include <math.h>
#include <QDir>
#include <config/coConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/PolygonMode>
#include "ParamPlugin.h"
#include <osg/LineWidth>

using namespace osg;

ParamPlugin::ParamPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool ParamPlugin::init()
{
    fprintf(stderr, "ParamPlugin::ParamPlugin\n");

    numColors = 0;

    readConfig();

    nheight = 70;
    numSegments = 4;
    radius = 20;
    topRadius = 1;
    segHeight = 3.7;
    twist = 0;
    currentMap = 3;
    squareParam = 0;
    cubicParam = 0;
    toolRadiusSquare = 400;

    paramTab = new coTUITab("Turm", coVRTui::instance()->mainFolder->getID());
    paramTab->setPos(0, 0);

    heightEdit = new coTUIEditFloatField("height", paramTab->getID());
    heightEdit->setEventListener(this);
    heightEdit->setPos(1, 0);
    heightEdit->setValue(segHeight);
    heightLabel = new coTUILabel("floorHeight:", paramTab->getID());
    heightLabel->setPos(0, 0);

    /* numHEdit = new coTUIEditFloatField("numHEdit",paramTab->getID());
   numHEdit->setEventListener(this);
   numHEdit->setPos(1,0);
   numHEdit->setValue(nheight);
   numHLabel = new coTUILabel("numH:",paramTab->getID());
   numHLabel->setPos(0,0); */

    numHSlider = new coTUISlider("numHSlider", paramTab->getID());
    numHSlider->setEventListener(this);
    numHSlider->setPos(1, 1);
    numHSlider->setMin(2);
    numHSlider->setMax(100);
    numHSlider->setValue(nheight);
    numHLabel = new coTUILabel("floors:", paramTab->getID());
    numHLabel->setPos(0, 1);

    /* radiusEdit = new coTUIEditFloatField("radius",paramTab->getID());
   radiusEdit->setEventListener(this);
   radiusEdit->setPos(1,2);
   radiusEdit->setValue(radius);
   radiusLabel = new coTUILabel("radius:",paramTab->getID());
   radiusLabel->setPos(0,2); */

    radiusSlider = new coTUIFloatSlider("bottom radius", paramTab->getID());
    radiusSlider->setEventListener(this);
    radiusSlider->setPos(1, 2);
    radiusSlider->setMin(1);
    radiusSlider->setMax(100);
    radiusSlider->setValue(radius);
    radiusLabel = new coTUILabel("bottom radius:", paramTab->getID());
    radiusLabel->setPos(0, 2);

    twistRadiusSlider = new coTUIFloatSlider("twist radius", paramTab->getID());
    twistRadiusSlider->setEventListener(this);
    twistRadiusSlider->setPos(7, 2);
    twistRadiusSlider->setMin(0);
    twistRadiusSlider->setMax(100);
    twistRadiusSlider->setValue(0);
    twistRadiusLabel = new coTUILabel("twist radius:", paramTab->getID());
    twistRadiusLabel->setPos(6, 2);

    topRadiusSlider = new coTUIFloatSlider("top radius", paramTab->getID());
    topRadiusSlider->setEventListener(this);
    topRadiusSlider->setPos(4, 2);
    topRadiusSlider->setMin(1);
    topRadiusSlider->setMax(100);
    topRadiusSlider->setValue(topRadius);
    topRadiusLabel = new coTUILabel("top radius:", paramTab->getID());
    topRadiusLabel->setPos(3, 2);

    /* numSegmentsEdit = new coTUIEditFloatField("numSegments",paramTab->getID());
   numSegmentsEdit->setEventListener(this);
   numSegmentsEdit->setPos(1,1);
   numSegmentsEdit->setValue(numSegments);
   numSegmentsLabel = new coTUILabel("numSegments:",paramTab->getID());
   numSegmentsLabel->setPos(0,1); */

    numSegmentsSlider = new coTUISlider("numSegments", paramTab->getID());
    numSegmentsSlider->setEventListener(this);
    numSegmentsSlider->setPos(1, 3);
    numSegmentsSlider->setMin(3);
    numSegmentsSlider->setMax(100);
    numSegmentsSlider->setValue(numSegments);
    numSegmentsLabel = new coTUILabel("numSegments:", paramTab->getID());
    numSegmentsLabel->setPos(0, 3);

    twistSlider = new coTUIFloatSlider("twistSlider", paramTab->getID());
    twistSlider->setEventListener(this);
    twistSlider->setPos(1, 4);
    twistSlider->setMin(-20);
    twistSlider->setMax(20);
    twistSlider->setValue(twist);
    twistLabel = new coTUILabel("twist:", paramTab->getID());
    twistLabel->setPos(0, 4);

    /* twistEdit = new coTUIEditFloatField("twist",paramTab->getID());
   twistEdit->setEventListener(this);
   twistEdit->setPos(1,4);
   twistEdit->setValue(twist);
   twistLabel = new coTUILabel("twist:",paramTab->getID());
   twistLabel->setPos(0,4); */

    squareEdit = new coTUIEditFloatField("square:", paramTab->getID());
    squareEdit->setEventListener(this);
    squareEdit->setPos(1, 5);
    squareEdit->setValue(squareParam);
    squareLabel = new coTUILabel("square:", paramTab->getID());
    squareLabel->setPos(0, 5);

    cubicEdit = new coTUIEditFloatField("cubic:", paramTab->getID());
    cubicEdit->setEventListener(this);
    cubicEdit->setPos(3, 5);
    cubicEdit->setValue(cubicParam);
    cubicLabel = new coTUILabel("cubic:", paramTab->getID());
    cubicLabel->setPos(2, 5);

    mapChoice = new coTUIComboBox("mapChoice", paramTab->getID());
    mapChoice->setEventListener(this);
    int i;
    for (i = 0; i < mapNames.count(); i++)
    {
        mapChoice->addEntry(mapNames[i].toStdString());
    }
    mapChoice->setSelectedEntry(currentMap);
    mapChoice->setPos(1, 6);
    mapLabel = new coTUILabel("map:", paramTab->getID());
    mapLabel->setPos(0, 6);

    deformButton = new coTUIToggleButton("deform", paramTab->getID());
    deformButton->setEventListener(this);
    deformButton->setPos(0, 7);
    deformButton->setState(false);

    toolChoice = new coTUIComboBox("mapChoice", paramTab->getID());
    toolChoice->setEventListener(this);
    toolChoice->addEntry("Cylinder");
    toolChoice->addEntry("Sphere");
    toolChoice->setSelectedEntry(0);
    toolChoice->setPos(1, 8);
    toolLabel = new coTUILabel("Tool:", paramTab->getID());
    toolLabel->setPos(0, 8);

    toolRadiusSlider = new coTUIFloatSlider("twistSlider", paramTab->getID());
    toolRadiusSlider->setEventListener(this);
    toolRadiusSlider->setPos(3, 8);
    toolRadiusSlider->setMin(1);
    toolRadiusSlider->setMax(100);
    toolRadiusSlider->setValue(20);
    toolRadiusLabel = new coTUILabel("radius:", paramTab->getID());
    toolRadiusLabel->setPos(2, 8);

    wireframeButton = new coTUIToggleButton("wireframe", paramTab->getID());
    wireframeButton->setEventListener(this);
    wireframeButton->setPos(0, 9);
    wireframeButton->setState(false);

    geom = new osg::Geometry();
    geode = new Geode();
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(false);
    geode->addDrawable(geom.get());
    vert = new Vec3Array;
    primitives = new DrawArrayLengths(PrimitiveSet::POLYGON);
    //indices = new UShortArray();
    //nindices = new UShortArray();
    normals = new Vec3Array;
    //cindices = new UShortArray();
    colors = new Vec4Array();
    cylinderTransform = new osg::MatrixTransform();
    sphereTransform = new osg::MatrixTransform();
    cylinder = new osg::Cylinder(osg::Vec3(0, 0, 0), 20, 40);
    cylinder->setHeight(radius * 2.2);
    sphere = new osg::Sphere(osg::Vec3(0, 0, 0), 20);
    cylinderGeode = new Geode();
    sphereGeode = new Geode();
    ShapeDrawable *sd = new osg::ShapeDrawable(cylinder.get());
    sd->setUseDisplayList(false);
    cylinderGeode->addDrawable(sd);
    sd = new osg::ShapeDrawable(sphere.get());
    sd->setUseDisplayList(false);
    sphereGeode->addDrawable(sd);
    cylinderTransform->addChild(cylinderGeode.get());
    sphereTransform->addChild(sphereGeode.get());
    currentTool = CYLINDER_TOOL;
    //cover->getObjectsRoot()->addChild(cylinderTransform.get());

    sphereTransform->setNodeMask(sphereTransform->getNodeMask() & ~Isect::Intersection);
    cylinderTransform->setNodeMask(cylinderTransform->getNodeMask() & ~Isect::Intersection);

    StateSet *geoState = sphereGeode->getOrCreateStateSet();
    Material *transpmtl = new Material;
    //transpmtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    transpmtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 0.4));
    transpmtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.7f, 0.7f, 0.7f, 0.4));
    transpmtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.7f, 0.7f, 0.7f, 0.4));
    transpmtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 0.4));
    transpmtl->setShininess(Material::FRONT_AND_BACK, 5.0f);
    transpmtl->setTransparency(Material::FRONT_AND_BACK, 0.4);

    geoState->setAttributeAndModes(transpmtl, StateAttribute::ON);
    geoState->setRenderingHint(StateSet::TRANSPARENT_BIN);
    geoState->setMode(GL_BLEND, StateAttribute::ON);
    geoState->setNestRenderBins(false);

    geoState = cylinderGeode->getOrCreateStateSet();
    geoState->setAttributeAndModes(transpmtl, StateAttribute::ON);
    geoState->setRenderingHint(StateSet::TRANSPARENT_BIN);
    geoState->setMode(GL_BLEND, StateAttribute::ON);
    geoState->setNestRenderBins(false);

    createGeom();
    geom->addPrimitiveSet(primitives.get());

    geoState = geode->getOrCreateStateSet();
    if (globalmtl.get() == NULL)
    {
        globalmtl = new Material;
        globalmtl->ref();
        globalmtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        globalmtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalmtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setShininess(Material::FRONT_AND_BACK, 10.0f);
    }

    geoState->setRenderingHint(StateSet::OPAQUE_BIN);
    geoState->setMode(GL_BLEND, StateAttribute::OFF);
    geoState->setAttributeAndModes(globalmtl.get(), StateAttribute::ON);
    cover->getObjectsRoot()->addChild(geode.get());

    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA, "DeformPush", coInteraction::Menu);
    interactionB = new coTrackerButtonInteraction(coInteraction::ButtonB, "DeformPull", coInteraction::Menu);

    return true;
}

// this is called if the plugin is removed at runtime
ParamPlugin::~ParamPlugin()
{
    fprintf(stderr, "ParamPlugin::~ParamPlugin\n");
    cover->getObjectsRoot()->removeChild(geode.get());
    cover->getObjectsRoot()->removeChild(cylinderTransform.get());
    cover->getObjectsRoot()->removeChild(sphereTransform.get());
    delete heightLabel;
    delete radiusLabel;
    delete topRadiusLabel;
    delete numSegmentsLabel;
    delete numHLabel;
    delete twistLabel;
    delete mapLabel;
    delete squareLabel;
    delete cubicLabel;
    delete toolLabel;
    delete toolRadiusLabel;
    delete deformButton;
    delete radiusSlider;
    delete topRadiusSlider;
    delete toolRadiusSlider;
    delete heightEdit;
    delete squareEdit;
    delete cubicEdit;
    delete numSegmentsSlider;
    delete numHSlider;
    delete twistSlider;
    delete mapChoice;
    delete toolChoice;
    delete wireframeButton;
    delete paramTab;
    delete interactionA;
    delete interactionB;
}

void ParamPlugin::tabletEvent(coTUIElement *tUIItem)
{

    if (tUIItem == numHSlider)
    {
        nheight = numHSlider->getValue();
        createGeom();
    }

    if (tUIItem == radiusSlider)
    {
        radius = radiusSlider->getValue();
        cylinder->setHeight(radius * 2.2);
        createGeom();
    }

    if (tUIItem == heightEdit)
    {
        segHeight = heightEdit->getValue();
        createGeom();
    }

    if (tUIItem == numSegmentsSlider)
    {
        numSegments = numSegmentsSlider->getValue();
        createGeom();
    }

    /* if(tUIItem == twistEdit)
   {
   twist = twistEdit->getValue();
   createGeom();
   } */

    if (tUIItem == twistSlider)
    {
        twist = twistSlider->getValue();
        createGeom();
    }

    if (tUIItem == twistRadiusSlider)
    {
        createGeom();
    }
    if (tUIItem == topRadiusSlider)
    {
        topRadius = topRadiusSlider->getValue();
        createGeom();
    }
    if (tUIItem == squareEdit)
    {
        squareParam = squareEdit->getValue();
        createGeom();
    }
    if (tUIItem == cubicEdit)
    {
        cubicParam = cubicEdit->getValue();
        createGeom();
    }
    if (tUIItem == mapChoice)
    {
        currentMap = mapChoice->getSelectedEntry();
        calcColors();
    }
    if (tUIItem == toolChoice)
    {

        if (currentTool == CYLINDER_TOOL)
            cover->getObjectsRoot()->removeChild(cylinderTransform.get());
        else
            cover->getObjectsRoot()->removeChild(sphereTransform.get());

        currentTool = toolChoice->getSelectedEntry();
        if (deformButton->getState())
        {
            if (currentTool == CYLINDER_TOOL)
                cover->getObjectsRoot()->addChild(cylinderTransform.get());
            else
                cover->getObjectsRoot()->addChild(sphereTransform.get());
        }
    }

    if (tUIItem == toolRadiusSlider)
    {
        toolRadiusSquare = toolRadiusSlider->getValue() * toolRadiusSlider->getValue();
        cylinder->setRadius(toolRadiusSlider->getValue());
        sphere->setRadius(toolRadiusSlider->getValue());
    }

    if (tUIItem == deformButton)
    {

        if (currentTool == CYLINDER_TOOL)
            cover->getObjectsRoot()->removeChild(cylinderTransform.get());
        else
            cover->getObjectsRoot()->removeChild(sphereTransform.get());

        if (deformButton->getState())
        {
            if (!interactionA->isRegistered())
            {
                coInteractionManager::the()->registerInteraction(interactionA);
                coInteractionManager::the()->registerInteraction(interactionB);
            }
            if (currentTool == CYLINDER_TOOL)
                cover->getObjectsRoot()->addChild(cylinderTransform.get());
            else
                cover->getObjectsRoot()->addChild(sphereTransform.get());
        }
        else
        {
            if (interactionA->isRegistered())
            {
                coInteractionManager::the()->unregisterInteraction(interactionA);
                coInteractionManager::the()->unregisterInteraction(interactionB);
            }
        }
    }

    if (tUIItem == wireframeButton)
    {
        StateSet *geoState = geode->getOrCreateStateSet();
        if (wireframeButton->getState())
        {
            osg::PolygonMode *polymode = new osg::PolygonMode;
            polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);

            osg::LineWidth *lineWidth = new osg::LineWidth(4);
            geoState->setAttributeAndModes(lineWidth, osg::StateAttribute::ON);
            geoState->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        }
        else
        {
            osg::PolygonMode *polymode = new osg::PolygonMode;
            geoState->setAttributeAndModes(polymode, osg::StateAttribute::ON);
        }
    }
}

void ParamPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

void ParamPlugin::deleteColorMap(const QString &name)
{
    float *mval = mapValues.value(name);
    mapSize.remove(name);
    mapValues.remove(name);
    delete[] mval;
}

//------------------------------------------------------------------------------
//
// read colormaps from xml config file
// read local colormaps
//------------------------------------------------------------------------------
void ParamPlugin::readConfig()
{
    covise::coConfig *config = covise::coConfig::getInstance();

    // read the name of all colormaps in file
    auto list = config->getVariableList("Colormaps").entries();

    for (const auto &entry : list)
        mapNames.append(entry.entry.c_str());

    // read the values for each colormap
    for (int k = 1; k < mapNames.size(); k++)
    {
        // get all definition points for the colormap
        QString cmapname = "Colormaps." + mapNames[k];
        auto variable = config->getVariableList(cmapname.toStdString()).entries();

        mapSize.insert(mapNames[k], variable.size());
        float *cval = new float[variable.size() * 5];
        mapValues.insert(mapNames[k], cval);

        // read the rgbax values
        int it = 0;
        for (int l = 0; l < variable.size() * 5; l = l + 5)
        {
            std::string tmp = cmapname.toStdString() + ".Point:" + std::to_string(it);
            cval[l] = config->getFloat("x", tmp, -1.0);
            if (cval[l] == -1)
            {
                cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
            }
            cval[l + 1] = config->getFloat("r", tmp, 1.0);
            cval[l + 2] = config->getFloat("g", tmp, 1.0);
            cval[l + 3] = config->getFloat("b", tmp, 1.0);
            cval[l + 4] = config->getFloat("a", tmp, 1.0);
            it++;
        }
    }

    // read values of local colormap files in .covise
    std::string place = covise::coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "colormaps";

    QDir directory(place.c_str());
    if (directory.exists())
    {
        QStringList filters;
        filters << "colormap_*.xml";
        directory.setNameFilters(filters);
        directory.setFilter(QDir::Files);
        QStringList files = directory.entryList();

        // loop over all found colormap xml files
        for (int j = 0; j < files.size(); j++)
        {
            covise::coConfigGroup *colorConfig = new covise::coConfigGroup("ColorMap");
            colorConfig->addConfig(place + "/" + files[j].toStdString(), "local", true);

            // read the name of the colormaps
            auto list = colorConfig->getVariableList("Colormaps").entries();

            // loop over all colormaps in one file
            for (const auto &e : list)
            {
                QString entry = e.entry.c_str();
                // remove global colormap with same name
                int index = mapNames.indexOf(entry);
                if (index != -1)
                {
                    mapNames.removeAt(index);
                    deleteColorMap(entry);
                }
                mapNames.append(entry);

                // get all definition points for the colormap
                QString cmapname = "Colormaps." + mapNames.last();
                auto variable = colorConfig->getVariableList(cmapname.toStdString()).entries();

                mapSize.insert(entry, variable.size());
                float *cval = new float[variable.size() * 5];
                mapValues.insert(entry, cval);

                // read the rgbax values
                int it = 0;
                for (int l = 0; l < variable.size() * 5; l = l + 5)
                {
                    std::string tmp = cmapname.toStdString() + ".Point:" + std::to_string(it);
                    cval[l] = std::stof(colorConfig->getValue("x", tmp, " -1.0").entry);
                    if (cval[l] == -1)
                    {
                        cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
                    }
                    cval[l + 1] = std::stof(colorConfig->getValue("r", tmp, "1.0").entry);
                    cval[l + 2] = std::stof(colorConfig->getValue("g", tmp, "1.0").entry);
                    cval[l + 3] = std::stof(colorConfig->getValue("b", tmp, "1.0").entry);
                    cval[l + 4] = std::stof(colorConfig->getValue("a", tmp, "1.0").entry);
                    it++;
                }
            }
            config->removeConfig(place + "/" + files[j].toStdString());
        }
    }
    mapNames.sort();
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Interpolate a cmap to a given number of steps
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Vec4 ParamPlugin::getColor(float pos)
{

    Vec4 actCol;
    int idx = 0;
    //cerr << "name: " << (const char *)mapNames[currentMap].toAscii() << endl;
    float *map = mapValues.value(mapNames[currentMap]);
    int mapS = mapSize.value(mapNames[currentMap]);
    if (map == NULL)
    {
        return actCol;
    }
    while (map[(idx + 1) * 5] <= pos)
    {
        idx++;
        if (idx > mapS - 2)
        {
            idx = mapS - 2;
            break;
        }
    }
    double d = (pos - map[idx * 5]) / (map[(idx + 1) * 5] - map[idx * 5]);
    actCol[0] = (float)((1 - d) * map[idx * 5 + 1] + d * map[(idx + 1) * 5 + 1]);
    actCol[1] = (float)((1 - d) * map[idx * 5 + 2] + d * map[(idx + 1) * 5 + 2]);
    actCol[2] = (float)((1 - d) * map[idx * 5 + 3] + d * map[(idx + 1) * 5 + 3]);
    actCol[3] = (float)((1 - d) * map[idx * 5 + 4] + d * map[(idx + 1) * 5 + 4]);

    return actCol;
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Interpolate a cmap to a given number of steps
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ParamPlugin::FlColor *ParamPlugin::interpolateColormap(FlColor *map, int numSteps)
{

    FlColor *actMap = new FlColor[numSteps];
    double delta = 1.0 / (numSteps - 1) * (numColors - 1);
    double x;
    int i;

    delta = 1.0 / (numSteps - 1);
    int idx = 0;
    for (i = 0; i < numSteps - 1; i++)
    {
        x = i * delta;
        while (map[idx + 1][4] <= x)
        {
            idx++;
            if (idx > numColors - 2)
            {
                idx = numColors - 2;
                break;
            }
        }
        double d = (x - map[idx][4]) / (map[idx + 1][0] - map[idx][0]);
        actMap[i][0] = (float)((1 - d) * map[idx][1] + d * map[idx + 1][1]);
        actMap[i][1] = (float)((1 - d) * map[idx][2] + d * map[idx + 1][2]);
        actMap[i][2] = (float)((1 - d) * map[idx][3] + d * map[idx + 1][3]);
        actMap[i][3] = (float)((1 - d) * map[idx][4] + d * map[idx + 1][4]);
        actMap[i][4] = -1;
    }
    actMap[numSteps - 1][0] = map[numColors - 1][0];
    actMap[numSteps - 1][1] = map[numColors - 1][1];
    actMap[numSteps - 1][2] = map[numColors - 1][2];
    actMap[numSteps - 1][3] = map[numColors - 1][3];
    actMap[numSteps - 1][4] = -1;

    return actMap;
}

void ParamPlugin::calcColors()
{
    float values[10000];
    //cindices->clear();
    //colors->clear();
    colors->clear();
    float minVal = 10000000, maxVal = -100000000;
    for (int h = 0; h < (nheight - 1); h++)
    {
        float area = 0;
        float len = 0;
        for (int s = 0; s < numSegments; s++)
        {

            Vec3 v1, v2, n;
            osg::Vec3Array *v = vert.get();
            if (s == numSegments - 1)
                v1 = (*v)[(h * numSegments + s) * 4] - (*v)[(h * numSegments) * 4];
            else
                v1 = (*v)[(h * numSegments + s) * 4] - (*v)[(h * numSegments + s) * 4 + 1];
            v2 = (*v)[(h * numSegments + s) * 4];
            v1[2] = 0;
            v2[2] = 0;
            n = v1 ^ v2;
            float a = n.length() / 2;
            float u = 2 * M_PI * (v1 - v2).length();
            len += u;
            area += a;
        }
        if (area > 0)
        {
            values[h] = len / area;
        }
        else
        {
            values[h] = 0;
        }
        if (values[h] < minVal)
            minVal = values[h];
        if (values[h] > maxVal)
            maxVal = values[h];
    }
    if (minVal == maxVal)
        maxVal = minVal + 0.000000000001;
    /*for(int h=0; h<nheight; h++)
   {
      //Vec4 c1(1.0,0,0,1.0), c2(0,1.0,0,1.0), c;
      //c = c1*((values[h]-minVal) / (maxVal - minVal) ) + c2*(1.0 -((values[h]-minVal) / (maxVal - minVal)));
      colors->push_back(getColor((values[h]-minVal) / (maxVal - minVal)));
   }*/
    Vec4 cc;
    Vec4 lc;
    for (int h = 0; h < nheight; h++)
    {
        cc = getColor((values[h] - minVal) / (maxVal - minVal));
        if (h == 0)
        {
            lc = cc;
        }
        for (int s = 0; s < numSegments; s++)
        {
            //if(h>0)
            //{
            colors->push_back(lc);
            colors->push_back(lc);
            colors->push_back(cc);
            colors->push_back(cc);
            //}
        }
        lc = cc;
    }
    geom->setColorArray(colors.get());
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
}

void ParamPlugin::createGeom()
{
    primitives->clear();
    normals->clear();
    vert->clear();
    int poly = 0;
    int hmax = nheight - 1;
    float funcmax = squareParam * hmax * hmax + cubicParam * hmax * hmax * hmax;
    float anglePerHeight = 0.1;
    float twistAngle = 0;
    twistRadius = this->twistRadiusSlider->getValue();
    float oldTwistAngle;
    for (int h = 0; h < nheight; h++)
    {
        double hfrag = (hmax - h) / (double)hmax; // 1 unten, 0 oben
        double r = (radius - topRadius) * hfrag + topRadius - squareParam * h * h - cubicParam * h * h * h + funcmax * (1 - hfrag);
        for (int s = 0; s < numSegments; s++)
        {
            if (h > 0)
            {
                double oldhfrag = (hmax - (h - 1)) / (double)hmax; // 1 unten, 0 oben
                double oldr = (radius - topRadius) * oldhfrag + topRadius - squareParam * (h - 1) * (h - 1) - cubicParam * (h - 1) * (h - 1) * (h - 1) + funcmax * (1 - oldhfrag);
                if (s > 0)
                {
                    float angle = 2 * M_PI / numSegments * s + twist / nheight * (h - 1);
                    vert->push_back(Vec3(sin(angle) * oldr + twistRadius * sin(oldTwistAngle), cos(angle) * oldr + twistRadius * cos(oldTwistAngle), (h - 1) * segHeight));
                    angle = 2 * M_PI / numSegments * (s - 1) + twist / nheight * (h - 1);
                    vert->push_back(Vec3(sin(angle) * oldr + twistRadius * sin(oldTwistAngle), cos(angle) * oldr + twistRadius * cos(oldTwistAngle), (h - 1) * segHeight));
                    angle = 2 * M_PI / numSegments * (s - 1) + twist / nheight * h;
                    vert->push_back(Vec3(sin(angle) * r + twistRadius * sin(twistAngle), cos(angle) * r + twistRadius * cos(twistAngle), h * segHeight));
                    angle = 2 * M_PI / numSegments * s + twist / nheight * h;
                    vert->push_back(Vec3(sin(angle) * r + twistRadius * sin(twistAngle), cos(angle) * r + twistRadius * cos(twistAngle), h * segHeight));
                }
                else
                {

                    float angle = 2 * M_PI / numSegments * s + twist / nheight * (h - 1);
                    vert->push_back(Vec3(sin(angle) * oldr + twistRadius * sin(oldTwistAngle), cos(angle) * oldr + twistRadius * cos(oldTwistAngle), (h - 1) * segHeight));
                    angle = 2 * M_PI / numSegments * (numSegments - 1) + twist / nheight * (h - 1);
                    vert->push_back(Vec3(sin(angle) * oldr + twistRadius * sin(oldTwistAngle), cos(angle) * oldr + twistRadius * cos(oldTwistAngle), (h - 1) * segHeight));
                    angle = 2 * M_PI / numSegments * (numSegments - 1) + twist / nheight * h;
                    vert->push_back(Vec3(sin(angle) * r + twistRadius * sin(twistAngle), cos(angle) * r + twistRadius * cos(twistAngle), h * segHeight));
                    angle = 2 * M_PI / numSegments * s + twist / nheight * h;
                    vert->push_back(Vec3(sin(angle) * r + twistRadius * sin(twistAngle), cos(angle) * r + twistRadius * cos(twistAngle), h * segHeight));
                }
                Vec3 v1, v2, n;
                osg::Vec3Array *v = vert.get();
                int ind = vert->size();
                v1 = (*v)[ind - 1] - (*v)[ind - 2];
                v2 = (*v)[ind - 3] - (*v)[ind - 1];
                v1.normalize();
                v2.normalize();
                n = v1 ^ v2;
                n.normalize();
                normals->push_back(n);
                normals->push_back(n);
                normals->push_back(n);
                normals->push_back(n);
                primitives->push_back(4);
                poly++;
            }
        }
        oldTwistAngle = twistAngle;
        twistAngle += anglePerHeight;
    }

    geom->setVertexArray(vert.get());
    geom->setNormalArray(normals.get());
    calcColors();
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->dirtyBound();
}

void
ParamPlugin::preFrame()
{
    Matrix pm = cover->getPointerMat() * cover->getInvBaseMat();
    Matrix cyl = pm;
    Matrix sphereMat = pm;

    Vec3 p = pm.getTrans();
    Vec3 pointerDir;
    Vec3 pointerDirHor;
    pointerDir[0] = pm(1, 0);
    pointerDir[1] = pm(1, 1);
    pointerDir[2] = pm(1, 2);
    pointerDirHor = pointerDir;
    pointerDirHor[2] = 0;
    pointerDirHor.normalize();
    pointerDir.normalize();
    osg::Vec3Array *v = vert.get();

    float dist = -((pointerDirHor * p) / (pointerDirHor * pointerDir));
    cyl.setTrans(p + (pointerDir * dist));
    cyl = Matrix::rotate(90, 1, 0, 0) * cyl;
    // get rid of scaling factor
    coCoord coord = cyl;
    coord.makeMat(cyl);
    cylinderTransform->setMatrix(cyl);
    if (coVRConfig::instance()->mouseTracking())
    {
        sphereMat.setTrans(p + (pointerDir * dist));
    }
    coord = sphereMat;
    coord.makeMat(sphereMat);
    sphereTransform->setMatrix(sphereMat);
    bool modified = false;
    if (interactionA->isRunning() || interactionB->isRunning())
    {
        int vnum = 0;
        for (int h = 0; h < nheight; h++)
        {
            //double hfrag = (hmax-h)/(double)hmax; // 1 unten, 0 oben
            //double r = (radius-topRadius)*hfrag + topRadius - squareParam *h*h - cubicParam *h*h*h + funcmax * (1-hfrag);
            for (int s = 0; s < numSegments; s++)
            {
                vnum = h * numSegments + s;
                //float angle = 2*M_PI/numSegments*s+twist/nheight*h;
                float dist;
                if (currentTool == CYLINDER_TOOL)
                {
                    dist = (((*v)[vnum] - p) ^ pointerDir).length2();
                }
                else
                {
                    dist = ((*v)[vnum] - sphereMat.getTrans()).length2();
                }
                if (dist < toolRadiusSquare)
                {
                    float diff = toolRadiusSquare - dist;
                    float x = (*v)[vnum][0];
                    float y = (*v)[vnum][1];
                    float len = sqrt(x * x + y * y);
                    if (len > 1.0)
                    {
                        if (interactionA->isRunning())
                        {
                            float mult = 1 - ((diff / 20.0) / len);
                            x *= mult;
                            y *= mult;
                            if (mult <= 0 || sqrt(x * x + y * y) < 1.0)
                            {
                                (*v)[vnum][0] /= len;
                                (*v)[vnum][1] /= len;
                            }
                            else
                            {
                                (*v)[vnum][0] = x;
                                (*v)[vnum][1] = y;
                            }
                        }
                        else
                        {
                            float mult = 1 - ((diff / 20.0) / len);
                            for (int seg = 0; seg < numSegments; seg++)
                            {
                                vnum = h * numSegments + seg;
                                float x = (*v)[vnum][0];
                                float y = (*v)[vnum][1];
                                float len = sqrt(x * x + y * y);
                                x *= mult;
                                y *= mult;
                                if (mult <= 0 || sqrt(x * x + y * y) < 1.0)
                                {
                                    (*v)[vnum][0] /= len;
                                    (*v)[vnum][1] /= len;
                                }
                                else
                                {
                                    (*v)[vnum][0] = x;
                                    (*v)[vnum][1] = y;
                                }
                            }
                            s = numSegments; // continue on next floor
                        }
                    }
                    modified = true;
                }
            }
        }
    }
    if (modified)
        calcColors();
}

COVERPLUGIN(ParamPlugin)
