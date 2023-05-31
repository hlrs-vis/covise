/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Messwerte Plugin (does nothing)                              **
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
#include <device/VRTracker.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/PolygonMode>
#include "MesswertePlugin.h"
#include <osg/LineWidth>
using namespace osg;

MesswertePlugin::MesswertePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool MesswertePlugin::init()
{
    fprintf(stderr, "MesswertePlugin::MesswertePlugin\n");

    numColors = 0;

    numVertTotal = 126;
    numFaces = 4;
    numVertices = new int[numFaces];
    numVertices[0] = 48;
    numVertices[1] = 48;
    numVertices[2] = 12;
    numVertices[3] = 18;

    minVal = -1;
    maxVal = 0.3;

    floatValues = new float[numVertTotal];
    readConfig();

    currentMap = 3;

    MesswerteTab = new coTUITab("Messwerte", coVRTui::instance()->mainFolder->getID());
    MesswerteTab->setPos(0, 0);

    wireframeButton = new coTUIToggleButton("wireframe", MesswerteTab->getID());
    wireframeButton->setEventListener(this);
    wireframeButton->setPos(0, 9);
    wireframeButton->setState(false);

    discreteColorsButton = new coTUIToggleButton("discreteColors", MesswerteTab->getID());
    discreteColorsButton->setEventListener(this);
    discreteColorsButton->setPos(0, 8);
    discreteColorsButton->setState(false);

    mapChoice = new coTUIComboBox("mapChoice", MesswerteTab->getID());
    mapChoice->setEventListener(this);
    int i;
    for (i = 0; i < mapNames.count(); i++)
    {
        mapChoice->addEntry(mapNames[i].toAscii());
    }
    mapChoice->setSelectedEntry(currentMap);
    mapChoice->setPos(1, 6);
    mapLabel = new coTUILabel("map:", MesswerteTab->getID());
    mapLabel->setPos(0, 6);

    minEdit = new coTUIEditFloatField("min:", MesswerteTab->getID());
    minEdit->setEventListener(this);
    minEdit->setPos(1, 5);
    minEdit->setValue(minVal);
    minLabel = new coTUILabel("min:", MesswerteTab->getID());
    minLabel->setPos(0, 5);

    maxEdit = new coTUIEditFloatField("max:", MesswerteTab->getID());
    maxEdit->setEventListener(this);
    maxEdit->setPos(3, 5);
    maxEdit->setValue(maxVal);
    maxLabel = new coTUILabel("max:", MesswerteTab->getID());
    maxLabel->setPos(2, 5);

    numStepsEdit = new coTUIEditIntField("numSteps:", MesswerteTab->getID());
    numStepsEdit->setEventListener(this);
    numStepsEdit->setPos(5, 5);
    numStepsEdit->setValue(20);
    numStepsLabel = new coTUILabel("numSteps:", MesswerteTab->getID());
    numStepsLabel->setPos(4, 5);

    vert = new Vec3Array;
    primitives = new DrawArrayLengths(PrimitiveSet::POLYGON);
    indices = new UShortArray();
    nindices = new UShortArray();
    normals = new Vec3Array;
    colors = new Vec4Array();
    texCoords = new Vec2Array();

    tex = new osg::Texture2D;
    tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
    tex->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    tex->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
    texImage = new osg::Image;

    generateTexture();

    geom = new osg::Geometry();
    geode = new Geode();
    geode->setName("Messwerte");
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(false);
    geode->addDrawable(geom.get());

    geomTex = new osg::Geometry();
    geodeTex = new Geode();
    geodeTex->setName("MesswerteTexturiert");
    geomTex->setUseDisplayList(false);
    geomTex->setUseVertexBufferObjects(false);
    geodeTex->addDrawable(geomTex.get());

    createGeom();

    geom->setVertexArray(vert.get());
    geom->setVertexIndices(indices.get());
    geom->setNormalIndices(nindices.get());
    geom->setNormalArray(normals.get());
    geom->setColorIndices(indices.get());
    geom->setColorArray(colors.get());
    geom->addPrimitiveSet(primitives.get());

    geomTex->setVertexArray(vert.get());
    geomTex->setVertexIndices(indices.get());
    geomTex->setNormalIndices(nindices.get());
    geomTex->setNormalArray(normals.get());
    geomTex->setTexCoordIndices(0, indices.get());
    geomTex->setTexCoordArray(0, texCoords.get());
    geomTex->addPrimitiveSet(primitives.get());

    osg::StateSet *geoState = geode->getOrCreateStateSet();
    if (globalmtl.get() == NULL)
    {
        globalmtl = new Material;
        globalmtl->ref();
        globalmtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        globalmtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalmtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
        globalmtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setShininess(Material::FRONT_AND_BACK, 10.0f);
    }

    geoState->setRenderingHint(StateSet::OPAQUE_BIN);
    geoState->setMode(GL_BLEND, StateAttribute::OFF);
    geoState->setAttributeAndModes(globalmtl.get(), StateAttribute::ON);

    geoState = geodeTex->getOrCreateStateSet();
    geoState->setRenderingHint(StateSet::TRANSPARENT_BIN);
    geoState->setMode(GL_BLEND, StateAttribute::ON);
    geoState->setAttributeAndModes(globalmtl.get(), StateAttribute::ON);
    geoState->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
    osg::TexEnv *texEnv = new osg::TexEnv;
    texEnv->setMode(osg::TexEnv::MODULATE);
    geoState->setTextureAttributeAndModes(0, texEnv, osg::StateAttribute::ON);

    cover->getObjectsRoot()->addChild(geode.get());

    int numColors = numStepsEdit->getValue();
    float *r, *g, *b, *a;
    r = new float[numColors];
    g = new float[numColors];
    b = new float[numColors];
    a = new float[numColors];
    for (int i = 0; i < numColors; i++)
    {
        osg::Vec4 c = getColor((float)i / (float)(numColors - 1));
        r[i] = c[0];
        g[i] = c[1];
        b[i] = c[2];
        a[i] = c[3];
    }

    cBar = new ColorBar("Pessure", "Pascal", minVal, maxVal, numColors, r, g, b, a);
    cBar->setVisible(true);
    delete[] r;
    delete[] g;
    delete[] b;
    delete[] a;

    conn = NULL;

    serverHost = NULL;

    port = coCoviseConfig::getInt("port", "COVER.Plugin.Messwerte", 6340);
    std::string line = coCoviseConfig::getEntry("host", "COVER.Plugin.Messwerte", "192.168.128.127");
    //std::string line = coCoviseConfig::getEntry("host","COVER.Plugin.Messwerte","localhost");
    if (!line.empty())
    {
        if (strcasecmp(line.c_str(), "NONE") == 0)
            serverHost = NULL;
        else
            serverHost = new Host(line.c_str());
    }
    else
    {
        serverHost = new Host("localhost");
    }

    return true;
}

// this is called if the plugin is removed at runtime
MesswertePlugin::~MesswertePlugin()
{
    fprintf(stderr, "MesswertePlugin::~MesswertePlugin\n");
    cover->getObjectsRoot()->removeChild(geode.get());
    delete mapChoice;
    delete wireframeButton;
    delete discreteColorsButton;
    delete MesswerteTab;
    delete numVertices;
    delete[] floatValues;
    delete cBar;
}

void MesswertePlugin::tabletEvent(coTUIElement *tUIItem)
{

    if (tUIItem == mapChoice)
    {
        currentMap = mapChoice->getSelectedEntry();
        generateTexture();
        calcColors();
        updateColormap();
    }
    if (tUIItem == minEdit)
    {
        minVal = minEdit->getValue();
        calcColors();
        updateColormap();
    }
    if (tUIItem == maxEdit)
    {
        maxVal = maxEdit->getValue();
        calcColors();
        updateColormap();
    }
    if (tUIItem == numStepsEdit)
    {
        generateTexture();
        updateColormap();
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

    if (tUIItem == discreteColorsButton)
    {
        if (discreteColorsButton->getState())
        {

            cover->getObjectsRoot()->removeChild(geode.get());
            cover->getObjectsRoot()->addChild(geodeTex.get());
        }
        else
        {

            cover->getObjectsRoot()->removeChild(geodeTex.get());
            cover->getObjectsRoot()->addChild(geode.get());
        }
    }
}

void MesswertePlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

void MesswertePlugin::updateColormap()
{
    int numColors = numStepsEdit->getValue();
    float *r, *g, *b, *a;
    r = new float[numColors];
    g = new float[numColors];
    b = new float[numColors];
    a = new float[numColors];
    for (int i = 0; i < numColors; i++)
    {
        osg::Vec4 c = getColor((float)i / (float)(numColors - 1));
        r[i] = c[0];
        g[i] = c[1];
        b[i] = c[2];
        a[i] = c[3];
    }

    cBar->update("Pascal", minVal, maxVal, numColors, r, g, b, a);
    delete[] r;
    delete[] g;
    delete[] b;
    delete[] a;
}

void MesswertePlugin::deleteColorMap(const QString &name)
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
void MesswertePlugin::readConfig()
{
    coConfig *config = coConfig::getInstance();

    // read the name of all colormaps in file
    QStringList list;
    list = config->getVariableList("Colormaps");

    for (int i = 0; i < list.size(); i++)
        mapNames.append(list[i]);

    // read the values for each colormap
    for (int k = 1; k < mapNames.size(); k++)
    {
        // get all definition points for the colormap
        QString cmapname = "Colormaps." + mapNames[k];
        QStringList variable = config->getVariableList(cmapname);

        mapSize.insert(mapNames[k], variable.size());
        float *cval = new float[variable.size() * 5];
        mapValues.insert(mapNames[k], cval);

        // read the rgbax values
        int it = 0;
        for (int l = 0; l < variable.size() * 5; l = l + 5)
        {
            QString tmp = cmapname + ".Point:" + QString::number(it);
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
    QString place = coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "colormaps";

    QDir directory(place);
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
            coConfigGroup *colorConfig = new coConfigGroup("ColorMap");
            colorConfig->addConfig(place + "/" + files[j], "local", true);

            // read the name of the colormaps
            QStringList list;
            list = colorConfig->getVariableList("Colormaps");

            // loop over all colormaps in one file
            for (int i = 0; i < list.size(); i++)
            {

                // remove global colormap with same name
                int index = mapNames.indexOf(list[i]);
                if (index != -1)
                {
                    mapNames.removeAt(index);
                    deleteColorMap(list[i]);
                }
                mapNames.append(list[i]);

                // get all definition points for the colormap
                QString cmapname = "Colormaps." + mapNames.last();
                QStringList variable = colorConfig->getVariableList(cmapname);

                mapSize.insert(list[i], variable.size());
                float *cval = new float[variable.size() * 5];
                mapValues.insert(list[i], cval);

                // read the rgbax values
                int it = 0;
                for (int l = 0; l < variable.size() * 5; l = l + 5)
                {
                    QString tmp = cmapname + ".Point:" + QString::number(it);
                    cval[l] = (colorConfig->getValue("x", tmp, " -1.0")).toFloat();
                    if (cval[l] == -1)
                    {
                        cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
                    }
                    cval[l + 1] = (colorConfig->getValue("r", tmp, "1.0")).toFloat();
                    cval[l + 2] = (colorConfig->getValue("g", tmp, "1.0")).toFloat();
                    cval[l + 3] = (colorConfig->getValue("b", tmp, "1.0")).toFloat();
                    cval[l + 4] = (colorConfig->getValue("a", tmp, "1.0")).toFloat();
                    it++;
                }
            }
            config->removeConfig(place + "/" + files[j]);
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

Vec4 MesswertePlugin::getColor(float pos)
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

void MesswertePlugin::generateTexture()
{

    int numSteps = numStepsEdit->getValue();
    unsigned char *it = NULL;
    it = new unsigned char[256 * 4];
    int currentPos = 0;
    for (int i = 0; i < numSteps; i++)
    {
        int toPos = ((float)i / ((float)(numSteps - 1))) * 255;
        osg::Vec4 currentColor = getColor((float)i / (float)(numSteps - 1));
        while (currentPos <= toPos)
        {
            it[currentPos * 4] = currentColor[0] * 255;
            it[currentPos * 4 + 1] = currentColor[1] * 255;
            it[currentPos * 4 + 2] = currentColor[2] * 255;
            it[currentPos * 4 + 3] = currentColor[3] * 255;
            currentPos++;
        }
    }

    texImage->setImage(256, 1, 1, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, it, osg::Image::USE_NEW_DELETE);
    tex->setImage(texImage.get());
}

void MesswertePlugin::calcColors()
{
    colors->clear();
    texCoords->clear();
    float offset;
    long long dsec = cover->frameTime() * 10.0;
    offset = (dsec % 10) / 3.0;
    if (conn == NULL)
    {
        int n = 0;
        for (int i = 0; i < numVertTotal; i++)
        {
            floatValues[n] = sin((float)i / (float)50 * 2 * M_PI + offset) + sin((float)i / (float)50 * 2 * M_PI);
            n++;
        }
    }

    int numValues = numVertTotal;
    if (discreteColorsButton->getState())
    {
        for (int i = 0; i < numValues; i++)
        {
            texCoords->push_back(osg::Vec2((floatValues[i] - minVal) / (maxVal - minVal), (floatValues[i] - minVal) / (maxVal - minVal)));
        }
    }
    else
    {
        for (int i = 0; i < numValues; i++)
        {
            colors->push_back(getColor((floatValues[i] - minVal) / (maxVal - minVal)));
        }
    }
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
}

void MesswertePlugin::readCoords(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp != NULL)
    {
        char buf[1000];
        for (int i = 0; i < 9; i++)
            fgets(buf, 1000, fp); // header
        while (!feof(fp))
        {
            float xc, yc, zc;
            fgets(buf, 1000, fp);
            if (!feof(fp))
            {
                if (sscanf(buf, "%f %f %f", &xc, &yc, &zc) == 3)
                    vert->push_back(osg::Vec3(xc, yc, zc));
            }
        }
        fclose(fp);
    }
}

void MesswertePlugin::readVertices(const char *filename, int offset)
{
    FILE *fp = fopen(filename, "r");
    if (fp != NULL)
    {
        char buf[1000];

        while (!feof(fp))
        {
            int xc, yc, zc;
            fgets(buf, 1000, fp);
            if (!feof(fp))
            {
                if (sscanf(buf, "%d %d %d", &xc, &yc, &zc) == 3)
                {

                    indices->push_back(xc + offset);
                    indices->push_back(yc + offset);
                    indices->push_back(zc + offset);
                    ind += 3;

                    Vec3 v1, v2, n;
                    osg::Vec3Array *v = vert.get();
                    v1 = (*v)[xc + offset] - (*v)[yc + offset];
                    v2 = (*v)[xc + offset] - (*v)[zc + offset];
                    v1.normalize();
                    v2.normalize();
                    n = v1 ^ v2;
                    n.normalize();
                    normals->push_back(n);
                    nindices->push_back(poly);
                    primitives->push_back(3);
                    poly++;
                }
            }
        }
        fclose(fp);
    }
}

void MesswertePlugin::createGeom()
{
    primitives->clear();
    indices->clear();
    nindices->clear();
    normals->clear();
    vert->clear();
    geom->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE);

    readCoords("c:\\data\\Messwerte\\Stufenheck_Seite_links.dat");
    readCoords("c:\\data\\Messwerte\\Stufenheck_Seite_rechts.dat");
    readCoords("c:\\data\\Messwerte\\Stufenheck_Heck.dat");
    readCoords("c:\\data\\Messwerte\\Stufenheck_Heckschraege.dat");

    poly = 0;
    ind = 0;
    vertNum = 0;
    // int startVertNum=97;
    readVertices("c:\\data\\Messwerte\\Stufenheck_Seite_links.vert", -1);
    readVertices("c:\\data\\Messwerte\\Stufenheck_Seite_links.vert", 47);
    readVertices("c:\\data\\Messwerte\\Stufenheck_Heck.vert", -1);
    readVertices("c:\\data\\Messwerte\\Stufenheck_Heckschraege.vert", -1);
    /* for(int i=2; i<numFaces; i++)
   {
      for(int j=0; j<numVertices[i]-2; j++)
      {
         indices->push_back(vertNum); 
         indices->push_back(vertNum+1); 
         indices->push_back(vertNum+2); 
         ind +=3;

         Vec3 v1,v2,n;
         osg::Vec3Array *v = vert.get();
         osg::UShortArray *in = indices.get();
         v1 = (*v)[(*in)[vertNum]] - (*v)[(*in)[vertNum+1]];
         v2 = (*v)[(*in)[vertNum+2]] - (*v)[(*in)[vertNum]];
         v1.normalize();
         v2.normalize();
         n=v1 ^ v2;
         n.normalize();
         normals->push_back(n);
         nindices->push_back(poly);
         primitives->push_back(3);
         poly++;
         
         vertNum++;
      }
      startVertNum+=numVertices[i];
      vertNum = startVertNum;
   }*/
    calcColors();
    geom->dirtyBound();
}

bool MesswertePlugin::readVal(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    while (numRead < numBytes)
    {
        readBytes = conn->getSocket()->Read(((unsigned char *)buf) + readBytes, toRead);
        if (readBytes <= 0)
        {
            cerr << "error reading data from socket" << endl;
            return false;
        }
        cerr << "read " << readBytes << endl;
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}

void
MesswertePlugin::preFrame()
{
    bool readData = false;
    if (conn)
    {
        while (conn && conn->check_for_input())
        {
            if (!readVal(floatValues, numVertTotal * sizeof(float)))
            {
                delete conn;
                conn = NULL;
            }
            else
            {

                byteSwap(floatValues, numVertTotal);
                readData = true;
            }
        }
    }
    else if ((coVRMSController::instance()->isMaster()) && (serverHost != NULL))
    {
        // try to connect to server every 10 secnods
        if ((cover->frameTime() - oldTime) > 10)
        {
            conn = new SimpleClientConnection(serverHost, port, 0);

            if (!conn->is_connected()) // could not open server port
            {
#ifndef _WIN32
                if (errno != ECONNREFUSED)
                {
                    fprintf(stderr, "Could not connect to LabView on %s; port %d\n", serverHost->getName(), port);
                    delete serverHost;
                    serverHost = NULL;
                }
#endif
                delete conn;
                conn = NULL;
            }
            else
            {
                fprintf(stderr, "Connected to Server on %s; port %d\n", serverHost->getName(), port);
            }
            if (conn && conn->is_connected())
            {
                //int id=2;
                //conn->getSocket()->write(&id,sizeof(id));
            }
            oldTime = cover->frameTime();
        }
    }
    if (coVRMSController::instance()->isMaster())
    {
        int numFloats = 0;
        if (readData)
            numFloats = numVertTotal;
        coVRMSController::instance()->sendSlaves((char *)&numFloats, sizeof(int));
        if (numFloats > 0)
        {
            coVRMSController::instance()->sendSlaves((char *)&floatValues, numFloats * sizeof(float));
            calcColors();
        }
    }
    else
    {
        int numFloats = 0;
        if (readData)
            numFloats = numVertTotal;
        coVRMSController::instance()->readMaster((char *)&numFloats, sizeof(int));
        if (numFloats > 0)
        {
            coVRMSController::instance()->readMaster((char *)&floatValues, numFloats * sizeof(float));
            calcColors();
        }
    }

    //if(modified)

    if (conn == NULL)
    {
        calcColors();
    }
}

COVERPLUGIN(MesswertePlugin)
