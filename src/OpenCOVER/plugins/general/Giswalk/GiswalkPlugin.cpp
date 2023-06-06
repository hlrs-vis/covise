/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\
**                                                            (C)2005 HLRS  **
**                                                                          **
** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** April-05  v1	    				       		                         **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "GiswalkPlugin.h"
#define USE_MATH_DEFINES
#include <math.h>
#include <QDir>
#include <config/coConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/LineWidth>

#include <osg/LineSegment>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osgUtil/IntersectVisitor>
#include <cover/coVRAnimationManager.h>
#define MAXSAMPLES 1200
using namespace osg;
using namespace osgUtil;
/************************************************************************

       Copyright 2008 Mark Pictor

   This file is part of RS274NGC.

   RS274NGC is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   RS274NGC is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with RS274NGC.  If not, see <http://www.gnu.org/licenses/>.

   This software is based on software that was produced by the National
   Institute of Standards and Technology (NIST).

   ************************************************************************/

#include <stdio.h> /* gets, etc. */
#include <stdlib.h> /* exit       */
#include <string.h> /* strcpy     */

GiswalkPlugin *GiswalkPlugin::thePlugin = NULL;

GiswalkPlugin *GiswalkPlugin::instance()
{
    if (!thePlugin)
        thePlugin = new GiswalkPlugin();
    return thePlugin;
}

GiswalkPlugin::GiswalkPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //positions=NULL;
    thePlugin = this;
}

static FileHandler handlers[] = {
    { NULL,
      GiswalkPlugin::sloadGCode,
      GiswalkPlugin::unloadGCode,
      "smap" }
};

int GiswalkPlugin::sloadGCode(const char *filename, osg::Group *loadParent)
{

    instance()->loadGCode(filename, loadParent);
    return 0;
}

int GiswalkPlugin::loadGCode(const char *filename, osg::Group *loadParent)
{

    frameNumber = 0;
    /*   //delete[] positions;
   //positions = new float [3*MAXSAMPLES+3];
   
	geode = new Geode();
	geom = new Geometry();
	geode->setStateSet(geoState.get());

	geom->setColorBinding(Geometry::BIND_OFF);

	geode->addDrawable(geom.get());
	geode->setName("Viewer Positions");
	
	// set up geometry
	vert = new osg::Vec3Array;
	color = new osg::Vec4Array;

	primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
	primitives->push_back(vert->size());
	
	// Update animation frame:
	coVRAnimationManager::instance()->setNumTimesteps(vert->size(), this);

	geom->setVertexArray(vert);
	geom->setColorArray(color);
	geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	geom->addPrimitiveSet(primitives);
	geom->dirtyDisplayList();
	geom->setUseDisplayList(false);*/
    parentNode = loadParent;
    if (parentNode == NULL)
        parentNode = cover->getObjectsRoot();
    parentNode->addChild(geode.get());
    ;
    return 0;
}
//--------------------------------------------------------------------
void GiswalkPlugin::setTimestep(int t)
{
    if (primitives)
        primitives->at(0) = t;
}

int GiswalkPlugin::unloadGCode(const char *filename)
{
    (void)filename;

    return 0;
}

void GiswalkPlugin::deleteColorMap(const QString &name)
{
    float *mval = mapValues.value(name);
    mapSize.remove(name);
    mapValues.remove(name);
    delete[] mval;
}

bool GiswalkPlugin::init()
{
    fprintf(stderr, "GiswalkPlugin::GiswalkPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);

    length = 1;
    recordRate = 1;
    filename = NULL;
    primitives = NULL;
    currentMap = 0;

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

    PathTab = new coTUITab("Giswalk", coVRTui::instance()->mainFolder->getID());
    record = new coTUIToggleButton("Record", PathTab->getID());
    stop = new coTUIButton("Stop", PathTab->getID());
    play = new coTUIButton("Play", PathTab->getID());
    reset = new coTUIButton("Reset", PathTab->getID());
    saveButton = new coTUIButton("Save", PathTab->getID());

    mapChoice = new coTUIComboBox("mapChoice", PathTab->getID());
    mapChoice->setEventListener(this);
    int i;
    for (i = 0; i < mapNames.count(); i++)
    {
        mapChoice->addEntry(mapNames[i].toStdString());
    }
    mapChoice->setSelectedEntry(currentMap);
    mapChoice->setPos(6, 0);

    viewPath = new coTUIToggleButton("View Path", PathTab->getID());
    viewDirections = new coTUIToggleButton("Viewing Directions", PathTab->getID());
    viewlookAt = new coTUIToggleButton("View Target", PathTab->getID());

    lengthLabel = new coTUILabel("Length", PathTab->getID());
    lengthLabel->setPos(0, 4);
    lengthEdit = new coTUIEditFloatField("length", PathTab->getID());
    lengthEdit->setValue(1);
    lengthEdit->setPos(1, 4);

    radiusLabel = new coTUILabel("Radius", PathTab->getID());
    radiusLabel->setPos(2, 4);
    radiusEdit = new coTUIEditFloatField("radius", PathTab->getID());
    radiusEdit->setValue(1);
    radiusEdit->setEventListener(this);
    radiusEdit->setPos(3, 4);
    renderMethod = new coTUIComboBox("renderMethod", PathTab->getID());
    renderMethod->addEntry("renderMethod CPU Billboard");
    renderMethod->addEntry("renderMethod Cg Shader");
    renderMethod->addEntry("renderMethod Point Sprite");
    renderMethod->setSelectedEntry(0);
    renderMethod->setEventListener(this);
    renderMethod->setPos(0, 5);

    recordRateLabel = new coTUILabel("recordRate", PathTab->getID());
    recordRateLabel->setPos(0, 3);
    recordRateTUI = new coTUIEditIntField("Fps", PathTab->getID());
    recordRateTUI->setEventListener(this);
    recordRateTUI->setValue(1);
    //recordRateTUI->setText("Fps:");
    recordRateTUI->setPos(1, 3);

    fileNameBrowser = new coTUIFileBrowserButton("File", PathTab->getID());
    fileNameBrowser->setMode(coTUIFileBrowserButton::SAVE);
    fileNameBrowser->setFilterList("*.txt");
    fileNameBrowser->setPos(0, 7);
    fileNameBrowser->setEventListener(this);

    numSamples = new coTUILabel("SampleNum: 0", PathTab->getID());
    numSamples->setPos(0, 6);
    PathTab->setPos(0, 0);
    record->setPos(0, 0);
    record->setEventListener(this);
    stop->setPos(1, 0);
    stop->setEventListener(this);
    play->setPos(2, 0);
    play->setEventListener(this);
    reset->setPos(3, 0);
    reset->setEventListener(this);
    saveButton->setPos(4, 0);
    saveButton->setEventListener(this);
    //positions = new float [3*MAXSAMPLES+3];
    lookat[0] = new float[MAXSAMPLES + 1];
    lookat[1] = new float[MAXSAMPLES + 1];
    lookat[2] = new float[MAXSAMPLES + 1];
    objectName = new const char *[MAXSAMPLES + 3];
    viewPath->setPos(0, 2);
    viewPath->setEventListener(this);
    viewlookAt->setPos(1, 2);
    viewlookAt->setEventListener(this);
    viewDirections->setPos(2, 2);
    viewDirections->setEventListener(this);
    frameNumber = 0;
    record->setState(false);
    playing = false;

    geoState = new osg::StateSet();
    linemtl = new osg::Material;
    lineWidth = new osg::LineWidth(2.0);
    linemtl.get()->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    linemtl.get()->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    linemtl.get()->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0));
    linemtl.get()->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
    linemtl.get()->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    linemtl.get()->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    geoState->setAttributeAndModes(linemtl.get(), StateAttribute::ON);

    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    geoState->setAttributeAndModes(lineWidth.get(), StateAttribute::ON);

    return true;
}

// this is called if the plugin is removed at runtime
GiswalkPlugin::~GiswalkPlugin()
{
    fprintf(stderr, "GiswalkPlugin::~GiswalkPlugin\n");

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);

    delete record;
    delete stop;
    delete play;
    delete reset;
    delete saveButton;
    delete viewPath;
    delete viewDirections;
    delete viewlookAt;
    delete lengthLabel;
    delete lengthEdit;
    delete radiusLabel;
    delete radiusEdit;
    delete renderMethod;
    delete recordRateLabel;
    delete recordRateTUI;
    delete numSamples;
    delete PathTab;
    delete[] filename;

    //delete[] positions;
    delete[] lookat[0];
    delete[] lookat[1];
    delete[] lookat[2];
    delete[] objectName;
    if (geode->getNumParents() > 0)
    {
        parentNode = geode->getParent(0);
        if (parentNode)
            parentNode->removeChild(geode.get());
    }
}

void
GiswalkPlugin::preFrame()
{
    if (record->getState())
    {
    }
}

void GiswalkPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == lengthEdit)
    {
        length = lengthEdit->getValue();
    }
    else if (tUIItem == recordRateTUI)
    {
        recordRate = 1.0 / recordRateTUI->getValue();
    }
    else if (tUIItem == fileNameBrowser)
    {
        std::string fn = fileNameBrowser->getSelectedPath();
        delete filename;
        filename = new char[fn.length()];
        strcpy(filename, fn.c_str());

        if (filename[0] != '\0')
        {
            char *pchar;
            if ((pchar = strstr(filename, "://")) != NULL)
            {
                pchar += 3;
                strcpy(filename, pchar);
            }
        }
    }
}
void GiswalkPlugin::save()
{
    /*FILE * fp = fopen(filename,"w");
   if(fp)
   {
      fprintf(fp,"# x,      y,      z,      dx,      dy,     dz\n");
      fprintf(fp,"# numFrames: %d\n",frameNumber);
      for(int n=0;n<frameNumber;n++)
      {
          fprintf(fp,"%010.3f,%010.3f,%010.3f,%010.3f,%010.3f,%010.3f\n",positions[n*3  ],positions[n*3+1], positions[n*3+2],lookat[0][n],lookat[1][n],lookat[2][n]);
      }
      fclose(fp);
   }
   else
   {
      cerr << "could not open file " << filename << endl;
   }*/
}

void GiswalkPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == play)
    {
        playing = true;
    }

    if (tUIItem == saveButton)
    {
        save();
    }

    else if (tUIItem == record)
    {
        playing = false;
    }
    else if (tUIItem == stop)
    {
        record->setState(false);
        playing = false;
    }
    else if (tUIItem == reset)
    {
        frameNumber = 0;
        record->setState(false);
        playing = false;
    }
    else if (tUIItem == lengthEdit)
    {
        length = lengthEdit->getValue();
    }
    else if (tUIItem == viewPath)
    {
        char label[100];
        sprintf(label, "numSamples: %d", frameNumber);
        numSamples->setLabel(label);
        if (viewPath->getState())
        {

            if (parentNode == NULL)
                parentNode = cover->getObjectsRoot();
            parentNode->addChild(geode.get());
            ;
        }
        else
        {
            parentNode = geode->getParent(0);
            parentNode->removeChild(geode.get());
        }
    }
}

void GiswalkPlugin::straightFeed(double x, double y, double z, double a, double b, double c, double feedRate)
{
    /* positions[frameNumber*3  ] = x;
         positions[frameNumber*3+1] = y;
         positions[frameNumber*3+2] = z;*/

    vert->push_back(Vec3(x / 1000.0, y / 1000.0, z / 1000.0));
    float col = feedRate / 6000.0;
    if (col > 1)
        col = 1;
    color->push_back(getColor(col));
    frameNumber++;
    static double oldTime = 0;
    static double oldUpdateTime = 0;
    double time = cover->frameTime();
    if (time - oldUpdateTime > 1.0)
    {
        oldUpdateTime = time;
        char label[100];
        sprintf(label, "numSamples: %d", frameNumber);
        numSamples->setLabel(label);
    }
}
void GiswalkPlugin::tabletReleaseEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
}

osg::Vec4 GiswalkPlugin::getColor(float pos)
{

    osg::Vec4 actCol;
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

COVERPLUGIN(GiswalkPlugin)
