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

#include "RecordPathPlugin.h"
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
#define MAXSAMPLES 1200
using namespace osg;
using namespace osgUtil;

RecordPathPlugin::RecordPathPlugin()
{
}

bool RecordPathPlugin::init()
{
    fprintf(stderr, "RecordPathPlugin::RecordPathPlugin\n");

    length = 1;
    recordRate = 1;
    filename = NULL;

    PathTab = new coTUITab("RecordPath", coVRTui::instance()->mainFolder->getID());
    record = new coTUIToggleButton("Record", PathTab->getID());
    stop = new coTUIButton("Stop", PathTab->getID());
    play = new coTUIButton("Play", PathTab->getID());
    reset = new coTUIButton("Reset", PathTab->getID());
    saveButton = new coTUIButton("Save", PathTab->getID());

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
    positions = new float[3 * MAXSAMPLES + 3];
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
    linemtl = new Material;
    lineWidth = new LineWidth(2.0);
    linemtl.get()->setColorMode(Material::OFF);
    linemtl.get()->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    linemtl.get()->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
    linemtl.get()->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
    linemtl.get()->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    linemtl.get()->setShininess(Material::FRONT_AND_BACK, 16.0f);

    geoState->setAttributeAndModes(linemtl.get(), StateAttribute::ON);

    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    geoState->setAttributeAndModes(lineWidth.get(), StateAttribute::ON);

    lookAtSpheres = new coSphere();

    lookAtSpheres->setRenderMethod(coSphere::RENDER_METHOD_ARB_POINT_SPRITES);

    return true;
}

// this is called if the plugin is removed at runtime
RecordPathPlugin::~RecordPathPlugin()
{
    fprintf(stderr, "RecordPathPlugin::~RecordPathPlugin\n");

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

    delete[] positions;
    delete[] lookat[0];
    delete[] lookat[1];
    delete[] lookat[2];
    delete[] objectName;
    cover->getObjectsRoot()->removeChild(lookAtGeode.get());
    cover->getObjectsRoot()->removeChild(dirGeode.get());
    cover->getObjectsRoot()->removeChild(geode.get());
}

void
RecordPathPlugin::preFrame()
{
    if (record->getState())
    {
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
        if (time - oldTime > recordRate)
        {
            oldTime = time;
            osg::Matrix mat = cover->getInvBaseMat();
            osg::Matrix viewMat = cover->getViewerMat();
            osg::Matrix viewMatObj = viewMat * mat;

            positions[frameNumber * 3] = viewMatObj.getTrans().x();
            positions[frameNumber * 3 + 1] = viewMatObj.getTrans().y();
            positions[frameNumber * 3 + 2] = viewMatObj.getTrans().z();

            Vec3 q0, q1;
            q0.set(0.0f, 0.0f, 0.0f);
            q1.set(0.0f, 100000, 0.0f);

            // xform the intersection line segment
            Matrix handMat = cover->getViewerMat();
            q0 = handMat.preMult(q0);
            q1 = handMat.preMult(q1);

            ref_ptr<LineSegment> ray = new LineSegment();
            ray->set(q0, q1);

            IntersectVisitor visitor;
            visitor.setTraversalMask(Isect::Intersection);
            visitor.addLineSegment(ray.get());

            cover->getScene()->accept(visitor);

            if (visitor.getNumHits(ray.get()))
            {
                //VRUILOG("coIntersection::intersect info: hit")
                Hit hitInformation = visitor.getHitList(ray.get()).front();
                q0 = hitInformation.getWorldIntersectPoint();
                objectName[frameNumber] = hitInformation._geode->getName().c_str();
                osg::Vec3 temp;
                temp = q0 * cover->getInvBaseMat();

                lookat[0][frameNumber] = temp.x();
                lookat[1][frameNumber] = temp.y();
                lookat[2][frameNumber] = temp.z();
            }
            else
            {
                osg::Vec3 temp;
                osg::Vec3 dir(viewMatObj(1, 0), viewMatObj(1, 1), viewMatObj(1, 2));
                temp = viewMatObj.getTrans() + dir;
                temp = temp * cover->getInvBaseMat();
                lookat[0][frameNumber] = temp.x();
                lookat[1][frameNumber] = temp.y();
                lookat[2][frameNumber] = temp.z();
                objectName[frameNumber] = NULL;
            }

            frameNumber++;

            if (frameNumber >= MAXSAMPLES)
            {
                record->setState(false);
                playing = false;
            }
        }
    }
}

void RecordPathPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == lengthEdit)
    {
        length = lengthEdit->getValue();
    }
    else if (tUIItem == renderMethod)
    {
        lookAtSpheres->setRenderMethod((coSphere::RenderMethod)renderMethod->getSelectedEntry());
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
void RecordPathPlugin::save()
{
    FILE *fp = fopen(filename, "w");
    if (fp)
    {
        fprintf(fp, "# x,      y,      z,      dx,      dy,     dz\n");
        fprintf(fp, "# numFrames: %d\n", frameNumber);
        for (int n = 0; n < frameNumber; n++)
        {
            fprintf(fp, "%010.3f,%010.3f,%010.3f,%010.3f,%010.3f,%010.3f\n", positions[n * 3], positions[n * 3 + 1], positions[n * 3 + 2], lookat[0][n], lookat[1][n], lookat[2][n]);
        }
        fclose(fp);
    }
    else
    {
        cerr << "could not open file " << filename << endl;
    }
}

void RecordPathPlugin::tabletPressEvent(coTUIElement *tUIItem)
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
    else if (tUIItem == viewlookAt)
    {
        if (viewlookAt->getState())
        {
            float *radii, r;
            r = radiusEdit->getValue();
            radii = new float[frameNumber];
            for (int n = 0; n < frameNumber; n++)
                radii[n] = r;
            lookAtSpheres->setCoords(frameNumber, lookat[0], lookat[1], lookat[2], radii);
            for (int n = 0; n < frameNumber; n++)
                lookAtSpheres->setColor(n, 1, 0.2, 0.2, 1);
            cover->getObjectsRoot()->addChild(lookAtGeode.get());
        }
        else
        {
            cover->getObjectsRoot()->removeChild(lookAtGeode.get());

            lookAtGeode = new Geode();
            lookAtGeode->addDrawable(lookAtSpheres.get());
            lookAtGeode->setName("Viewer lookAtPositions");
        }
    }
    else if (tUIItem == viewPath)
    {
        char label[100];
        sprintf(label, "numSamples: %d", frameNumber);
        numSamples->setLabel(label);
        if (viewPath->getState())
        {
            geode = new Geode();
            geom = new Geometry();
            geode->setStateSet(geoState.get());

            geom->setColorBinding(Geometry::BIND_OFF);

            geode->addDrawable(geom.get());
            geode->setName("Viewer Positions");
            // set up geometry
            Vec3Array *vert = new Vec3Array;
            DrawArrayLengths *primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
            primitives->push_back(frameNumber);
            for (int n = 0; n < frameNumber; n++)
            {
                vert->push_back(Vec3(positions[n * 3], positions[n * 3 + 1], positions[n * 3 + 2]));
            }
            geom->setVertexArray(vert);
            geom->addPrimitiveSet(primitives);
            geom->dirtyDisplayList();
            cover->getObjectsRoot()->addChild(geode.get());
        }
        else
        {
            cover->getObjectsRoot()->removeChild(geode.get());
        }
    }
    else if (tUIItem == viewDirections)
    {
        if (viewDirections->getState())
        {

            dirGeode = new Geode();
            dirGeom = new Geometry();
            dirGeode->setStateSet(geoState.get());

            dirGeom->setColorBinding(Geometry::BIND_OFF);

            dirGeode->addDrawable(dirGeom.get());
            dirGeode->setName("Viewer Positions");
            // set up geometry
            Vec3Array *vert = new Vec3Array;
            DrawArrayLengths *primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
            for (int n = 0; n < frameNumber; n++)
            {
                primitives->push_back(2);
                Vec3 pos(positions[n * 3], positions[n * 3 + 1], positions[n * 3 + 2]);
                Vec3 look(lookat[0][n], lookat[1][n], lookat[2][n]);
                Vec3 diff = look - pos;
                diff.normalize();
                diff *= length;
                vert->push_back(pos);

                vert->push_back(pos + diff);
            }
            dirGeom->setVertexArray(vert);
            dirGeom->addPrimitiveSet(primitives);
            cover->getObjectsRoot()->addChild(dirGeode.get());
        }
        else
        {
            cover->getObjectsRoot()->removeChild(dirGeode.get());
        }
    }
}

void RecordPathPlugin::tabletReleaseEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
}

COVERPLUGIN(RecordPathPlugin)
