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
#include <cover/VRSceneGraph.h>
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
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>
#include "cover/coIntersection.h"
#include <config/CoviseConfig.h>
#define MAXSAMPLES 1200
using namespace osg;
using namespace osgUtil;

RecordPathPlugin::RecordPathPlugin() : ui::Owner("RecordPathPlugin", cover->ui)
{
    std::string proj_from = coCoviseConfig::getEntry("from", "COVER.Plugin.RecordPath.Projection", "+proj=latlong +datum=WGS84");
    if (!(pj_from = pj_init_plus(proj_from.c_str())))
    {
        fprintf(stderr, "ERROR: pj_init_plus failed with pj_from = %s\n", proj_from.c_str());
    }

    std::string proj_to = coCoviseConfig::getEntry("to", "COVER.Plugin.RecordPath.Projection", "+proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=9703.397 +y_0=-5384244.453 +ellps=bessel +datum=potsdam");// +nadgrids=" + dir + std::string("BETA2007.gsb");

    if (!(pj_to = pj_init_plus(proj_to.c_str())))
    {
        fprintf(stderr, "ERROR: pj_init_plus failed with pj_to = %s\n", proj_to.c_str());
    }
    projectOffset[0] = coCoviseConfig::getFloat("offetX", "COVER.Plugin.JSBSim.Projection", 0);
    projectOffset[1] = coCoviseConfig::getFloat("offetY", "COVER.Plugin.JSBSim.Projection", 0);
    projectOffset[2] = coCoviseConfig::getFloat("offetZ", "COVER.Plugin.JSBSim.Projection", 0);
}

bool RecordPathPlugin::init()
{
    fprintf(stderr, "RecordPathPlugin::RecordPathPlugin\n");

    length = 1;
    recordRate = 1;
    filename = "path.csv";


    recordPathMenu = new ui::Menu("RecordPath", this);

    record = new ui::Button(recordPathMenu, "Record");
    record->setText("Record");
    record->setState(false);
    record->setCallback([this](bool state) {
        if (state)
        {
            playing = false;
        }
        else
        {
        }
        });

    play = new ui::Button(recordPathMenu, "Play");
    play->setText("Play");
    play->setState(false);
    play->setCallback([this](bool state) {
        if (state)
        {

            playing = true;
        }
        else
        {
            playing = false;
        }
        });

    reset = new ui::Action(recordPathMenu, "reset");
    reset->setText("reset");
    reset->setCallback([this]() {
        frameNumber = 0;
        record->setState(false);
        playing = false;
        });

    saveButton = new ui::Action(recordPathMenu, "save");
    saveButton->setText("save");
    saveButton->setCallback([this]() {
            save();
        });

    viewPath = new ui::Button(recordPathMenu, "viewPath");
    viewPath->setText("viewPath");
    viewPath->setState(false);
    viewPath->setCallback([this](bool state) {
        char label[100];
        sprintf(label, "numSamples: %d", frameNumber);
        numSamples->setText(label);
        if (state)
        {
            geode = new Geode();
            geom = new Geometry();
            geode->setStateSet(geoState.get());

            geom->setColorBinding(Geometry::BIND_OFF);

            geode->addDrawable(geom.get());
            geode->setName("Viewer Positions");
            // set up geometry
            Vec3Array* vert = new Vec3Array;
            DrawArrayLengths* primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
            primitives->push_back(frameNumber);
            for (int n = 0; n < frameNumber; n++)
            {
                vert->push_back(Vec3(positions[n * 3], positions[n * 3 + 1], positions[n * 3 + 2]));
            }
            geom->setVertexArray(vert);
            geom->addPrimitiveSet(primitives);
            geom->dirtyDisplayList();
            opencover::cover->getObjectsRoot()->addChild(geode.get());
        }
        else
        {
            opencover::cover->getObjectsRoot()->removeChild(geode.get());
        }
        });

    viewlookAt = new ui::Button(recordPathMenu, "viewlookAt");
    viewlookAt->setText("viewlookAt");
    viewlookAt->setState(false);
    viewlookAt->setCallback([this](bool state) {
        if (state)
        {
            float* radii, r;
            r = radiusEdit->number();
            radii = new float[frameNumber];
            for (int n = 0; n < frameNumber; n++)
                radii[n] = r;
            lookAtSpheres->setCoords(frameNumber, lookat[0], lookat[1], lookat[2], radii);
            for (int n = 0; n < frameNumber; n++)
                lookAtSpheres->setColor(n, 1, 0.2, 0.2, 1);
            opencover::cover->getObjectsRoot()->addChild(lookAtGeode.get());
        }
        else
        {
            opencover::cover->getObjectsRoot()->removeChild(lookAtGeode.get());

            lookAtGeode = new Geode();
            lookAtGeode->addDrawable(lookAtSpheres.get());
            lookAtGeode->setName("Viewer lookAtPositions");
        }
        });

    viewDirections = new ui::Button(recordPathMenu, "viewDirections");
    viewDirections->setText("viewDirections");
    viewDirections->setState(false);
    viewDirections->setCallback([this](bool state) {
        if (state)
        {

            dirGeode = new Geode();
            dirGeom = new Geometry();
            dirGeode->setStateSet(geoState.get());

            dirGeom->setColorBinding(Geometry::BIND_OFF);

            dirGeode->addDrawable(dirGeom.get());
            dirGeode->setName("Viewer Positions");
            // set up geometry
            Vec3Array* vert = new Vec3Array;
            DrawArrayLengths* primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
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
            opencover::cover->getObjectsRoot()->addChild(dirGeode.get());
        }
        else
        {
            opencover::cover->getObjectsRoot()->removeChild(dirGeode.get());
        }
        });


    lengthEdit = new ui::EditField(recordPathMenu, "length");
    lengthEdit->setText("length");
    lengthEdit->setCallback([this](std::string value) {
        length = lengthEdit->number();
        });

    recordRateEdit = new ui::EditField(recordPathMenu, "recordRate");
    recordRateEdit->setText("recordRate");
    recordRateEdit->setCallback([this](std::string value) {
        recordRate = 1.0 / recordRateEdit->number();
        });

    renderMethod = new ui::SelectionList(recordPathMenu, "renderMethod");
    renderMethod->setText("Task type");
    renderMethod->append("CPU Billboard");
    renderMethod->append("Cg Shader");
    renderMethod->append("Point Sprite");
    renderMethod->select(0);
    renderMethod->setCallback([this](int idx) {
        lookAtSpheres->setRenderMethod((coSphere::RenderMethod)idx);
        });



    filenameEdit = new ui::EditField(recordPathMenu, "filename");
    lengthEdit->setText("path.csv");
    lengthEdit->setCallback([this](std::string value) {
        filename = value;
        });

    numSamples = new ui::Label(recordPathMenu, "numSamples");
    numSamples->setText("");


    positions = new float[3 * MAXSAMPLES + 3];
    lookat[0] = new float[MAXSAMPLES + 1];
    lookat[1] = new float[MAXSAMPLES + 1];
    lookat[2] = new float[MAXSAMPLES + 1];
    objectName = new const char *[MAXSAMPLES + 3];
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
    delete play;
    delete reset;
    delete saveButton;
    delete viewPath;
    delete viewDirections;
    delete viewlookAt;
    delete lengthEdit;
    delete radiusEdit;
    delete renderMethod;
    delete recordRateEdit;
    delete filenameEdit;
    delete numSamples;
    delete recordPathMenu;

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
    if (record->state())
    {
        static double oldTime = 0;
        static double oldUpdateTime = 0;
        double time = cover->frameTime();
        if (time - oldUpdateTime > 1.0)
        {
            oldUpdateTime = time;
            char label[100];
            sprintf(label, "numSamples: %d", frameNumber);
            numSamples->setText(label);
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

            osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
            osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector;
            intersector = coIntersection::instance()->newIntersector(ray->start(), ray->end());
            igroup->addIntersector(intersector);

            osgUtil::IntersectionVisitor visitor(igroup);
            visitor.setTraversalMask(Isect::Walk);
            VRSceneGraph::instance()->getTransform()->accept(visitor);

            if (intersector->containsIntersections())
            {

                osgUtil::LineSegmentIntersector::Intersection is = intersector->getFirstIntersection();
                q0 = is.getWorldIntersectPoint();
                auto npIt = is.nodePath.end();
                if (npIt != is.nodePath.begin())
                {
                    npIt--;
                    osg::Node* n = *(npIt);
                    objectName[frameNumber] = n->getName().c_str();
                }
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
        };
    }
}

void RecordPathPlugin::save()
{
    FILE *fp = fopen(filename.c_str(), "w");
    if (fp)
    {
        fprintf(fp, "# lat, lon, x,      y,      z,      dx,      dy,     dz\n");
        fprintf(fp, "# numFrames: %d\n", frameNumber);
        for (int n = 0; n < frameNumber; n++)
        { 

        double v[3];

        v[0] = positions[n * 3 + 0]- projectOffset[0];
        v[1] = positions[n * 3 + 1] - projectOffset[1];
        v[2] = positions[n * 3 + 2] - projectOffset[2];
        int error = pj_transform(pj_to, pj_from, 1, 0, v,v+1, v+2);
        if (error != 0)
        {
            fprintf(stderr, "%s \n ------ \n", pj_strerrno(error));
        }
        double mLon = v[0];
        double mLat = v[1];
            fprintf(fp, "%010.3f,%010.3f,%010.3f,%010.3f,%010.3f,%010.3f,%010.3f,%010.3f\n", mLat, mLon, positions[n * 3], positions[n * 3 + 1], positions[n * 3 + 2], lookat[0][n], lookat[1][n], lookat[2][n]);
        }
        fclose(fp);
    }
    else
    {
        cerr << "could not open file " << filename << endl;
    }
}


COVERPLUGIN(RecordPathPlugin)
