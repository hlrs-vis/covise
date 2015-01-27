/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cmath>
#include <vector>
#include <iostream>
#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>
#include <osgProducer/Viewer>
#include <osg/PositionAttitudeTransform>
#include "coPlotCoordSystem.h"

using std::endl;
using std::cout;

static Producer::CameraConfig *BuildConfig(void)
{
    // create a new render surface
    Producer::RenderSurface *rsWindow = new Producer::RenderSurface;

    // set the number of screens
    rsWindow->setScreenNum(0);

    // set the window title bar caption
    rsWindow->setWindowName("osgFncPlotter Test");

    // set the window client size
    rsWindow->setWindowRectangle(0, 0, 800, 800);

    // create a new camera
    Producer::Camera *camera1 = new Producer::Camera;

    // set the render surface of the camera to the window that was just created
    camera1->setRenderSurface(rsWindow);

    // set the size of the camera viewport to the same size as the window
    camera1->setProjectionRectangle(0, 0, 800, 800);

    // set the camera position
    // y is up
    camera1->setViewByLookat(Producer::Vec3(0.0f, 0.0f, 0.0f), Producer::Vec3(0.0f, -1.0f, 0.0f), Producer::Vec3(0.0f, 0.0f, 1.0f));

    // create a new camera configuration container
    Producer::CameraConfig *cfg = new Producer::CameraConfig;

    // add the camera that was just created to the configuration
    cfg->addCamera("Camera One", camera1);

    return (cfg);
}

int main(int argc, char **argv)
{
    osg::Group *rootNode = new osg::Group();

    //float max=pow(2.0f, 32), min=4.0f;
    float max = 10, min = -5;

    std::vector<float> g1, g2, g3;
    for (float f = min; f < max; f += 0.25f)
    {
        g1.push_back(f);
        g1.push_back(sin(f));

        g2.push_back(f);
        g2.push_back(f);

        if (f >= 0)
        {
            g3.push_back(f);
            g3.push_back(log(f));
        }
    }

    // Now add our coordinate system
    CoordSystem coordsys;
#ifdef WIN32
    coordsys.setFont("c:\\windows\\fonts\\cour.ttf");
#else
    coordsys.setFont("/usr/X11R6/lib/X11/fonts/TTF/luximr.ttf");
#endif
    coordsys.setFontColor(osg::Vec4(1, 1, 1, 1));
    // move away from camera so we can see something
    coordsys.setPosition(osg::Vec3(0, -1000, 0));
    coordsys.setMinMaxValues(min, max, min, max);

    // Now add every object for rendering
    rootNode->addChild(coordsys.createWithLinearScaling().get());
    //rootNode->addChild( coordsys.createWithBinaryLogScaling().get() );

    // add some graphs here
    coordsys.addGraph(new CoordSystem::Graph(&g1[0], g1.size() / 2, osg::Vec4(1, 1, 1, 1)));
    //coordsys.addGraph(new CoordSystem::Graph(&g2[0], g2.size()/2, osg::Vec4(1,0,0,1) ) );
    //coordsys.addGraph(new CoordSystem::Graph(&g3[0], g3.size()/2, osg::Vec4(0,1,0,1) ) );

    osgProducer::Viewer viewer(BuildConfig());
    viewer.setUpViewer(osgProducer::Viewer::ESCAPE_SETS_DONE);

    // add model to viewer.
    viewer.setSceneData(rootNode);

    // create the windows and run the threads.
    viewer.realize();

    while (!viewer.done())
    {
        // wait for all cull and draw threads to complete.
        viewer.sync();

        // update the scene by traversing it with the the update visitor which will
        // call all node update callbacks and animations.
        viewer.update();

        // fire off the cull and draw traversals of the scene.
        viewer.frame();
    }

    // wait for all cull and draw threads to complete before exit.
    viewer.sync();

    return (0);
}
