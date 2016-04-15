/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ViewerWidget.h"

ViewerWidget::ViewerWidget(QWidget *parent)
    : QWidget(parent)
    , scene(new SceneConf)
{

    load();
    setThreadingModel(threadingModel);

    QWidget *widget1 = addViewWidget(createCamera(0, 0, 100, 100), scene->getSceneGroup());

    QGridLayout *grid = new QGridLayout;
    grid->addWidget(widget1, 0, 0);
    setLayout(grid);

    connect(&_timer, SIGNAL(timeout()), this, SLOT(update()));
    _timer.start(10);
}

ViewerWidget::~ViewerWidget(void)
{
}

void ViewerWidget::load()
{
    /*
	//Lade ThreadingModel für Viewer
	std::string tModelType = coCoviseConfig::getEntry("COVER.Plugin.DistortConfig.threadingModel");
	
	if (!tModelType.empty())
	{
		if(strcasecmp(tModelType.c_str(),"SingleThreaded")==0)
			threadingModel = osgViewer::ViewerBase::SingleThreaded;
		if(strcasecmp(tModelType.c_str(),"CullDrawThreadPerContext")==0)
			threadingModel = osgViewer::ViewerBase::CullDrawThreadPerContext;
		if(strcasecmp(tModelType.c_str(),"DrawThreadPerContext")==0)
			threadingModel = osgViewer::ViewerBase::DrawThreadPerContext;
		if(strcasecmp(tModelType.c_str(),"CullThreadPerCameraDrawThreadPerContext")==0)
			threadingModel = osgViewer::ViewerBase::CullThreadPerCameraDrawThreadPerContext;
	}
	else threadingModel = osgViewer::ViewerBase::CullDrawThreadPerContext;
	*/
    threadingModel = osgViewer::ViewerBase::CullDrawThreadPerContext;
}

QWidget *ViewerWidget::addViewWidget(osg::Camera *camera, osg::Node *scene)
{
    osgViewer::View *view = new osgViewer::View;
    view->setCamera(camera);
    addView(view);

    view->setSceneData(scene);
    view->addEventHandler(new osgViewer::StatsHandler);
    view->setCameraManipulator(new osgGA::TrackballManipulator);

    osgQt::GraphicsWindowQt *gw = dynamic_cast<osgQt::GraphicsWindowQt *>(camera->getGraphicsContext());
    return gw ? gw->getGLWidget() : NULL;
}

osg::Camera *ViewerWidget::createCamera(int x, int y, int w, int h, const std::string &name, bool windowDecoration)
{
    osg::DisplaySettings *ds = osg::DisplaySettings::instance().get();
    osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
    traits->windowName = name;
    traits->windowDecoration = windowDecoration;
    if(windowDecoration == false)
    {
        traits->overrideRedirect = true; 
    }
    traits->x = x;
    traits->y = y;
    traits->width = w;
    traits->height = h;
    traits->doubleBuffer = true;
    traits->alpha = ds->getMinimumNumAlphaBits();
    traits->stencil = ds->getMinimumNumStencilBits();
    traits->sampleBuffers = ds->getMultiSamples();
    traits->samples = ds->getNumMultiSamples();

    osg::ref_ptr<osg::Camera> camera = new osg::Camera;
    camera->setGraphicsContext(new osgQt::GraphicsWindowQt(traits.get()));

    camera->setClearColor(osg::Vec4(0.2, 0.2, 0.6, 1.0));
    camera->setViewport(new osg::Viewport(0, 0, traits->width, traits->height));
    camera->setProjectionMatrixAsPerspective(
        30.0f, static_cast<double>(traits->width) / static_cast<double>(traits->height), 1.0f, 10000.0f);
    return camera.release();
}

void ViewerWidget::paintEvent(QPaintEvent *event)
{
    frame();
}