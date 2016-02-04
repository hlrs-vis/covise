/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include "SceneConf.h"

#include <QTimer>
#include <QApplication>
#include <QGridLayout>

#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>

#include <osgGA/TrackballManipulator>

#include <osgDB/ReadFile>

#include <osgQt/GraphicsWindowQt>

#include <iostream>

class ViewerWidget : public QWidget, public osgViewer::CompositeViewer
{
public:
    ViewerWidget(QWidget *parent);
    ~ViewerWidget(void);
    void load(void);
    QWidget *addViewWidget(osg::Camera *camera, osg::Node *scene);
    osg::Camera *createCamera(int x, int y, int w, int h, const std::string &name = "", bool windowDecoration = false);
    virtual void paintEvent(QPaintEvent *event);

    Scene *getScene(void)
    {
        return scene;
    };

protected:
    osgViewer::ViewerBase::ThreadingModel threadingModel;
    Scene *scene;
    QTimer _timer;
};
