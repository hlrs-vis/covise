// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef GLUT_MANAGEMENT_H
#define GLUT_MANAGEMENT_H

namespace lamure
{
namespace pvs
{

class glut_management
{
public:
    virtual bool MainLoop() = 0;
    virtual void update_trackball(int x, int y) = 0;
    virtual void RegisterMousePresses(int button, int state, int x, int y) = 0;
    virtual void dispatchKeyboardInput(unsigned char key) = 0;
    virtual void dispatchResize(int w, int h) = 0;
};

}
}

#endif