// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/pvs/glut_wrapper.h>

namespace lamure
{
namespace pvs
{

glut_management* glut_wrapper::manager;

void glut_wrapper::initialize(int argc, char** argv, const uint32_t& width, const uint32_t& height, glut_management* manager)
{
	glut_wrapper::manager = manager;

	glutInit(&argc, argv);
    glutInitContextVersion(4, 4);
    glutInitContextProfile(GLUT_CORE_PROFILE);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA | GLUT_MULTISAMPLE);

    glutInitWindowPosition(400,300);
    glutInitWindowSize(width, height);

    int window = glutCreateWindow("PVS Renderer");

    glutSetWindow(window);

    glutReshapeFunc(glut_wrapper::resize);
    glutDisplayFunc(glut_wrapper::display);
    glutKeyboardFunc(glut_wrapper::keyboard);
    glutKeyboardUpFunc(glut_wrapper::keyboard_release);
    glutMouseFunc(glut_wrapper::mousefunc);
    glutMotionFunc(glut_wrapper::mousemotion);
    glutIdleFunc(glut_wrapper::idle);
}

void glut_wrapper::set_management(glut_management* manager)
{
    glut_wrapper::manager = manager;
}

void glut_wrapper::quit()
{
    glutExit();
}

void glut_wrapper::resize(int w, int h)
{
    if (glut_wrapper::manager != nullptr)
    {
        glut_wrapper::manager->dispatchResize(w, h);
    }
}

void glut_wrapper::display()
{
	bool signaled_shutdown = false;

    if (glut_wrapper::manager != nullptr)
    {
        signaled_shutdown = glut_wrapper::manager->MainLoop(); 
        glutSwapBuffers();
    }

    if(signaled_shutdown)
    {
        glutLeaveMainLoop();
    }

    // Exiting glut provokes an exception in schism.
    // Prefer to exit when the application quits. 
    /*if(signaled_shutdown)
    {
        glutExit();
        exit(0);
    }*/
}

void glut_wrapper::keyboard(unsigned char key, int x, int y)
{
    switch(key)
    {
        case 27:
            glutExit();
            exit(0);
            break;

        case '.':
            glutFullScreenToggle();
            break;

        default:
            if (glut_wrapper::manager != nullptr)
            {
                glut_wrapper::manager->dispatchKeyboardInput(key);
            }
            break;
    }
}

void glut_wrapper::keyboard_release(unsigned char key, int x, int y)
{
}

void glut_wrapper::mousefunc(int button, int state, int x, int y)
{
    if (glut_wrapper::manager != nullptr)
    {
        glut_wrapper::manager->RegisterMousePresses(button, state, x, y);
    }
}

void glut_wrapper::mousemotion(int x, int y)
{
    if (glut_wrapper::manager != nullptr)
    {
        glut_wrapper::manager->update_trackball(x, y);
    }
}

void glut_wrapper::idle()
{
	glutPostRedisplay();
}

}
}
