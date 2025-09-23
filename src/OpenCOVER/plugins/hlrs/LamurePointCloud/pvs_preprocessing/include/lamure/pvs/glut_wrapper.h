// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef GLUT_WRAPPER_H
#define GLUT_WRAPPER_H

#include <GL/freeglut.h>
#include <cstdint>

#include "lamure/pvs/glut_management.h"

namespace lamure
{
namespace pvs 
{

class glut_wrapper
{
	public:
		static void initialize(int argc, char** argv, const uint32_t& width, const uint32_t& height, glut_management* manager);
        static void set_management(glut_management* manager);
        static void quit();

	private:
		static void resize(int w, int h);
    	static void display();
    	static void keyboard(unsigned char key, int x, int y);
    	static void keyboard_release(unsigned char key, int x, int y);
    	static void mousefunc(int button, int state, int x, int y);
    	static void mousemotion(int x, int y);
    	static void idle();

        static glut_management* manager;
};

}
}

#endif
