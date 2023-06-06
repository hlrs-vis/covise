/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: create windows with Qt
 **                                                                          **
 **                                                                          **
 ** Author: Martin Aumüller <aumueller@hlrs.de>
 **                                                                          **
\****************************************************************************/

#include "WindowTypeMesa.h"

#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coCommandLine.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <png.h>

using namespace opencover;
using covise::coCoviseConfig;


WindowTypeMesaPlugin::WindowTypeMesaPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "WindowTypeMesaPlugin::WindowTypeMesaPlugin\n");
    frameCounter=0;
	frameRate = coCoviseConfig::getInt("COVER.Mesa.FrameRate",1);
	writeRate = coCoviseConfig::getInt("COVER.Mesa.WriteRate",1);
	coVRConfig::instance()->setFrameRate(frameRate);
}

// this is called if the plugin is removed at runtime
WindowTypeMesaPlugin::~WindowTypeMesaPlugin()
{
    fprintf(stderr, "WindowTypeMesaPlugin::~WindowTypeMesaPlugin\n");
}

bool WindowTypeMesaPlugin::destroy()
{
    while (!m_windows.empty())
    {
        windowDestroy(m_windows.begin()->second.index);
    }
    return true;
}

bool WindowTypeMesaPlugin::update()
{
    return true;
}

bool WindowTypeMesaPlugin::windowCreate(int i)
{
    auto &conf = *coVRConfig::instance();
    auto it = m_windows.find(i);
    if (it != m_windows.end())
    {
        std::cerr << "WindowTypeQt: already managing window no. " << i << std::endl;
        return false;
    }

    auto &win = m_windows[i];
    win.index = i;

    win.buffer = new char[conf.windows[i].sx*conf.windows[i].sy*14];
    win.width = conf.windows[i].sx;
    win.height=conf.windows[i].sy;
    win.context = OSMesaCreateContext(GL_RGBA, NULL);
OSMesaMakeCurrent(win.context, win.buffer, GL_UNSIGNED_BYTE, conf.windows[i].sx, conf.windows[i].sy);

    coVRConfig::instance()->windows[i].window = new osgViewer::GraphicsWindowEmbedded(0,0,conf.windows[i].sx, conf.windows[i].sy);
    coVRConfig::instance()->windows[i].context = coVRConfig::instance()->windows[i].window;
    //std::cerr << "window " << i << ": ctx=" << coVRConfig::instance()->windows[i].context << std::endl;
    return true;
}

void WindowTypeMesaPlugin::windowCheckEvents(int num)
{
}

void WindowTypeMesaPlugin::windowUpdateContents(int num)
{
    auto it = m_windows.find(num);
    if (it == m_windows.end())
    {
        std::cerr << "WindowTypeQt: window no. " << num << " not managed by this plugin" << std::endl;
        return;
    }
    auto &win = it->second;
    char filename[100];
    sprintf(filename,"test%d.png",frameCounter);
    frameCounter++;
    if((frameCounter % writeRate) == 0)
{
    writeImage(filename,win.width,win.height,win.buffer,filename);
}

}

void WindowTypeMesaPlugin::windowDestroy(int num)
{
    auto it = m_windows.find(num);
    if (it == m_windows.end())
    {
        std::cerr << "WindowTypeQt: window no. " << num << " not managed by this plugin" << std::endl;
        return;
    }

    auto &conf = *coVRConfig::instance();
    conf.windows[num].context = nullptr;
    conf.windows[num].windowPlugin = nullptr;
    conf.windows[num].window = nullptr;

    auto &win = it->second;
    m_windows.erase(it);

}

int WindowTypeMesaPlugin::writeImage(char* filename, int width, int height, char *buffer, char* title)
{
	int code = 0;
	FILE *fp = NULL;
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep row = NULL;
	
	// Open file for writing (binary mode)
	fp = fopen(filename, "wb");
	if (fp == NULL) {
		fprintf(stderr, "Could not open file %s for writing\n", filename);
		code = 1;
		goto finalise;
	}

	// Initialize write structure
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		fprintf(stderr, "Could not allocate write struct\n");
		code = 1;
		goto finalise;
	}

	// Initialize info structure
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fprintf(stderr, "Could not allocate info struct\n");
		code = 1;
		goto finalise;
	}

	// Setup Exception handling
	if (setjmp(png_jmpbuf(png_ptr))) {
		fprintf(stderr, "Error during png creation\n");
		code = 1;
		goto finalise;
	}

	png_init_io(png_ptr, fp);

	// Write header (8 bit colour depth)
	png_set_IHDR(png_ptr, info_ptr, width, height,
			8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	// Set title
	if (title != NULL) {
		png_text title_text;
		title_text.compression = PNG_TEXT_COMPRESSION_NONE;
                title_text.key = new char[100];
		strcpy(title_text.key,"Title");
		title_text.text = title;
		png_set_text(png_ptr, info_ptr, &title_text, 1);
               delete[] title_text.key;
	}

	png_write_info(png_ptr, info_ptr);

	// Write image data
	int x, y;
	for (y=0 ; y<height ; y++) {
		png_write_row(png_ptr, (unsigned char *)(buffer+((height -y)*width*4)));
	}

	// End write
	png_write_end(png_ptr, NULL);

	finalise:
	if (fp != NULL) fclose(fp);
	if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	if (row != NULL) free(row);

	return code;
}


COVERPLUGIN(WindowTypeMesaPlugin)
