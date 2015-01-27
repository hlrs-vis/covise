/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE ColorMap application module                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

#define COLORMAP_WIDTH 256
#define TEXTURE_HEIGHT 1
#define TEXTURE_LEVEL 0
#define PIXEL_SIZE 4
#define PIXEL_FORMAT 4

class Application
{

private:
    // callback stub functions
    //
    static void computeCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    // private member functions
    //
    void compute(void *callbackData);
    void param(const char *, bool inMapLoading);
    void quit(void *callbackData);

    void interpolate_colmap(float (*colmap)[3], int color,
                            int first, int last,
                            float first_val, float last_val);

    void print_colmap(float (*colmap)[3], int first, int last);

    void make_rgba_colors(float *scalar, int np, int ncolor,
                          int *pc, float *rcol, float *gcol, float *bcol, float *acol,
                          float par_min, float par_max);

    int pack_rgba_color(float rcol, float gcol, float bcol, float alpha);
    void make_texCoord(float *, int, int, float, float);
    void initColormap(void);

    // private data
    char colormap_as_an_attribute[8000];
    int map; // type of mapping (RGBA colors or texture coordinates)
    long steps; // number of steps in colormap
    float min, max;
    int *pc;
    float **tex_coords; // array of texture coordinates
    float *scalar;

public:
    Application(int argc, char *argv[])
    {

        Covise::set_module_description("Map scalar data to rgb colors in packed color format");
        Covise::add_port(INPUT_PORT, "Data", "Set_Float|Set_Float", "scalar data");
        Covise::add_port(INPUT_PORT, "minmax", "MinMax_Data", "Minimum And Maximum Values");
        Covise::add_port(OUTPUT_PORT, "Colors", "Set_RGBA|Set_Texture", "mapped rgb colors");

        Covise::add_port(PARIN, "colormap_file", "Browser", "filename");
        Covise::set_port_default("colormap_file", "/var/tmp/ *");

        Covise::add_port(PARIN, "Map", "Choice", "mapping (color|texture)");
        Covise::add_port(PARIN, "Min", "FloatScalar", "Min");
        Covise::add_port(PARIN, "Max", "FloatScalar", "Max");
        Covise::add_port(PARIN, "colormap", "Colormap", "Colors");
        Covise::add_port(PARIN, "Annotation", "String", "Annotation");
        Covise::add_port(PARIN, "Steps", "IntScalar", "Steps of colormap");
        Covise::set_port_default("Map", "1 color texture");
        Covise::set_port_default("Min", "0.0");
        Covise::set_port_default("Max", "0.0");
        Covise::set_port_default("colormap", "0.0 1.0 256");
        Covise::set_port_default("Annotation", "Colors");
        Covise::set_port_default("Steps", "0");
        Covise::set_port_required("minmax", 0);
        Covise::init(argc, argv);

        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_param_callback(Application::paramCallback, this);

        initColormap();
    }

    // fprintf(stderr,"starting\n");
    // fprintf(stderr,"covise init\n");
    // fprintf(stderr,"init Colotmap\n");
    void handle_objects(coDistributedObject *data_obj, char *Outname, coDistributedObject **set_out = NULL);
    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _APPLICATION_H
