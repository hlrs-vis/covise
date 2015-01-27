/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE ColorEdit application module                      **
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
#include "ColorEdit.h"
#include <colormap/IntColorMap.h>

// GLOBAL VARIABLES
//
static int depth;
int i_dim, j_dim, k_dim, npoint, ncolor;
int np, norm_attr, inst;
char *dataType;
char *Data, *Colors;
float colormap[4][COLORMAP_WIDTH];
char *annotation;
char *annotation_default = "Colors";
char *image;

coDoSet *data = NULL;
coDistributedObject *tmp_obj, *data_obj;
coDoFloat *s_data = NULL;
coDoFloat *u_data = NULL;
coDoRGBA *p_colors = NULL;
coDoTexture *p_texture = NULL;
coDoPixelImage *colorLUT = NULL; // look-up-table for color

int main(int argc, char *argv[])
{
    Application *application = new Application(argc, argv);
    application->run();
    return 0;
}

//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

void Application::paramCallback(bool inMapLoading, void *userData, void *callbackData)
{
    (void)callbackData;
    Application *thisApp = (Application *)userData;
    const char *paramname = Covise::get_reply_param_name();
    thisApp->param(paramname, inMapLoading);
}

//
//
//..........................................................................
//
//

//======================================================================
// Called to request the initial colormap
//======================================================================
void Application::initColormap(void)
{
    Covise::request_param("colormap");
}

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
}

//======================================================================
// Immediate mode parameter routine (called when PARAM message arrrives)
//======================================================================
void Application::param(const char *paramName, bool inMapLoading)
{
    if (strcmp(paramName, "colormap_file") == 0)
    {
        // get filename
        const char *newVal;
        char *filename = "ColorEdit.dummy";
        if (Covise::get_reply_browser(&newVal))
        {
            char buffer[1024];
            Covise::getname(buffer, newVal);
            if (buffer[0] == '\0' && newVal)
                filename = strcpy(new char[strlen(newVal) + 1], newVal);
            else
                filename = strcpy(new char[strlen(buffer) + 1], buffer);
        }

        //load file
        IntColorMap colorMap(filename);
        char puffer[1024];

        if (colorMap.isValid())
        {
            int size = colorMap.getSize();
            char string[30];
            snprintf(string, sizeof(string), "%i", (8 + size));
#ifdef _STANDARD_C_PLUS_PLUS
            std::ostringstream buffer;
#else
            ostrstream buffer;
#endif
            buffer << "colormap" << endl
                   << "Colormap" << endl
                   << string << endl
                   << "loadColorMap\n";
            colorMap.write(buffer);

#ifdef _STANDARD_C_PLUS_PLUS
            std::string str = buffer.str();
            const char *message = str.c_str();
#else
            const char *message = buffer.str();
#endif

            // send message to all UIF
            Covise::send_ui_message("PARAM_RESTORE", message);
            Covise::send_ui_message("PARAM_SLAVE", message);
        }
        else
        {
            if (!inMapLoading)
            {
                sprintf(puffer, "%s is not a valid colormap file.", filename);
                Covise::sendError(puffer);
            }
        }
    }
    else if (strcmp(paramName, "Min") == 0)
    {
        Covise::get_reply_float_scalar(&min);
    }
    else if (strcmp(paramName, "Max") == 0)
    {
        Covise::get_reply_float_scalar(&max);
    }
    else if (strcmp(paramName, "colormap") == 0)
    {
        colormap_type type;
        Covise::get_reply_colormap(&min, &max, &ncolor, &type);
        Covise::update_scalar_param("Min", min);
        Covise::update_scalar_param("Max", max);

        if (ncolor <= COLORMAP_WIDTH)
        {
            for (int i = 0; i < ncolor; i++)
            {
                Covise::get_reply_colormap(i, &colormap[0][i], &colormap[1][i], &colormap[2][i], &colormap[3][i], &colormap[4][i]);
            }
        }
        else
        {
            char buf[255];
            sprintf(buf, "Colormap > %d is too long for me.", COLORMAP_WIDTH);
            Covise::sendWarning(buf);
        }
    }
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
void Application::compute(void *)
{
    int i, j;
    //float my_min,my_max;
    char *MinMax_Name;
    float *s;
    // restore map after interpolation
    float restore[5][COLORMAP_WIDTH];

    coDoFloat *minmax_data = NULL;

    depth = 0;

    Covise::get_choice_param("Map", &map);
    // min and max saved in param
    Covise::get_string_param("Annotation", &annotation);
    Covise::get_scalar_param("Steps", &steps);

    if (annotation == NULL)
        annotation = annotation_default;
    /*************
      if ( my_min == 0.0 && my_max == 0.0 )
      {
         // use min,max given in param routine already from color editor
         // error = Covise::get_colormap_param("colormap",&min,&max,&ncolor);
      }
      else
      { // use min,max given in user interface of this module
         min = my_min;
         max = my_max;
      }
   **************/
    // an explicit module connection overrides min, max given so far
    MinMax_Name = Covise::get_object_name("minmax");
    if (MinMax_Name == 0L)
    {
        // ignore
    }
    else // get min, max from another module
    {
        minmax_data = new coDoFloat(MinMax_Name);
        if ((minmax_data == NULL) || (!(minmax_data->objectOk())))
        {
#ifndef TOLERANT
            Covise::sendError("ERROR: Data object 'Data' can't be accessed in shared memory");
#else
            Covise::send_stop_pipeline();
#endif
            return;
        }
        minmax_data->getAddress(&s);
        min = s[0];
        max = s[1];
    }

    if (steps > 0)
    {
        // transform map into color-ramps

        float map[4][COLORMAP_WIDTH];
        for (i = 0; i < 4; i++)
        {
            for (j = 0; j < COLORMAP_WIDTH; j++)
            {
                restore[i][j] = map[i][j] = colormap[i][j];
            }
        }
        // interpolate colormap
        double delta = 1.0 / (steps - 1) * (ncolor - 1);

        for (i = 0; i < steps - 1; i++)
        {
            double x = i * delta;
            int idx = (int)x;
            double d = x - idx;
            colormap[0][i] = (1 - d) * map[0][idx] + d * map[0][idx + 1];
            colormap[1][i] = (1 - d) * map[1][idx] + d * map[1][idx + 1];
            colormap[2][i] = (1 - d) * map[2][idx] + d * map[2][idx + 1];
            colormap[3][i] = (1 - d) * map[3][idx] + d * map[3][idx + 1];
            //colormap[i][4] = -1;
        }
        colormap[0][steps - 1] = map[0][ncolor - 1];
        colormap[1][steps - 1] = map[1][ncolor - 1];
        colormap[2][steps - 1] = map[2][ncolor - 1];
        colormap[3][steps - 1] = map[3][ncolor - 1];
        //colormap[steps-1][4] = -1;
    }

    // generate pixel-image
    image = new char[ncolor * 4];
    for (i = 0; i < ncolor; i++)
    {
        image[i * 4] = (unsigned char)(255.0 * colormap[0][i] + 0.5);
        image[i * 4 + 1] = (unsigned char)(255.0 * colormap[1][i] + 0.5);
        image[i * 4 + 2] = (unsigned char)(255.0 * colormap[2][i] + 0.5);
        image[i * 4 + 3] = (unsigned char)(255.0 * colormap[3][i] + 0.5);
    }

    //	get input data object names
    Data = Covise::get_object_name("Data");
    if (Data == 0L)
    {
        Covise::sendError("ERROR: Object name not correct for 'Data'");
        return;
    }

    //	get output data object	names
    Colors = Covise::get_object_name("Colors");
    if (Colors == 0L)
    {
        Covise::sendError("ERROR: Object name not correct for 'Colors'");
        return;
    }

    //	retrieve object from shared memeory
    tmp_obj = new coDistributedObject(Data);
    data_obj = tmp_obj->createUnknown();
    handle_objects(tmp_obj->createUnknown(), Colors);

    if (steps > 0)
    {
        // undo interpolation
        for (i = 0; i < 4; i++)
        {
            for (j = 0; j < COLORMAP_WIDTH; j++)
            {
                colormap[i][j] = restore[i][j];
            }
        }
    }

    // free
    delete[] image;
    delete tmp_obj;
}

void Application::handle_objects(coDistributedObject *data_obj, char *Outname, coDistributedObject **set_out)
{
    int i;
    int setFlag;
    int num;

    char buf[500];
    char colormap_buf[255];
    char *img_name;
    char **attr;
    char **setting;

    coDoSet *D_set;
    coDistributedObject **set_objs;
    int set_num_elem;
    coDistributedObject *const *data_objs;

    int *vertex_list;

    depth++; // recurrence depth

    if (data_obj != 0L)
    {
        // prepare colormap attribute for attachment
        strcpy(colormap_as_an_attribute, "");
        sprintf(colormap_buf, "%s\n%s\n%g\n%g\n%ld\n%d", Outname, annotation, min, max, (steps > 0) ? steps : ncolor, 0);
        strcat(colormap_as_an_attribute, colormap_buf);

        for (i = 0; i < ncolor; i++)
        {
            sprintf(colormap_buf, "\n%f\n%f\n%f", colormap[0][i], colormap[1][i], colormap[2][i]);
            strcat(colormap_as_an_attribute, colormap_buf);
        }

        // get all attributes
        num = data_obj->get_all_attributes(&attr, &setting);

        dataType = data_obj->getType();
        if (strcmp(dataType, "USTSDT") == 0)
        {
            u_data = (coDoFloat *)data_obj;
            npoint = u_data->getNumPoints();
            u_data->getAddress(&scalar);
        }

        else if (strcmp(dataType, "SETELE") == 0)
        {
            if (min == max)
                Covise::sendWarning("Min==Max in Set: no automatic setting!");

            data = (coDoSet *)data_obj;
            data_objs = ((coDoSet *)data_obj)->getAllElements(&set_num_elem);
            set_objs = new coDistributedObject *[set_num_elem + 1];
            set_objs[0] = NULL;
            if (depth == 1)
                setFlag = 0;
            else
                setFlag = 1;

            for (i = 0; i < set_num_elem; i++)
            {
                sprintf(buf, "%s_%d", Outname, i);
                handle_objects(data_objs[i], buf, set_objs);
            }

            D_set = new coDoSet(Outname, set_objs);
            // setting attributes
            if (num > 0)
                D_set->addAttributes(num, attr, setting);

            if (setFlag == 0)
            {
                D_set->addAttribute("COLORMAP", colormap_as_an_attribute);
            }

            if (set_out)
            {
                for (i = 0; set_out[i]; i++)
                    ;
                set_out[i] = D_set;
                set_out[i + 1] = NULL;
            }
            else
                delete D_set;
            delete ((coDoSet *)data_obj);
            for (i = 0; set_objs[i]; i++)
                delete set_objs[i];
            delete[] set_objs;
            return;
        }

        else
        {
            Covise::sendError("ERROR: Data object 'Data' has wrong data type");
            return;
        }
    }
    else
    {
#ifndef TOLERANT
        Covise::sendError("ERROR: Data object 'Data' can't be accessed in shared memory");
#else
        Covise::send_stop_pipeline();
#endif
        return;
    }

    /////////////////////////////////////////////////////////////////
    //      generate the output data objects

    if (map == 1 || npoint == 0) // rgba colors : always use rgba for empty obj
    {

        if (strcmp(dataType, "STRSDT") == 0)
        {
            p_colors = new coDoRGBA(Outname, i_dim * j_dim * k_dim);
            if (!p_colors->objectOk())
            {
                Covise::sendError("ERROR: creation of data object 'Colors' failed");
                return;
            }
            p_colors->getAddress(&pc);

            make_rgba_colors(scalar, npoint, (steps > 0) ? steps : ncolor, pc,
                             &colormap[0][0], &colormap[1][0],
                             &colormap[2][0], &colormap[3][0], min, max);

            // setting attributes
            if (num > 0)
                p_colors->addAttributes(num, attr, setting);
            if (depth == 1)
                p_colors->addAttribute("COLORMAP", colormap_as_an_attribute);
            delete s_data;
        }

        else if (strcmp(dataType, "USTSDT") == 0)
        {
            p_colors = new coDoRGBA(Outname, npoint);
            if (!p_colors->objectOk())
            {
                Covise::sendError("ERROR: creation of data object 'Colors' failed");
                return;
            }
            p_colors->getAddress(&pc);

            make_rgba_colors(scalar, npoint, (steps > 0) ? steps : ncolor, pc,
                             &colormap[0][0], &colormap[1][0],
                             &colormap[2][0], &colormap[3][0], min, max);

            // setting attributes
            if (num > 0)
                p_colors->addAttributes(num, attr, setting);
            if (depth == 1)
                p_colors->addAttribute("COLORMAP", colormap_as_an_attribute);
            delete u_data;
        }
    }

    else if (map == 2) // texture coordinates
    {
        // simple vertex_list
        vertex_list = new int[npoint];
        for (i = 0; i < npoint; i++)
            vertex_list[i] = i;

        // allocate tex_coords
        tex_coords = new float *[2];
        for (i = 0; i < 2; i++)
            tex_coords[i] = new float[npoint];
        // initialize y-component of texture coordinate
        for (i = 0; i < npoint; i++)
            tex_coords[1][i] = 0.0;

        make_texCoord(scalar, npoint, ncolor, min, max);

        img_name = new char[strlen(Outname) + 5];
        strcpy(img_name, Outname);
        strcat(img_name, "_Img");

        colorLUT = new coDoPixelImage(img_name, ncolor, 1,
                                      PIXEL_SIZE, PIXEL_FORMAT, image);

        p_texture = new coDoTexture(Outname, colorLUT, 0, 4, TEXTURE_LEVEL,
                                    npoint, vertex_list, npoint, tex_coords);

        // setting attributes
        if (num > 0)
            p_texture->addAttributes(num, attr, setting);
        // set the colormap attribute
        if (depth == 1)
            p_texture->addAttribute("COLORMAP", colormap_as_an_attribute);

        //free
        delete[] vertex_list;
        delete[] tex_coords[0];
        delete[] tex_coords[1];
        delete[] tex_coords;
    }

    //      add objects to set
    if (set_out)
    {
        for (i = 0; set_out[i]; i++)
            ;

        if (map == 1)
            set_out[i] = p_colors;
        else if (map == 2)
            set_out[i] = p_texture;
        set_out[i + 1] = NULL;
    }
    else
    {
        //delete p_colors;
        //delete p_texture;
        //delete colorLUT;
    }
}

//====================================================================
// Linear interpolation between to indices in a rgb array
//
//   colmap    - pointer to the first element of the colormap.
//   color     - modified color, [0,1,2] for r/g/b.
//   first     - start index
//   last      - stop index
//   first_val - value of the start index, [0.0 .. 1.0].
//   last_val  - value of the stop index, [0.0 .. 1.0].
//=======================================================================
void Application::interpolate_colmap(float (*colmap)[3], int color,
                                     int first, int last,
                                     float first_val, float last_val)
{
    int i;
    float delta;
    float mem; /* The previous value. */

    delta = (last_val - first_val) / (last - first);
    (*(colmap + first))[color] = mem = first_val;
    for (i = first + 1; i <= last; i++)
        (*(colmap + i))[color] = mem += delta;
}

//==================================================================
// Print the current color map  to stdout.
// Inputs:
//   colmap - colormap of any length.
//   first  - index of the first printed element.
//   last   - index of the last printed element.
//===================================================================
void Application::print_colmap(float (*colmap)[3], int first, int last)
{
    int i;
    float(*p)[3];

    p = colmap + first;
    for (i = first; i <= last; i++, p++)
        fprintf(stdout, " %d %f %f %f \n", i, (*p)[0], (*p)[1], (*p)[2]);
}

//======================================================================
// Interpolate to rgba packed
//======================================================================
int Application::pack_rgba_color(float rf, float gf, float bf, float af)
{

    // convert to long
    // changed from abgr to rgba
    // Uwe Woessner
    unsigned r, g, b, a;
    r = (unsigned)rf * 255;
    g = (unsigned)gf * 255;
    b = (unsigned)bf * 255;
    a = (unsigned)af * 255;
    unsigned int packed = (r << 24) | (g << 16) | (b << 8) | a;
    return packed;
}

//======================================================================
// Generation of the colormap (currently fixed)
//======================================================================
void Application::make_rgba_colors(float *scalar, int np, int ncolor,
                                   int *pc, float *rcol, float *gcol, float *bcol, float *acol,
                                   float par_min, float par_max)
{

    int i;
    float rc, bc, gc, ac;
    int entry = 0;
    float minimum = par_min;
    float maximum = par_max;

    //
    // find min and max in scalar data
    //
    if (par_min == par_max)
    {
        minimum = scalar[0];
        maximum = scalar[0];

        for (i = 0; i < np; i++)
        {
            if (scalar[i] < minimum)
                minimum = scalar[i];
            if (scalar[i] > maximum)
                maximum = scalar[i];
        }

        // set new min/max-values
        min = minimum;
        max = maximum;

        if (min == max)
        {
            for (i = 0; i < np; i++)
            {
                rc = rcol[(int)min % (ncolor - 1)];
                gc = gcol[(int)min % (ncolor - 1)];
                bc = bcol[(int)min % (ncolor - 1)];
                ac = acol[(int)min % (ncolor - 1)];
                pc[i] = pack_rgba_color(rc, gc, bc, ac);
            }
        }
        else
        {
            for (i = 0; i < np; i++)
            {
                entry = (int)((float)(ncolor - 1) * (scalar[i] - min) / (max - min));
                if (entry < 0)
                    entry = 0;
                else if (entry > (ncolor - 1))
                    entry = (ncolor - 1);

                rc = rcol[entry];
                gc = gcol[entry];
                bc = bcol[entry];
                ac = acol[entry];
                pc[i] = pack_rgba_color(rc, gc, bc, ac);
            }
        }
    }
    else
    {

        for (i = 0; i < np; i++)
        {
            entry = (int)((float)(ncolor - 1) * (scalar[i] - par_min) / (par_max - par_min));
            if (entry < 0)
                entry = 0;
            else if (entry > (ncolor - 1))
                entry = (ncolor - 1);

            rc = rcol[entry];
            gc = gcol[entry];
            bc = bcol[entry];
            ac = acol[entry];
            pc[i] = pack_rgba_color(rc, gc, bc, ac);
        }
    }
}

//=======================================================================================
// evaluate texture coordinates
//=======================================================================================

void Application::make_texCoord(float *scalar, int num_points, int num_colors,
                                float min_param, float max_param)
{
    float minimum, maximum;
    float t, entry;

    int i;

    float numerator;
    float s;

    (void)num_colors;

    minimum = min_param;
    maximum = max_param;

    //////
    ////// get min/max if not given
    //////

    if (minimum == maximum)
    {
        // we assume that no min/max-values are given, so compute them ourselves

        minimum = scalar[0];
        maximum = scalar[0];

        for (i = 0; i < num_points; i++)
        {
            if (scalar[i] > maximum)
                maximum = scalar[i];
            else if (scalar[i] < minimum)
                minimum = scalar[i];
        }

        // set new min/max-values
        min = minimum;
        max = maximum;
    }

    //////
    ////// compute
    //////

    if (min == max)
    {
        t = ((int)min % 255) / 255;

        for (i = 0; i < num_points; i++)
        {
            tex_coords[0][i] = t;
        }
    }
    else
    {
        numerator = max - min;

        s = 1 / numerator;

        for (i = 0; i < num_points; i++)
        {
            entry = s * (scalar[i] - min);
            if (entry < 0.0)
                entry = 0.0;
            else if (entry > 1.0)
                entry = 1.0;

            tex_coords[0][i] = entry;
        }
    }

    // done
    return;
}
