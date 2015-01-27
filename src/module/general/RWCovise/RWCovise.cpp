/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                    (C) 2000 VirCinity  **
 **                                                                        **
 **                                                                        **
 ** Description: Read/Write module for COVISE data     	                   **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.11.95  V1.0 						                           **
 ** Modified: 06.12.2000 B.Teplitski:                                      **
 **     working with datatype coDoTexture                                   **
\**************************************************************************/
#include <config/CoviseConfig.h>
#include "RWCovise.h"

RWCovise::RWCovise(int argc, char *argv[])
    : coModule(argc, argv, "Read OR Write COVISE Data")
    , _fd(0)
    , _number_of_elements(0)
    , _trueOpen(true)
{
    p_mesh_in =

        addInputPort("mesh_in", "UniformGrid|Text|Points|Spheres|UnstructuredGrid|RectilinearGrid|StructuredGrid|Tensor|Float|Vec2|Vec3|Polygons|TriangleStrips|Geometry|Lines|PixelImage|Texture|IntArr|RGBA|USR_DistFenflossBoco|Int|OctTree|OctTreeP", "mesh_in");

    p_mesh_in->setRequired(0);

    p_mesh =

        addOutputPort("mesh", "UniformGrid|Text|Points|Spheres|UnstructuredGrid|RectilinearGrid|StructuredGrid|Tensor|Float|Vec2|Vec3|Polygons|TriangleStrips|Geometry|Lines|PixelImage|Texture|IntArr|RGBA|USR_DistFenflossBoco|Int|OctTree|OctTreeP", "mesh");

    p_grid_path = addFileBrowserParam("grid_path", "File path");
    p_grid_path->setValue(".", "*.covise/*");
    _p_force = addBooleanParam("forceReading", "Force reading (don't whine if COVISE crashes)");
    _p_force->setValue(0);

    _p_firstStep = addInt32Param("firstStepNo", "first Step Nr.");
    _p_firstStep->setValue(0);

    _p_numSteps = addInt32Param("numSteps", "Number of steps to read (0 reads all)");
    _p_numSteps->setValue(0);

    _p_skipStep = addInt32Param("skipSteps", "number of steps to skip between timesteps");
    _p_skipStep->setValue(0);

    _p_step = addInt32Param("stepNo", "stepNo");
    _p_step->setValue(0);

    _p_rotate = addBooleanParam("rotate_output", "Rotate output");
    _p_rotate->setValue(0);

    _p_RotAxis = addChoiceParam("rotation_axis", "Rotation axis");
    s_RotAxis[0] = strdup("x");
    s_RotAxis[1] = strdup("y");
    s_RotAxis[2] = strdup("z");

    _p_RotAxis->setValue(3, s_RotAxis, 2);

    _p_rot_speed = addFloatParam("rot_speed", "Rotation speed");
    _p_rot_speed->setValue(2.);

    _p_increment_suffix = addBooleanParam("increment_filename", "use this to add a suffix to the filename which is incremented every time the module is executed");
    _p_increment_suffix->setValue(0);

    suffix_number = 0;
}

RWCovise::~RWCovise()
{
}

void RWCovise::postInst()
{
}

void RWCovise::param(const char *name, bool /*inMapLoading*/)
{

    if (strcmp(name, p_grid_path->getName()) == 0)
    {
        grid_Path = p_grid_path->getValue();

        // module title is the filename without path and '.covise'
        std::string title("RW:");
#ifdef _WIN32
        size_t baseoff = grid_Path.find_last_of("\\/");
#else
        size_t baseoff = grid_Path.find_last_of("/");
#endif
        if (baseoff == grid_Path.length() - 1)
            title += "(no file)";
        else
        {
            if (baseoff == std::string::npos)
                baseoff = 0;
            else
                ++baseoff;
            const size_t end = grid_Path.rfind(".covise");
            if (end == std::string::npos)
                title += grid_Path.substr(baseoff);
            else
                title += grid_Path.substr(baseoff, end - baseoff);
        }

        setTitle(title.c_str());

        suffix_number = 0;
    }
    if (strcmp(name, "increment_filename") == 0)
    {
        if (_p_increment_suffix->getValue() == 0)
        {
            suffix_number = 0;
        }
    }
}

int RWCovise::compute(const char *)
{
    if (!coCoviseConfig::getEntry("Module.RWCovise.NoMagic").empty())
        useMagic = 0;
    else
        useMagic = 1;

    grid_Path = p_grid_path->getValue();

    if (p_mesh_in->getCurrentObject() != NULL)
    {
        _trueOpen = true;
        if (_p_increment_suffix->getValue())
        {
            const char *suffix = strrchr(grid_Path.c_str(), '.');
            char *outfile;
            if (suffix != NULL)
            {
                int baselen = strlen(grid_Path.c_str()) - strlen(suffix);
                outfile = new char[baselen + 1 + 4 + strlen(suffix)];

                snprintf(outfile, baselen + 1, "%s", grid_Path.c_str());
                sprintf(outfile, "%s_%03d%s", outfile, suffix_number, suffix);
            }
            else
            {
                outfile = new char[strlen(grid_Path.c_str()) + 5];
                sprintf(outfile, "%s_%03d", grid_Path.c_str(), suffix_number);
            }

            WriteFile(outfile, p_mesh_in->getCurrentObject());
            suffix_number++;

            delete[] outfile;
        }
        else
        {
            WriteFile(grid_Path.c_str(), p_mesh_in->getCurrentObject());
        }
    }
    else
    {
        if (_p_step->getValue() < 0)
        {
            _p_step->setValue(0);
            _trueOpen = true;
        }
        if (_p_firstStep->getValue() < 0)
        {
            _p_firstStep->setValue(0);
            _trueOpen = true;
        }

        else if (_p_step->getValue() > 0 && _trueOpen)
        {
            // check if we have to do with a set or not
            int fd = ::covOpenInFile(grid_Path.c_str());
            char Data_Type[7];
            if (!fd)
            {
                sendError("Failed to open '%s' for reading: %s", grid_Path.c_str(), strerror(errno));
                return STOP_PIPELINE;
            }
            if (::covReadDescription(fd, Data_Type) == -1 && !_p_force->getValue())
            {
                Covise::sendError("ERROR: probably not a COVISE file");
                Covise::sendError("ERROR: try RWCovsiseASCII if it's an ASCII COVISE file");
                return STOP_PIPELINE;
            }
            Data_Type[6] = '\0';
            _fd = fd;
            _trueOpen = false;
            // if it is a set, look for the number of elements
            if (strcmp(Data_Type, "SETELE") == 0)
            {
                ::covReadSetBegin(fd, &_number_of_elements);
                // and correct _p_step if necessary
                if (_p_step->getValue() > _number_of_elements)
                {
                    _p_step->setValue(_number_of_elements);
                }
            }
            else
            {
                _number_of_elements = 0;
            }
        }
        else if (_p_step->getValue() == 0) // no pipelinecollect
        {
            _trueOpen = true;
        }

        coDistributedObject *obj = ReadFile(grid_Path.c_str(), p_mesh->getObjName(), _p_force->getValue(), _p_firstStep->getValue(), _p_numSteps->getValue(), _p_skipStep->getValue());

        if (_p_step->getValue() > 0 && _number_of_elements > 0)
        {
            // add in this case pertinent attributes for pipelinecollect
            string module_id("!"); // just for compatibility
            module_id += string(Covise::get_module()) + string("\n") + Covise::get_instance() + string("\n");
            module_id += string(Covise::get_host()) + string("\n");
            obj->addAttribute("BLOCK_FEEDBACK", module_id.c_str());

            obj->addAttribute("NEXT_STEP_PARAM", "stepNo\nScalar\n1\n");
            // increase stepNo by 1
            char stepNr[32];
            sprintf(stepNr, "%ld", _p_step->getValue() + 1);
            obj->addAttribute("NEXT_STEP", stepNr);

            // show last elem
            if (_p_step->getValue() + 1 > _number_of_elements)
            {
                // we have to read attributes!!! FIXME
                if (1)
                {
                    obj->addAttribute("LAST_STEP", " ");
                }
                else
                {
                    obj->addAttribute("LAST_BLOCK", " ");
                }
                // if this is the last step for pipelinecollect reset _trueOpen and close
                _trueOpen = true;
                this->covCloseInFile(_fd);
            }
        }

        if (_p_rotate->getValue() && obj)
        {
            char axis_string[20];
            char buf[30];
            int x = 0, y = 0, z = 0;
            int rotaxis = _p_RotAxis->getValue();
            if (!strcmp(s_RotAxis[rotaxis], "x"))
            {
                x = 1;
            }
            if (!strcmp(s_RotAxis[rotaxis], "y"))
            {
                y = 1;
            }
            if (!strcmp(s_RotAxis[rotaxis], "z"))
            {
                z = 1;
            }
            obj->addAttribute("ROTATE_POINT", "0 0 0");

            sprintf(axis_string, "%d %d %d\n", x, y, z);
            obj->addAttribute("ROTATE_VECTOR", axis_string);

            sprintf(buf, "%f", (_p_rot_speed->getValue()));
            obj->addAttribute("ROTATE_SPEED", buf);
        }

        p_mesh->setCurrentObject(obj);
    }

    return CONTINUE_PIPELINE;
}

int
RWCovise::covOpenInFile(const char *grid_Path)
{
    if (_trueOpen)
    {
        _fd = CoviseIO::covOpenInFile(grid_Path);
    }
    return _fd;
}

int
RWCovise::covCloseInFile(int fd)
{
    if (_trueOpen)
    {
        return CoviseIO::covCloseInFile(fd);
        _fd = 0;
    }
    return 1;
}

MODULE_MAIN(IO, RWCovise)
