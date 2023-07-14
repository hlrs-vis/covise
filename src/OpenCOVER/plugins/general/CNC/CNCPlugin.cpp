/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\
**                                                            (C)2023 HLRS  **
**                                                                          **
** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
**    Visualises path and workpiece of CNC machining                        **
**                                                                          **
** Author: U.Woessner, A.Kaiser		                                        **
**                                                                          **
** History:  								                                **
** April-05  v1	    				       		                            **
** April-23  v2                                                             **
**                                                                          **
\****************************************************************************/

#include "CNCPlugin.h"
//#include "CNCTree.h"
#define USE_MATH_DEFINES
#include <math.h>
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include <cover/RenderObject.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/LineWidth>
#include <osg/PolygonMode>

#include <osg/LineSegment>
#include <osg/Matrix>
#include <osg/Vec3>
#include <cover/coVRAnimationManager.h>
#include <osgUtil/SmoothingVisitor>
using namespace osg;
using namespace osgUtil;
/************************************************************************

       Copyright 2008 Mark Pictor

   This file is part of RS274NGC.

   RS274NGC is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   RS274NGC is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with RS274NGC.  If not, see <http://www.gnu.org/licenses/>.

   This software is based on software that was produced by the National
   Institute of Standards and Technology (NIST).

   ************************************************************************/

#include "rs274ngc.hh"
#include "rs274ngc_return.hh"
#include <stdio.h> /* gets, etc. */
#include <stdlib.h> /* exit       */
#include <string.h> /* strcpy     */

extern CANON_TOOL_TABLE _tools[]; /* in canon.cc */
extern int _tool_max; /* in canon.cc */
extern char _parameter_file_name[]; /* in canon.cc */

FILE *_outfile; /* where to print, set in main */

/*

   This file contains the source code for an emulation of using the six-axis
   rs274 interpreter from the EMC system.

   */

/*********************************************************************/

/* report_error

   Returned Value: none

   Side effects: an error message is printed on stderr

   Called by:
   interpret_from_file
   interpret_from_keyboard
   main

   This

   1. calls rs274ngc_error_text to get the text of the error message whose
   code is error_code and prints the message,

   2. calls rs274ngc_line_text to get the text of the line on which the
   error occurred and prints the text, and

   3. if print_stack is on, repeatedly calls rs274ngc_stack_name to get
   the names of the functions on the function call stack and prints the
   names. The first function named is the one that sent the error
   message.

   */

void report_error(/* ARGUMENTS                            */
                  int error_code, /* the code number of the error message */
                  int print_stack) /* print stack if ON, otherwise not     */
{
    char buffer[RS274NGC_TEXT_SIZE];
    int k;

    rs274ngc_error_text(error_code, buffer, 5); /* for coverage of code */
    rs274ngc_error_text(error_code, buffer, RS274NGC_TEXT_SIZE);
    fprintf(stderr, "%s\n",
            ((buffer[0] IS 0) ? "Unknown error, bad error code" : buffer));
    rs274ngc_line_text(buffer, RS274NGC_TEXT_SIZE);
    fprintf(stderr, "%s\n", buffer);
    if (print_stack IS RS_ON)
    {
        for (k SET_TO 0;; k++)
        {
            rs274ngc_stack_name(k, buffer, RS274NGC_TEXT_SIZE);
            if (buffer[0] ISNT 0)
                fprintf(stderr, "%s\n", buffer);
            else
                break;
        }
    }
}

/***********************************************************************/

/* interpret_from_keyboard

   Returned Value: int (0)

   Side effects:
   Lines of NC code entered by the user are interpreted.

   Called by:
   interpret_from_file
   main

   This prompts the user to enter a line of rs274 code. When the user
   hits <enter> at the end of the line, the line is executed.
   Then the user is prompted to enter another line.

   Any canonical commands resulting from executing the line are printed
   on the monitor (stdout).  If there is an error in reading or executing
   the line, an error message is printed on the monitor (stderr).

   To exit, the user must enter "quit" (followed by a carriage return).

   */

int interpret_from_keyboard(/* ARGUMENTS                 */
                            int block_delete, /* switch which is ON or OFF */
                            int print_stack) /* option which is ON or OFF */
{
    char line[RS274NGC_TEXT_SIZE];
    int status;

    for (;;)
    {
        printf("READ => ");
        fgets(line, sizeof(line), stdin);
        if (strcmp(line, "quit") IS 0)
            return 0;
        status SET_TO rs274ngc_read(line);
        if ((status IS RS274NGC_EXECUTE_FINISH)AND(block_delete IS RS_ON))
            ;
        else if (status IS RS274NGC_ENDFILE)
            ;
        else if ((status ISNT RS274NGC_EXECUTE_FINISH)AND(status ISNT RS274NGC_OK))
            report_error(status, print_stack);
        else
        {
            status SET_TO rs274ngc_execute();
            if ((status IS RS274NGC_EXIT)OR(status IS RS274NGC_EXECUTE_FINISH))
                ;
            else if (status ISNT RS274NGC_OK)
                report_error(status, print_stack);
        }
    }
}

/*********************************************************************/

/* interpret_from_file

   Returned Value: int (0 or 1)
   If any of the following errors occur, this returns 1.
   Otherwise, it returns 0.
   1. rs274ngc_read returns something other than RS274NGC_OK or
   RS274NGC_EXECUTE_FINISH, no_stop is off, and the user elects
   not to continue.
   2. rs274ngc_execute returns something other than RS274NGC_OK,
   EXIT, or RS274NGC_EXECUTE_FINISH, no_stop is off, and the user
   elects not to continue.

   Side Effects:
   An open NC-program file is interpreted.

   Called By:
   main

   This emulates the way the EMC system uses the interpreter.

   If the do_next argument is 1, this goes into MDI mode if an error is
   found. In that mode, the user may (1) enter code or (2) enter "quit" to
   get out of MDI. Once out of MDI, this asks the user whether to continue
   interpreting the file.

   If the do_next argument is 0, an error does not stop interpretation.

   If the do_next argument is 2, an error stops interpretation.

   */

int interpret_from_file(/* ARGUMENTS                  */
                        int do_next, /* what to do if error        */
                        int block_delete, /* switch which is ON or OFF  */
                        int print_stack) /* option which is ON or OFF  */
{
    int status;
    char line[RS274NGC_TEXT_SIZE];

    for (;;)
    {
        status SET_TO rs274ngc_read(NULL);
        if ((status IS RS274NGC_EXECUTE_FINISH)AND(block_delete IS RS_ON))
            continue;
        else if (status IS RS274NGC_ENDFILE)
            break;
        if ((status ISNT RS274NGC_OK)AND // should not be EXIT
            (status ISNT RS274NGC_EXECUTE_FINISH))
        {
            report_error(status, print_stack);
            if ((status IS NCE_FILE_ENDED_WITH_NO_PERCENT_SIGN)OR(do_next IS 2)) /* 2 means stop */
            {
                status SET_TO 1;
                break;
            }
            else if (do_next IS 1) /* 1 means MDI */
            {
                fprintf(stderr, "starting MDI\n");
                interpret_from_keyboard(block_delete, print_stack);
                fprintf(stderr, "continue program? y/n =>");
                fgets(line, sizeof(line), stdin);
                if (line[0] ISNT 'y')
                {
                    status SET_TO 1;
                    break;
                }
                else
                    continue;
            }
            else /* if do_next IS 0 -- 0 means continue */
                continue;
        }
        status SET_TO rs274ngc_execute();
        if ((status ISNT RS274NGC_OK)AND(status ISNT RS274NGC_EXIT) AND(status ISNT RS274NGC_EXECUTE_FINISH))
        {
            report_error(status, print_stack);
            status SET_TO 1;
            if (do_next IS 1) /* 1 means MDI */
            {
                fprintf(stderr, "starting MDI\n");
                interpret_from_keyboard(block_delete, print_stack);
                fprintf(stderr, "continue program? y/n =>");
                fgets(line, sizeof(line), stdin);
                if (line[0] ISNT 'y')
                    break;
            }
            else if (do_next IS 2) /* 2 means stop */
                break;
        }
        else if (status IS RS274NGC_EXIT)
            break;
    }
    return ((status IS 1) ? 1 : 0);
}

/************************************************************************/

/* read_tool_file

   Returned Value: int
   If any of the following errors occur, this returns 1.
   Otherwise, it returns 0.
   1. The file named by the user cannot be opened.
   2. No blank line is found.
   3. A line of data cannot be read.
   4. A tool slot number is less than 1 or >= _tool_max

   Side Effects:
   Values in the tool table of the machine setup are changed,
   as specified in the file.

   Called By: main

   Tool File Format
   -----------------
   Everything above the first blank line is read and ignored, so any sort
   of header material may be used.

   Everything after the first blank line should be data. Each line of
   data should have four or more items separated by white space. The four
   required items are slot, tool id, tool length offset, and tool diameter.
   Other items might be the holder id and tool description, but these are
   optional and will not be read. Here is a sample line:

   20  1419  4.299  1.0   1 inch carbide end mill

   The tool_table is indexed by slot number.

   */

int read_tool_file(/* ARGUMENTS         */
                   const char *file_name) /* name of tool file */
{
    FILE *tool_file_port;
    char buffer[1000];
    int slot;
    int tool_id;
    double offset;
    double diameter;

    if (file_name[0] IS 0) /* ask for name if given name is empty string */
    {
        fprintf(stderr, "name of tool file => ");
        fgets(buffer, sizeof(buffer), stdin);
        tool_file_port SET_TO fopen(buffer, "r");
    }
    else
        tool_file_port SET_TO fopen(file_name, "r");
    if (tool_file_port IS NULL)
    {
        fprintf(stderr, "Cannot open %s\n",
                ((file_name[0] IS 0) ? buffer : file_name));
        return 1;
    }
    for (;;) /* read and discard header, checking for blank line */
    {
        if (fgets(buffer, 1000, tool_file_port) IS NULL)
        {
            fprintf(stderr, "Bad tool file format\n");
            return 1;
        }
        else if (buffer[0] IS '\n')
            break;
    }

    for (slot SET_TO 0; slot <= _tool_max; slot++) /* initialize */
    {
        _tools[slot].id SET_TO - 1;
        _tools[slot].length SET_TO 0;
        _tools[slot].diameter SET_TO 0;
    }
    for (; (fgets(buffer, 1000, tool_file_port) ISNT NULL);)
    {
        if (sscanf(buffer, "%d %d %lf %lf", &slot,
                   &tool_id, &offset, &diameter) < 4)
        {
            fprintf(stderr, "Bad input line \"%s\" in tool file\n", buffer);
            return 1;
        }
        if ((slot < 0)OR(slot > _tool_max)) /* zero and max both OK */
        {
            fprintf(stderr, "Out of range tool slot number %d\n", slot);
            return 1;
        }
        _tools[slot].id SET_TO tool_id;
        _tools[slot].length SET_TO offset;
        _tools[slot].diameter SET_TO diameter;
    }
    fclose(tool_file_port);
    return 0;
}

/************************************************************************/

/* designate_parameter_file

   Returned Value: int
   If any of the following errors occur, this returns 1.
   Otherwise, it returns 0.
   1. The file named by the user cannot be opened.

   Side Effects:
   The name of a parameter file given by the user is put in the
   file_name string.

   Called By: main

   */

int designate_parameter_file(char *file_name, size_t allocated_size)
{
    FILE *test_port;

    fprintf(stderr, "name of parameter file => ");
    fgets(file_name, allocated_size, stdin);
    test_port SET_TO fopen(file_name, "r");
    if (test_port IS NULL)
    {
        fprintf(stderr, "Cannot open %s\n", file_name);
        return 1;
    }
    fclose(test_port);
    return 0;
}

/************************************************************************/

/* adjust_error_handling

   Returned Value: int (0)

   Side Effects:
   The values of print_stack and do_next are set.

   Called By: main

   This function allows the user to set one or two aspects of error handling.

   By default the driver does not print the function stack in case of error.
   This function always allows the user to turn stack printing on if it is off
   or to turn stack printing off if it is on.

   When interpreting from the keyboard, the driver always goes ahead if there
   is an error.

   When interpreting from a file, the default behavior is to stop in case of
   an error. If the user is interpreting from a file (indicated by args being
   2 or 3), this lets the user change what it does on an error.

   If the user has not asked for output to a file (indicated by args being 2),
   the user can choose any of three behaviors in case of an error (1) continue,
   (2) stop, (3) go into MDI mode. This function allows the user to cycle among
   the three.

   If the user has asked for output to a file (indicated by args being 3),
   the user can choose any of two behaviors in case of an error (1) continue,
   (2) stop. This function allows the user to toggle between the two.

   */

int adjust_error_handling(
    int args,
    int *print_stack,
    int *do_next)
{
    char buffer[80];
    int choice;

    for (;;)
    {
        fprintf(stderr, "enter a number:\n");
        fprintf(stderr, "1 = done with error handling\n");
        fprintf(stderr, "2 = %sprint stack on error\n",
                ((*print_stack IS RS_ON) ? "do not " : ""));
        if (args IS 3)
        {
            if (*do_next IS 0) /* 0 means continue */
                fprintf(stderr,
                        "3 = stop on error (do not continue)\n");
            else /* if do_next IS 2 -- 2 means stopping on error */
                fprintf(stderr,
                        "3 = continue on error (do not stop)\n");
        }
        else if (args IS 2)
        {
            if (*do_next IS 0) /* 0 means continue */
                fprintf(stderr,
                        "3 = mdi on error (do not continue or stop)\n");
            else if (*do_next IS 1) /* 1 means MDI */
                fprintf(stderr,
                        "3 = stop on error (do not mdi or continue)\n");
            else /* if do_next IS 2 -- 2 means stopping on error */
                fprintf(stderr,
                        "3 = continue on error (do not stop or mdi)\n");
        }
        fprintf(stderr, "enter choice => ");
        fgets(buffer, sizeof(buffer), stdin);
        if (sscanf(buffer, "%d", &choice) ISNT 1)
            continue;
        if (choice IS 1)
            break;
        else if (choice IS 2)
            *print_stack SET_TO((*print_stack IS RS_OFF) ? RS_ON : RS_OFF);
        else if ((choice IS 3)AND(args IS 3))
            *do_next SET_TO((*do_next IS 0) ? 2 : 0);
        else if ((choice IS 3)AND(args IS 2))
            *do_next SET_TO((*do_next IS 2) ? 0 : (*do_next + 1));
    }
    return 0;
}

CNCPlugin *CNCPlugin::thePlugin = NULL;

CNCPlugin *CNCPlugin::instance()
{
    if (!thePlugin)
        thePlugin = new CNCPlugin();
    return thePlugin;
}

CNCPlugin::CNCPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("CNCPlugin", cover->ui)
, PathTab(new ui::Menu("CNC", this))
, record(new ui::Button(PathTab, "Record"))
, playPause(new ui::Button(PathTab, "Play"))
, reset(new ui::Action(PathTab, "Reset"))
, saveButton(new ui::Action(PathTab, "Save"))
, colorMap(*PathTab)
, viewDirections(new ui::Button(PathTab, "ViewingDirections"))
, viewlookAt(new ui::Button(PathTab, "ViewTarget"))
, lengthEdit(new ui::EditField(PathTab, "Length"))
, radiusEdit(new ui::EditField(PathTab, "Radius"))
, renderMethod(new ui::SelectionList(PathTab, "RenderMethod"))
, recordRateTUI(new ui::EditField(PathTab, "Fps"))
, numSamples(new ui::Label(PathTab, "numSamples"))
, fileNameBrowser(new ui::FileBrowser(PathTab, "File", true))
, viewPath(new ui::Button(PathTab, "ViewPath"))
{
    thePlugin = this;
}

static FileHandler handlers[] = {
    { NULL,
      CNCPlugin::sloadGCode,
      CNCPlugin::unloadGCode,
      "gcode" }
};

int CNCPlugin::sloadGCode(const char *filename, osg::Group *loadParent, const char *)
{

    instance()->loadGCode(filename, loadParent);
    return 0;
}

int CNCPlugin::loadGCode(const char *filename, osg::Group *loadParent)
{

    frameNumber = 0;
    //delete[] positions;
    //positions = new float [3*MAXSAMPLES+3];

    geode = new Geode();
    geom = new Geometry();
    geode->setStateSet(geoState.get());

    geom->setColorBinding(Geometry::BIND_OFF);

    geode->addDrawable(geom.get());
    geode->setName("Viewer Positions");

    // set up geometry
    vert = new osg::Vec3Array;
    color = new osg::Vec4Array;

    int status;
    int do_next; /* 0=continue, 1=mdi, 2=stop */
    int block_delete;
    char buffer[80];
    int tool_flag;
    int gees[RS274NGC_ACTIVE_G_CODES];
    int ems[RS274NGC_ACTIVE_M_CODES];
    double sets[RS274NGC_ACTIVE_SETTINGS];
    char default_name[] SET_TO "rs274ngc.var";
    int print_stack;

    do_next SET_TO 2; /* 2=stop */
    block_delete SET_TO RS_OFF;
    print_stack SET_TO RS_OFF;
    tool_flag SET_TO 0;
    
    const char *varFileName = opencover::coVRFileManager::instance()->getName("share/covise/rs274ngc.var");
    strcpy(_parameter_file_name, varFileName);
    _outfile SET_TO stdout; /* may be reset below */

    fprintf(stderr, "executing\n");
    if (tool_flag IS 0)
    {
        const char *toolFileName = opencover::coVRFileManager::instance()->getName("share/covise/rs274ngc.tool_default");

        if (read_tool_file(toolFileName) ISNT 0)
            exit(1);
    }

    if ((status SET_TO rs274ngc_init())ISNT RS274NGC_OK)
    {
        report_error(status, print_stack);
        exit(1);
    }

    status SET_TO rs274ngc_open(filename);
    if (status ISNT RS274NGC_OK) /* do not need to close since not open */
    {
        report_error(status, print_stack);
        exit(1);
    }
    status SET_TO interpret_from_file(do_next, block_delete, print_stack);
    rs274ngc_file_name(buffer, 5); /* called to exercise the function */
    rs274ngc_file_name(buffer, 79); /* called to exercise the function */
    rs274ngc_close();
    rs274ngc_line_length(); /* called to exercise the function */
    rs274ngc_sequence_number(); /* called to exercise the function */
    rs274ngc_active_g_codes(gees); /* called to exercise the function */
    rs274ngc_active_m_codes(ems); /* called to exercise the function */
    rs274ngc_active_settings(sets); /* called to exercise the function */
    rs274ngc_exit(); /* saves parameters */

    primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
    primitives->push_back(vert->size());

    // Update animation frame:
    coVRAnimationManager::instance()->setNumTimesteps(vert->size(), this);

    geom->setVertexArray(vert);
    geom->setColorArray(color);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->addPrimitiveSet(primitives);
    geom->dirtyDisplayList();
    geom->setUseDisplayList(false);
    parentNode = loadParent;
    if (parentNode == NULL)
        parentNode = cover->getObjectsRoot();
    //parentNode->addChild(geode.get());


    assert(thePlugin);

    extractToolInfos(filename);
    wpGroup = new osg::Group;
    loadParent->addChild(wpGroup);
    if (filename != NULL)
    {   wpGroup->setName(filename);
    }
    thePlugin->createPath(loadParent);
    if(enableWp)
        thePlugin->createWorkpiece(wpGroup);

    return 0;
}

//--------------------------------------------------------------------
void CNCPlugin::setTimestep(int t)
{
    //if (primitives)
    //{
    //    primitives->at(0) = t+1;
    //    geom->dirtyDisplayList();
    //}
    if (pathPrimitives)
    {
        pathPrimitives->at(0) = pathLineStrip[t];
        pathGeom->dirtyDisplayList();
    }

    if (wpDynamicGeom && t == 0)
    {
        wpResetCutsVec();
        wpDynamicGeom->dirtyDisplayList();
        wpDynamicGeomX->dirtyDisplayList();
        wpDynamicGeomY->dirtyDisplayList();
    }

    if (wpDynamicGeom && t>0)
    {   
        wpMillCutVec(t);
        wpDynamicGeom->dirtyDisplayList();
        wpDynamicGeomX->dirtyDisplayList();
        wpDynamicGeomY->dirtyDisplayList();
    }


//TODO:
    //if t < t_previous
}

int CNCPlugin::unloadGCode(const char *filename, const char *)
{
    (void)filename;

    return 0;
}

bool CNCPlugin::init()
{
    fprintf(stderr, "CNCPlugin::CNCPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);

    coConfig *config = coConfig::getInstance();

    enableWp = configBool("workpiece", "enabled", false, config::Flag::PerModel);
    enableWp->setUpdater([this](bool val) {
        std::cerr << "Workpiece: updating enabled from config" << std::endl;
        if (EnableWpButton)
            EnableWpButton->setState(val);
        //ClipNode* clipNode = cover->getObjectsRoot();
        //if (val)
        //    clipNode->addClipPlane(plane[i].clip.get());
        //else
        //    clipNode->removeClipPlane(plane[i].clip.get());
        });

    EnableWpButton = new ui::Button(PathTab, "Enable");
    EnableWpButton->setText("Enable Workpiece");
    EnableWpButton->setShared(true);
    //plane[i].EnableButton->setPos(0, i);
    EnableWpButton->setCallback([this](bool state) {
        *enableWp = state;
        /*ClipNode* clipNode = cover->getObjectsRoot();
        if (state)
            clipNode->addClipPlane(plane[i].clip.get());
        else
            clipNode->removeClipPlane(plane[i].clip.get());  */
        });
        EnableWpButton->setState(*enableWp); 


    // read the values for each colormap
    numSamples->setText("SampleNum: 0");
    playPause->setText("Play");
    playPause->setCallback([this](bool state) {
        playPause->setText(state ? "Pause" : "Play");
        });
    saveButton->setCallback([this]() {save(); });

    viewPath->setCallback([this](bool state) {
        numSamples->setText("numSamples: " + std::to_string(frameNumber));
        if (state)
        {
            if (!parentNode)
                parentNode = cover->getObjectsRoot();
            parentNode->addChild(geode.get());
        }
        else
        {
            parentNode = geode->getParent(0);
            parentNode->removeChild(geode.get());
        }
        });

    lengthEdit->setValue(1);

    radiusEdit->setValue(1);


    renderMethod->append("renderMethod CPU Billboard");
    renderMethod->append("renderMethod Cg Shader");
    renderMethod->append("renderMethod Point Sprite");
    renderMethod->select(0);

    recordRateTUI->setValue(1);

    fileNameBrowser->setFilter("*.txt");


    //positions = new float [3*MAXSAMPLES+3];

    frameNumber = 0;
    record->setState(false);

    reset->setCallback([this]() {
        frameNumber = 0;
        record->setState(false);
        });

    geoState = new osg::StateSet();
    linemtl = new osg::Material;
    lineWidth = new osg::LineWidth(4.0);
    linemtl.get()->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    linemtl.get()->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    linemtl.get()->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0));
    linemtl.get()->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
    linemtl.get()->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    linemtl.get()->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    geoState->setAttributeAndModes(linemtl.get(), StateAttribute::ON);

    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    geoState->setAttributeAndModes(lineWidth.get(), StateAttribute::ON);

    return true;
}

// this is called if the plugin is removed at runtime
CNCPlugin::~CNCPlugin()
{
    fprintf(stderr, "CNCPlugin::~CNCPlugin\n");

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    if (geode && geode->getNumParents() > 0)
    {
        parentNode = geode->getParent(0);
        if (parentNode)
            parentNode->removeChild(geode.get());
    }
}


void CNCPlugin::save()
{
    /*FILE * fp = fopen(filename,"w");
   if(fp)
   {
      fprintf(fp,"# x,      y,      z,      dx,      dy,     dz\n");
      fprintf(fp,"# numFrames: %d\n",frameNumber);
      for(int n=0;n<frameNumber;n++)
      {
          fprintf(fp,"%010.3f,%010.3f,%010.3f,%010.3f,%010.3f,%010.3f\n",positions[n*3  ],positions[n*3+1], positions[n*3+2],lookat[0][n],lookat[1][n],lookat[2][n]);
      }
      fclose(fp);
   }
   else
   {
      cerr << "could not open file " << filename << endl;
   }*/
}


void CNCPlugin::straightFeed(double x, double y, double z, double a, double b, double c, double feedRate, int tool, int gmode)
{   
    pathX.push_back(x * scaleFactor);
    pathY.push_back(y * scaleFactor);
    pathZ.push_back(z * scaleFactor);
    pathTool.push_back(tool);
    pathG.push_back(gmode);
    pathCenterX.push_back(0);
    pathCenterY.push_back(0);
    pathFeedRate.push_back(feedRate * scaleFactor);
    pathLineStrip.push_back(2);

    /* positions[frameNumber*3  ] = x;
         positions[frameNumber*3+1] = y;
         positions[frameNumber*3+2] = z;*/

    vert->push_back(Vec3(x * scaleFactor, y * scaleFactor, z * scaleFactor));
    float col = feedRate / 6000.0;
    if (col > 1)
        col = 1;
    color->push_back(getColor(col));
    frameNumber++;
    static double oldTime = 0;
    static double oldUpdateTime = 0;
    double time = cover->frameTime();
    if (time - oldUpdateTime > 1.0)
    {
        oldUpdateTime = time;
        numSamples->setText("numSamples: " + std::to_string(frameNumber));
    }
}

void CNCPlugin::arcFeed(double x, double y, double z, double centerX, double centerY, int rotation, double feedRate, int tool)
{
    pathX.push_back(x * scaleFactor);
    pathY.push_back(y * scaleFactor);
    pathZ.push_back(z * scaleFactor);
    pathTool.push_back(tool);
    pathCenterX.push_back(centerX * scaleFactor);
    pathCenterY.push_back(centerY * scaleFactor);
    pathFeedRate.push_back(feedRate * scaleFactor);
    pathLineStrip.push_back(2);
    //rotation gives the direction and the amount of 360� circles (offset by 1)
    if (rotation < 0)
        pathG.push_back(2); //clockwise
    else if (rotation > 0)
        pathG.push_back(3); //counter_clockwise
    else
        pathG.push_back(1); //no arc, straight z movement

    
    /* positions[frameNumber*3  ] = x;
         positions[frameNumber*3+1] = y;
         positions[frameNumber*3+2] = z;*/

    vert->push_back(Vec3(x * scaleFactor, y * scaleFactor, z * scaleFactor));
    float col = feedRate / 6000.0;
    if (col > 1)
        col = 1;
    color->push_back(getColor(col));
    frameNumber++;
    static double oldTime = 0;
    static double oldUpdateTime = 0;
    double time = cover->frameTime();
    if (time - oldUpdateTime > 1.0)
    {
        oldUpdateTime = time;
        numSamples->setText("numSamples: " + std::to_string(frameNumber));
    }
}

osg::Vec4 CNCPlugin::getColor(float pos)
{
    return colorMap.getColor(pos);
}

COVERPLUGIN(CNCPlugin)

/* createPath

   Returned Value: void

   Side Effects:
   Creates the path followed by th cutter.

   Called By:
   CNCPlugin::loadGCode
*/
void CNCPlugin::createPath(osg::Group* parent)
{
    colorG0 = Vec4(0.8f, 0.8f, 0.8f, 1.0);
    colorG1 = Vec4(1.0f, 0.90f, 0.0f, 0.90);
    colorG2 = Vec4(0.0f, 0.9f, 0.2f, 0.90);
    colorG3 = Vec4(0.0f, 0.9f, 0.9f, 0.90);
    Vec4 pushColor = colorG1;

    pathGeode = new Geode();
    pathGeom = new Geometry();
    pathGeode->setStateSet(geoState.get());

    pathGeom->setColorBinding(Geometry::BIND_OFF);

    pathGeode->addDrawable(pathGeom.get());
    pathGeode->setName("Cutting Path");

    // set up geometry
    pathVert = new osg::Vec3Array;
    pathColor = new osg::Vec4Array;
    
    // t = 0
    pathVert->push_back(Vec3(pathX[0], pathY[0], pathZ[0]));
    pathVert->push_back(Vec3(pathX[0], pathY[0], pathZ[0]));
    if (!colorModeGCode)
    {
        float col = pathFeedRate[0] / 6000.0;
        if (col > 1)
            col = 1;
        pushColor = getColor(col);
    }
    else if (pathG[0] == 0)
    {
        pushColor = colorG0;
    }
    else if (pathG[0] == 1)
    {
        pushColor = colorG1;
    }
    else if (pathG[0] == 2)
    {
        pushColor = colorG2;
    }
    else if (pathG[0] == 3)
    {
        pushColor = colorG3;
    }
    pathColor->push_back(pushColor);
    pathColor->push_back(pushColor);
    pathLineStrip[0] = 2;

    for (int t = 1; t < coVRAnimationManager::instance()->getNumTimesteps(); t++)
    {
        if (!colorModeGCode)
        {
            float col = pathFeedRate[0] / 6000.0;
            if (col > 1)
                col = 1;
            pushColor = getColor(col);
        }
        else if (pathG[t] == 0)
            pushColor = colorG0;
        else if (pathG[t] == 1)
            pushColor = colorG1;
        else if (pathG[t] == 2)
            pushColor = colorG2;
        else if (pathG[t] == 3)
            pushColor = colorG3;

        pathLineStrip[t] = pathLineStrip[t - 1];
        pathVert->push_back(Vec3(pathX[t-1], pathY[t-1], pathZ[t-1]));
        pathColor->push_back(pushColor);
        pathLineStrip[t]++;
        if (pathG[t] == 1 || pathG[t] == 0)
        {
            pathVert->push_back(Vec3(pathX[t], pathY[t], pathZ[t]));
            pathColor->push_back(pushColor);
            pathLineStrip[t]++;
        }
        else
        {
            vector<double> arcVec = arcApproximation(t);
            for (int i = 0; i < arcVec.size(); i = i+3)
            {
                pathVert->push_back(Vec3(arcVec[i], arcVec[i + 1], pathZ[t]));
                pathColor->push_back(pushColor);
                pathLineStrip[t]++;
            }
        }
    }
    pathPrimitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
    pathPrimitives->push_back(pathVert->size());
    pathGeom->setVertexArray(pathVert);
    pathGeom->setColorArray(pathColor);
    pathGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    pathGeom->addPrimitiveSet(pathPrimitives);
    pathGeom->dirtyDisplayList();
    pathGeom->setUseDisplayList(false);
    if (parent == NULL)
        parent = cover->getObjectsRoot();
    parent->addChild(pathGeode.get());
}

/* createWorkpiece

   Returned Value: void

   Side Effects:
   Creates the workpiece and adds it to the wpGroup.
   Calls ..

   Called By:
   CNCPlugin::loadGCode
*/
void CNCPlugin::createWorkpiece(Group *parent)
{
    setWpSize();                        //only if size is not provided
    setWpResolution();
    setWpMaterial();

    treeRoot = new TreeNode(Point(0, 0), Point(wpTotalQuadsX, wpTotalQuadsY), 0, wpMaxZ, nullptr);    //hier
    wpAddQuadsToTree(treeRoot);
    wpCreateTimestepVector(treeRoot);

    wpDynamicGeom = new osg::Geometry();
    wpStaticGeom = new osg::Geometry();
    wpDynamicGeomX = new osg::Geometry();
    wpStaticGeomX = new osg::Geometry();
    wpDynamicGeomY = new osg::Geometry();
    wpStaticGeomY = new osg::Geometry();
    wpTreeToGeometry(*wpDynamicGeom, *wpStaticGeom, *wpDynamicGeomX, *wpStaticGeomX, *wpDynamicGeomY, *wpStaticGeomY);

    wpDynamicGeode = new osg::Geode();
    wpDynamicGeode->setName("wpDynamicGeode");
    parent->addChild(wpDynamicGeode);
    wpDynamicGeode->addDrawable(wpDynamicGeom);
    wpDynamicGeodeX = new osg::Geode();
    wpDynamicGeodeX->setName("wpDynamicGeodeX");
    parent->addChild(wpDynamicGeodeX);
    wpDynamicGeodeX->addDrawable(wpDynamicGeomX);
    wpDynamicGeodeY = new osg::Geode();
    wpDynamicGeodeY->setName("wpDynamicGeodeY");
    parent->addChild(wpDynamicGeodeY);
    wpDynamicGeodeY->addDrawable(wpDynamicGeomY);
    //wpDynamicGeom->dirtyDisplayList();
    wpStaticGeode = new osg::Geode();
    wpStaticGeode->setName("wpStaticGeode");
    parent->addChild(wpStaticGeode);
    wpStaticGeode->addDrawable(wpStaticGeom);
    wpStaticGeodeX = new osg::Geode();
    wpStaticGeodeX->setName("wpStaticGeodeX");
    parent->addChild(wpStaticGeodeX);
    wpStaticGeodeX->addDrawable(wpStaticGeomX);
    wpStaticGeodeY = new osg::Geode();
    wpStaticGeodeY->setName("wpStaticGeodeY");
    parent->addChild(wpStaticGeodeY);
    wpStaticGeodeY->addDrawable(wpStaticGeomY);
    //wpStaticGeom->dirtyDisplayList();
}

/* wpAddQuadsToTree

   Returned Value: void

   Side Effects:
   Adds all Quads that will be cutted as to the tree.

   Called By:
   CNCPlugin::createWorkpiece
*/
void CNCPlugin::wpAddQuadsToTree(TreeNode* treeRoot)
{
    for (int t = 1; t < coVRAnimationManager::instance()->getNumTimesteps(); t++)
    {
        if (activeTool != pathTool[t])
        {
            setActiveTool(pathTool[t]);
        }
        if (pathG[t] == 2 || pathG[t] == 3)
            wpAddQuadsG2G3(wpMaxZ, t, treeRoot);
        else
            wpAddQuadsG0G1(wpMaxZ, t, treeRoot);
    }
    treeRoot->setDescendants();
    treeRoot->traverseAndCombineMillQuads();
    treeRoot->setDescendants();
    treeRoot->traverseAndCombineAll();

    treeRoot->searchForInvalid();
    //treeRoot->sortChildren();
    treeRoot->setLevels();
    treeRoot->setDescendants();
    treeRoot->setSideWalls();
}

void CNCPlugin::wpAddQuadsG0G1(double z, int t, TreeNode* treeRoot)
{
    if (pathZ[t] < z)
    {
        //bounding Box
        double boxMinX = std::min(pathX[t - 1], pathX[t]) - cuttingRad;
        double boxMaxX = std::max(pathX[t - 1], pathX[t]) + cuttingRad;
        double boxMinY = std::min(pathY[t - 1], pathY[t]) - cuttingRad;
        double boxMaxY = std::max(pathY[t - 1], pathY[t]) + cuttingRad;

        int ixMin = (boxMinX - wpMinX) / wpResX;
        int ixMax = (boxMaxX - wpMinX) / wpResX;
        int iyMin = (boxMinY - wpMinY) / wpResY;
        int iyMax = (boxMaxY - wpMinY) / wpResY;
        if (ixMin < 1)          //hier
            ixMin = 1;
        if (iyMin < 1)
            iyMin = 1;
        if (ixMax > wpTotalQuadsX)
            ixMax = wpTotalQuadsX;
        if (iyMax > wpTotalQuadsY)
            iyMax = wpTotalQuadsY;

        for (int iy = iyMin; iy <= iyMax; iy++)
        {
            for (int ix = ixMin; ix <= ixMax; ix++)
            {
                int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
                double wpQuadIXCenter = wpMinX + ix * wpResX - wpResX / 2;      //hier
                double wpQuadIYCenter = wpMinY + iy * wpResY - wpResY / 2;
                double dist = distancePointLineSegment(wpQuadIXCenter, wpQuadIYCenter, pathX[t - 1], pathY[t - 1], pathX[t], pathY[t]);
                if (dist < cuttingRad)
                {   
                    treeRoot->millQuad(ix, iy, pathZ[t], t);
                }
            }
        }
    }
}
void CNCPlugin::wpAddQuadsG2G3(double z, int t, TreeNode* treeRoot)
{
    if (pathZ[t] < z)
    {
        double arcRadius = distancePointPoint(pathCenterX[t], pathCenterY[t], pathX[t], pathY[t]);// Abstand center zu t;
        //bounding Box
        double boxMinX = pathCenterX[t] - arcRadius - cuttingRad;
        double boxMaxX = pathCenterX[t] + arcRadius + cuttingRad;
        double boxMinY = pathCenterY[t] - arcRadius - cuttingRad;
        double boxMaxY = pathCenterY[t] + arcRadius + cuttingRad;

        int ixMin = (boxMinX - wpMinX) / wpResX;
        int ixMax = (boxMaxX - wpMinX) / wpResX;
        int iyMin = (boxMinY - wpMinY) / wpResY;
        int iyMax = (boxMaxY - wpMinY) / wpResY;
        if (ixMin < 1)      //hier
            ixMin = 1;
        if (iyMin <= 1)
            iyMin = 1;
        if (ixMax > wpTotalQuadsX)
            ixMax = wpTotalQuadsX;
        if (iyMax > wpTotalQuadsY)
            iyMax = wpTotalQuadsY;

        double angleEnd = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t], pathY[t]);
        double angleStart = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t - 1], pathY[t - 1]);

        for (int iy = iyMin; iy <= iyMax; iy++)
        {
            for (int ix = ixMin; ix <= ixMax; ix++)
            {
                int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
                double wpQuadIXCenter = wpMinX + ix * wpResX - wpResX / 2;      //hier
                double wpQuadIYCenter = wpMinY + iy * wpResY - wpResY / 2;
                double distPointCircle = abs(distancePointPoint(pathCenterX[t], pathCenterY[t], wpQuadIXCenter, wpQuadIYCenter) - arcRadius);
                if (distPointCircle < cuttingRad)
                {
                    double angleI = anglePointPoint(pathCenterX[t], pathCenterY[t], wpQuadIXCenter, wpQuadIYCenter);
                    bool inside = checkInsideArcG2(angleI, angleStart, angleEnd);
                    if (pathG[t] == 3)
                        inside = !inside;
                    bool insert = inside;
                    if (!inside)
                    {
                        double dist = std::min(distancePointPoint(pathX[t], pathY[t], wpQuadIXCenter, wpQuadIYCenter),
                            distancePointPoint(pathX[t - 1], pathY[t - 1], wpQuadIXCenter, wpQuadIYCenter));
                        if (dist < cuttingRad)
                            insert = true;
                    }
                    if (insert)
                    {
                        treeRoot->millQuad(ix, iy, pathZ[t], t);
                    }
                }
            }
        }
    }
}

///* Return minimum distance between point p and line vw */
//double CNCPlugin::distancePointLine(double px, double py, double x1, double y1, double x2, double y2)
//{
//    double a = y1 - y2;
//    double b = x2 - x1;
//    double c = x1 * y2 - x2 * y1;
//
//    return (abs(a * px + b * py + c) / sqrt(a * a + b * b));
//}

/* Return minimum distance between point p and linesegment vw */
double CNCPlugin::distancePointLineSegment(double px, double py, double x1, double y1, double x2, double y2)
{
    double a = px - x1;
    double b = py - y1;
    double c = x2 - x1;
    double d = y2 - y1;
    double dot = a * c + b * d;
    double len_sq = c * c + d * d;
    double param = -1;
    if (len_sq != 0)        //in case of 0 length line
        param = dot / len_sq;

    double xx, yy;

    if (param < 0) {
        xx = x1;
        yy = y1;
    }
    else if (param > 1) {
        xx = x2;
        yy = y2;
    }
    else {
        xx = x1 + param * c;
        yy = y1 + param * d;
    }

    double dx = px - xx;
    double dy = py - yy;
    return sqrt(dx * dx + dy * dy);
}

/* Return distance between point p and point 1 */
double CNCPlugin::distancePointPoint(double px, double py, double x1, double y1)
{
    return sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1));
}

/* Return angle between centerpoint p - point 1 and x axis */
double CNCPlugin::anglePointPoint(double px, double py, double x1, double y1)
{   
    return atan2(y1 - py, x1 - px);
}

/* Checks if pAngle is between 1 and 2, depending on gCode (G2/G3)*/
bool CNCPlugin::checkInsideArcG2(double pAngle, double angle1, double angle2)
{
    double delta1 = pAngle - angle1;
    double delta2 = pAngle - angle2;
    
    if (angle1 < angle2)
    {
        if (delta1 * delta2 > 0)
            return true;
        else
            return false;
    }
    else
    {
        if (delta1 * delta2 < 0)
            return true;
        else
            return false;
    }
}

/* wpTreeToGeometry

   Returned Value: by reference, sets dynamicGeo and staticGeo for Vertices that change and those that doesn't change during milling.

   Side Effects:
   Creates two Geometry, using the Quads from the CNCTree treeRoot.

   Called By:
   CNCPlugin::createWorkpiece
*/
//osg::ref_ptr<osg::Geometry> CNCPlugin::wpTreeToGeometry()
void CNCPlugin::wpTreeToGeometry(osg::Geometry& dynamicGeo, osg::Geometry& staticGeo, osg::Geometry& dynamicGeoX, osg::Geometry& staticGeoX, osg::Geometry& dynamicGeoY, osg::Geometry& staticGeoY)
{
    //create geometry
    wpDynamicColors = new Vec4Array();
    wpStaticColors = new Vec4Array();
    auto pointsDynamic = new Vec3Array();
    auto pointsStatic = new Vec3Array();
    wpDynamicPrimitives = new DrawArrayLengths(PrimitiveSet::QUADS);
    wpStaticPrimitives = new DrawArrayLengths(PrimitiveSet::QUADS);

    wpTreeToGeoTop(*pointsDynamic, *pointsStatic);

    primitiveResetCounterDynamic = pointsDynamic->size();

    wpDynamicPrimitives->push_back(pointsDynamic->size());
    wpDynamicPrimitives->setName("wpDynamicPrimitives");
    dynamicGeo.addPrimitiveSet(wpDynamicPrimitives);
    wpStaticPrimitives->push_back(pointsStatic->size());
    wpStaticPrimitives->setName("wpStaticPrimitives");
    staticGeo.addPrimitiveSet(wpStaticPrimitives);

    wpDynamicVerticalPrimX = new DrawElementsUInt(PrimitiveSet::QUADS);
    wpDynamicVerticalPrimY = new DrawElementsUInt(PrimitiveSet::QUADS);
    wpStaticVerticalPrimX = new DrawElementsUInt(PrimitiveSet::QUADS);
    wpStaticVerticalPrimY = new DrawElementsUInt(PrimitiveSet::QUADS);
    wpTreeToGeoSideWalls(*pointsDynamic, *pointsStatic, *wpDynamicVerticalPrimX, *wpDynamicVerticalPrimY, *wpStaticVerticalPrimX, *wpStaticVerticalPrimY);

    wpDynamicVerticalPrimX->setName("wpDynamicVerticalPrimX");
    wpDynamicVerticalPrimY->setName("wpDynamicVerticalPrimY");
    dynamicGeoX.addPrimitiveSet(wpDynamicVerticalPrimX);
    dynamicGeoY.addPrimitiveSet(wpDynamicVerticalPrimY);
    wpStaticVerticalPrimX->setName("wpStaticVerticalPrimX");
    wpStaticVerticalPrimY->setName("wpStaticVerticalPrimY");
    staticGeoX.addPrimitiveSet(wpStaticVerticalPrimX);
    staticGeoY.addPrimitiveSet(wpStaticVerticalPrimY);

    wpDynamicColors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.50f));
    //wpDynamicColors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.50f));
    //wpDynamicColors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.50f));
    dynamicGeo.setVertexArray(pointsDynamic);
    dynamicGeo.setColorArray(wpDynamicColors);
    dynamicGeo.setColorBinding(osg::Geometry::BIND_OVERALL);
    //dynamicGeo.setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    dynamicGeoX.setVertexArray(pointsDynamic);
    dynamicGeoX.setColorArray(wpDynamicColors);
    dynamicGeoX.setColorBinding(osg::Geometry::BIND_OVERALL);
    dynamicGeoY.setVertexArray(pointsDynamic);
    dynamicGeoY.setColorArray(wpDynamicColors);
    dynamicGeoY.setColorBinding(osg::Geometry::BIND_OVERALL);
    wpStaticColors->push_back(osg::Vec4(0.5f, 0.0f, 0.0f, 0.50f));
    //wpStaticColors->push_back(osg::Vec4(0.5f, 0.0f, 0.0f, 0.50f));
    //wpStaticColors->push_back(osg::Vec4(0.5f, 0.0f, 0.0f, 0.50f));
    staticGeo.setVertexArray(pointsStatic);
    staticGeo.setColorArray(wpStaticColors);
    staticGeo.setColorBinding(osg::Geometry::BIND_OVERALL);
    //staticGeo.setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    staticGeoX.setVertexArray(pointsStatic);
    staticGeoX.setColorArray(wpStaticColors);
    staticGeoX.setColorBinding(osg::Geometry::BIND_OVERALL);
    staticGeoY.setVertexArray(pointsStatic);
    staticGeoY.setColorArray(wpStaticColors);
    staticGeoY.setColorBinding(osg::Geometry::BIND_OVERALL);

    //dynamicGeo.setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    wpDynamicNormals = new osg::Vec3Array;
    wpDynamicNormalsX = new osg::Vec3Array;
    wpDynamicNormalsY = new osg::Vec3Array;
    wpDynamicNormals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
    wpDynamicNormalsX->push_back(osg::Vec3(0.0f, 1.0f, 0.0f));
    wpDynamicNormalsY->push_back(osg::Vec3(1.0f, 0.0f, 0.0f));
    //dynamicGeo.setNormalArray(wpDynamicNormals);// , osg::Array::BIND_OVERALL);
    //staticGeo.setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    //staticGeo.setNormalArray(wpDynamicNormals);
    dynamicGeo.setNormalArray(wpDynamicNormals, osg::Array::BIND_OVERALL);
    staticGeo.setNormalArray(wpDynamicNormals, osg::Array::BIND_OVERALL);
    dynamicGeoX.setNormalArray(wpDynamicNormalsX, osg::Array::BIND_OVERALL);
    staticGeoX.setNormalArray(wpDynamicNormalsX, osg::Array::BIND_OVERALL);
    dynamicGeoY.setNormalArray(wpDynamicNormalsY, osg::Array::BIND_OVERALL);
    staticGeoY.setNormalArray(wpDynamicNormalsY, osg::Array::BIND_OVERALL);

    dynamicGeo.setStateSet(wpStateSet.get());
    dynamicGeo.dirtyDisplayList();
    staticGeo.setStateSet(wpStateSet.get());
    dynamicGeoX.setStateSet(wpStateSet.get());
    dynamicGeoX.dirtyDisplayList();
    staticGeoX.setStateSet(wpStateSet.get());
    dynamicGeoY.setStateSet(wpStateSet.get());
    dynamicGeoY.dirtyDisplayList();
    staticGeoY.setStateSet(wpStateSet.get());
    //  staticGeo.dirtyDisplayList();
}
/* wpTreeToGeoTop

   Returned Value: void, by reference

   Side Effects:
   Adds the topsurface from treeRoot to VerticeVec "pointsDynamic" and "pointsStatic", depending on millTimestep.size().

   Called By:
   wpTreeToGeometry
*/
void CNCPlugin::wpTreeToGeoTop(osg::Vec3Array& pointsDynamic, osg::Vec3Array& pointsStatic)
{
    wpAddVertexsForGeo(&pointsStatic, 0, wpTotalQuadsX, 0, wpTotalQuadsY, wpMinZ, primitivePosCounterStatic);   // bottom

    std::stack<TreeNode*> nodeStack;
    nodeStack.push(treeRoot);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        for (TreeNode* childTree : node->getChildTrees())
        {
            nodeStack.push(childTree);
        }
        if (node->getChildTrees().size() == 0)
        {
            auto tl = node->getTopLeft();
            auto br = node->getBotRight();
            if (node->getMillTimesteps().size() == 0)
            {
                node->setPrimitivePos(primitivePosCounterStatic);
                wpAddVertexsForGeo(&pointsStatic, tl.x + 0, br.x, tl.y + 0, br.y, wpMaxZ, primitivePosCounterStatic);
            }
            else
            {
                node->setPrimitivePos(primitivePosCounterDynamic);
                wpAddVertexsForGeo(&pointsDynamic, tl.x + 0, br.x, tl.y + 0, br.y, wpMaxZ, primitivePosCounterDynamic);
            }
        }
    }
}
/* wpTreeToGeoSideWalls

   Returned Value: void, by Reference

   Side Effects:
   Adds the sidewalls from treeRoot by index using VerticeVec "pointsDynamic" and "pointsStatic".

   Called By:
   wpTreeToGeometry
*/
void CNCPlugin::wpTreeToGeoSideWalls(osg::Vec3Array& pointsDynamic, osg::Vec3Array& pointsStatic, osg::DrawElementsUInt& wpDynamicVerticalPrimX, osg::DrawElementsUInt& wpDynamicVerticalPrimY, osg::DrawElementsUInt& wpStaticVerticalPrimX, osg::DrawElementsUInt& wpStaticVerticalPrimY)
{
    std::stack<TreeNode*> nodeStack;
    nodeStack.push(treeRoot);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        for (TreeNode* childTree : node->getChildTrees())
        {
            nodeStack.push(childTree);
        }
        if (node->getChildTrees().size() == 0)
        {
            int primPos = node->getPrimitivePos();
            vector<bool> sideWalls = node->getSideWalls();
            if (std::any_of(sideWalls.begin(), sideWalls.end(), [](bool b) {return b; }))  // if any_of sideWalls is true 
            {
                auto tl = node->getTopLeft();
                auto br = node->getBotRight();
                if (node->getMillTimesteps().size() == 0)
                {
                    wpAddVertexsForGeo(&pointsStatic, tl.x + 0, br.x, tl.y + 0, br.y, wpMinZ, primitivePosCounterStatic);
                    for (int side = 0; side < 4; side++)
                    {
                        if (sideWalls[side])
                        {
                            wpAddSideForGeo(&wpStaticVerticalPrimX, &wpStaticVerticalPrimY, primPos, primitivePosCounterStatic - 4, side);
                        }
                    }
                }
                else
                {
                    wpAddVertexsForGeo(&pointsDynamic, tl.x + 0, br.x, tl.y + 0, br.y, wpMinZ, primitivePosCounterDynamic);
                    for (int side = 0; side < 4; side++)
                    {
                        if (sideWalls[side])
                        {
                            wpAddSideForGeo(&wpDynamicVerticalPrimX, &wpDynamicVerticalPrimY, primPos, primitivePosCounterDynamic - 4, side);
                        }
                    }
                }
            }
        }
    }
}
/* wpAddVertexsForGeo

   Side Effects:
   Adds 4 Vertices and increases the primitive PositionCounter according to the passed variables.

   Called By:
   wpTreeToGeoTop, wpTreeToGeoSideWalls
*/
void CNCPlugin::wpAddVertexsForGeo(osg::Vec3Array* points, int minIX, int maxIX, int minIY, int maxIY, double z, int &primPosCounter)
{
    if (minIX < 0)          // needed for border - 1 
        minIX = 0;
    if (minIY < 0)
        minIY = 0;
    points->push_back(Vec3(wpMinX + minIX * wpResX, wpMinY + minIY * wpResY, z));
    points->push_back(Vec3(wpMinX + maxIX * wpResX, wpMinY + minIY * wpResY, z));
    points->push_back(Vec3(wpMinX + maxIX * wpResX, wpMinY + maxIY * wpResY, z));
    points->push_back(Vec3(wpMinX + minIX * wpResX, wpMinY + maxIY * wpResY, z));
    primPosCounter += 4;
    return;
}
/* wpAddSideForGeo

   Side Effects:
   Adds a vertical sideWall by adding the Indices to the primitive lists according to the given position and the side.

   Called By:
   wpTreeToGeoSideWalls
*/
void CNCPlugin::wpAddSideForGeo(osg::DrawElementsUInt* wpVerticalPrimitivesX, osg::DrawElementsUInt* wpVerticalPrimitivesY, int primPosTop, int primPosBot, int side)
{
    int point0, point1, point2, point3;

    if (side == 0)
    {
        point0 = primPosTop;
        point1 = primPosTop + 3;
        point2 = primPosBot + 3;
        point3 = primPosBot;
    }
    else if (side == 1)
    {
        point0 = primPosTop;
        point1 = primPosTop + 1;
        point2 = primPosBot + 1;
        point3 = primPosBot;
    }
    else if (side == 2)
    {
        point0 = primPosTop + 1;
        point1 = primPosTop + 2;
        point2 = primPosBot + 2;
        point3 = primPosBot + 1;
    }
    else if (side == 3)
    {
        point0 = primPosTop + 2;
        point1 = primPosTop + 3;
        point2 = primPosBot + 3;
        point3 = primPosBot + 2;
    }

    if (side == 0 || side == 2)
    {
        wpVerticalPrimitivesY->push_back(point0);
        wpVerticalPrimitivesY->push_back(point1);
        wpVerticalPrimitivesY->push_back(point2);
        wpVerticalPrimitivesY->push_back(point3);
    }
    else if (side == 1 || side == 3)
    {
        wpVerticalPrimitivesX->push_back(point0);
        wpVerticalPrimitivesX->push_back(point1);
        wpVerticalPrimitivesX->push_back(point2);
        wpVerticalPrimitivesX->push_back(point3);
    }
}

/* wpCreateTimestepVector

   Side Effects:
   Adds all TreeNodes that will be cutted in time step t to timestepVec[t].

   Called By:
   CNCPlugin::createWorkpiece
*/
void CNCPlugin::wpCreateTimestepVector(TreeNode* treeRoot)
{
    std::vector<TreeNode*> tempVec;
    for (int t = 0; t < pathZ.size(); t++)
        timestepVec.push_back(tempVec);
    timestepVec = treeRoot->writeTimestepVector(timestepVec);
}
/* setWpSize

   Side Effects:
   Sets wpMinX, wpMaxX, wpMinY, wpMaxY, wpMinZ, wpMaxZ by extracting Information from the G commands and adding some allowance.

   Called By:
   createWorkpiece
*/
void CNCPlugin::setWpSize()
{
    if(!wpSizeExtracted)
    {
        wpMinX = *std::min_element(pathX.begin(), pathX.end() - 1) - wpAllowance;
        wpMaxX = *std::max_element(pathX.begin(), pathX.end() - 1) + wpAllowance;
        wpMinY = *std::min_element(pathY.begin(), pathY.end() - 1) - wpAllowance;
        wpMaxY = *std::max_element(pathY.begin(), pathY.end() - 1) + wpAllowance;
        wpMinZ = *std::min_element(pathZ.begin(), pathZ.end()) - wpAllowance;
        wpMaxZ = 0;
        for (int i = 0; i < pathG.size(); i++)
        {
            if (pathG[i] == 2 || pathG[i] == 3)
            {
                double rad = distancePointPoint(pathCenterX[i], pathCenterY[i], pathX[i], pathY[i]);
                wpMinX = std::min(wpMinX, pathCenterX[i] - rad - wpAllowance / 2);
                wpMaxX = std::max(wpMaxX, pathCenterX[i] + rad + wpAllowance / 2);
                wpMinY = std::min(wpMinY, pathCenterY[i] - rad - wpAllowance / 2);
                wpMaxY = std::max(wpMaxY, pathCenterY[i] + rad + wpAllowance / 2);
            }
        }
    }
    wpLengthX = wpMaxX - wpMinX;
    wpLengthY = wpMaxY - wpMinY;
    wpLengthZ = wpMaxZ - wpMinZ;
}
/* setWpResolution

   Side Effects:
   Sets the number of quads and their actual size for both X and Y.

   Called By:
   createWorkpiece
*/
void CNCPlugin::setWpResolution()
{
    wpTotalQuadsX = ::round(wpLengthX / wpResolution);
    wpResX = wpLengthX / wpTotalQuadsX;

    wpTotalQuadsY = ::round(wpLengthY / wpResolution);
    wpResY = wpLengthY / wpTotalQuadsY;
}
/* setWpMaterial

   Side Effects:
   Defines the material / the stateSet which will be used.

   Called By:
   createWorkpiece
*/
void CNCPlugin::setWpMaterial()
{
    //wpStateSet = new osg::StateSet();
    wpStateSet = new osg::StateSet;
    wpMaterial = new osg::Material;
 //   wpLineWidth = new osg::LineWidth(2.0);
    wpMaterial.get()->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    wpMaterial.get()->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    wpMaterial.get()->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(0.80f, 0.80f, 0.80f, 1.0));
    wpMaterial.get()->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(0.5f, 0.5f, 0.5f, 1.0));
    wpMaterial.get()->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    wpMaterial.get()->setShininess(osg::Material::FRONT_AND_BACK, 8.0f);
    

    wpStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    wpStateSet->setMode(GL_NORMALIZE, osg::StateAttribute::ON);
    wpStateSet->setMode(GL_BLEND, osg::StateAttribute::OFF);
    wpStateSet->setMode(osg::StateAttribute::PROGRAM, osg::StateAttribute::OFF);
    wpStateSet->setMode(osg::StateAttribute::VERTEXPROGRAM, osg::StateAttribute::OFF);
    wpStateSet->setMode(osg::StateAttribute::FRAGMENTPROGRAM, osg::StateAttribute::OFF);
    wpStateSet->setMode(osg::StateAttribute::TEXTURE, osg::StateAttribute::OFF);

    wpStateSet->setAttributeAndModes(wpMaterial.get(), StateAttribute::ON);
  
  //  wpStateSet->setAttributeAndModes(wpLineWidth.get(), StateAttribute::ON); //nur fur Linien. Ueberflussig?

/*    osg::ref_ptr<osg::PolygonMode> polyMode(new osg::PolygonMode());
    polyMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL); //LINE);
    wpStateSet->setAttribute(polyMode.get());
    */
}
/* extractToolInfos

   Side Effects:
   Tries to gain tool information from the gcode file using regex. adds a default tool if the used number couldn't be extracted.
   Tries to read workpiece dimensions using regex.

   Called By:
   loadGCode
*/
void CNCPlugin::extractToolInfos(const std::string& filename)
{
    // ein default tool speicher, falls keine tools definiert / eingelesen wurden
    ToolInfo tool_default;
    tool_default.toolNumber = -1;
    tool_default.diameter = 1.0 * scaleFactor;
    tool_default.cornerRadius = 0;
    tool_default.coneAngle = 180;
    tool_default.zMin = 0;
    tool_default.toolType = "Default Tool, r=0.5mm";
    toolInfoList.push_back(tool_default);

    std::regex toolRegex("\\(T(\\d+)\\s*D=(\\d*\\.?\\d*)\\s*CR=(\\d*\\.?\\d*)\\s*(?:KONIK=(\\d+\\.?\\d*)\\w*\\s*)?-\\s+ZMIN=(-?\\d*\\.?\\d*)\\s*-\\s*(\\S*)\\)");
    std::regex wpDimensionRegex("\\(\\s*[xX][mM][iI][nN]\\s*=\\s*(-?\\d*\\.?\\d*)\\s*\\,?\\s*[xX][mM][aA][xX]\\s*=\\s*(-?\\d*\\.?\\d*)\\s*\\,?\\s*[yY][mM][iI][nN]\\s*=\\s*(-?\\d*\\.?\\d*)\\s*\\,?\\s*[yY][mM][aA][xX]\\s*=\\s*(-?\\d*\\.?\\d*)\\s*\\,?\\s*[zZ][mM][iI][nN]\\s*=\\s*(-?\\d*\\.?\\d*)\\s*\\,?\\s*[zZ][mM][aA][xX]\\s*=\\s*(-?\\d*\\.?\\d*)\\s*\\)");
    std::ifstream infile(filename);
    std::string line;
    while (std::getline(infile, line))
    {
        std::smatch match;
        if (std::regex_search(line, match, toolRegex)) {
            ToolInfo tool_info;
            tool_info.toolNumber = std::stoi(match[1]);
            tool_info.diameter = std::stod(match[2]) * scaleFactor;
            tool_info.cornerRadius = std::stod(match[3]) * scaleFactor;
            if (match[4].matched)
                tool_info.coneAngle = std::stod(match[4]);
            tool_info.zMin = std::stod(match[5]) * scaleFactor;
            tool_info.toolType = match[6];
            toolInfoList.push_back(tool_info);
        }
        std::smatch wpMatch;
        if (std::regex_search(line, wpMatch, wpDimensionRegex)) {
            wpSizeExtracted = true;
            wpMinX = std::stod(wpMatch[1]) * scaleFactor;
            wpMaxX = std::stod(wpMatch[2]) * scaleFactor;
            wpMinY = std::stod(wpMatch[3]) * scaleFactor;
            wpMaxY = std::stod(wpMatch[4]) * scaleFactor;
            wpMinZ = std::stod(wpMatch[5]) * scaleFactor;
            wpMaxZ = std::stod(wpMatch[6]) * scaleFactor;
        }
    }
}
/* setActiveTool

   Side Effects:
   Sets activeTool and cuttingRad if toolInfoList contains this number. Else set to default tool -1.

   Called By:
   wpAddQuadsToTree
*/
void CNCPlugin::setActiveTool(int slot)
{
    activeTool = -1;
    ToolInfo tool = toolInfoList[0];
    for (const auto& t : toolInfoList)
    {
        if (t.toolNumber == slot)
        {
            tool = t;
            activeTool = slot;
        }
    }
    cuttingRad = tool.diameter / 2;
}

/* arcApproximation
    
   Returned Value: vector<double> with the coordinates in the order x, y, z for every used Vertex.

   Side Effects:
   Approximates the arc of timestep t by using multiple linear lines.

   Called By:
   createPath
*/
vector<double> CNCPlugin::arcApproximation(int t)
{
    vector<double> xyzCoords;
    double r = distancePointPoint(pathCenterX[t], pathCenterY[t], pathX[t], pathY[t]);
    double thetaStart = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t - 1], pathY[t - 1]);
    double thetaEnd = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t], pathY[t]);

    double dtheta = thetaEnd - thetaStart;
    if (pathG[t] == 2)
    {
        if (dtheta > 0)
            dtheta = dtheta - (2 * PI);
    }
    else 
    {
        if (dtheta < 0)
            dtheta = dtheta + (2 * PI);
    }

    int numSegments = ceil(abs(dtheta) * approxLength);
    numSegments = std::max(2, numSegments);
    double segmentTheta = dtheta / numSegments;

    double currentTheta = thetaStart;
    for (int i = 1; i <= numSegments; i++) {
        double nextTheta = currentTheta + segmentTheta;
        double x = pathCenterX[t] + r * cos(nextTheta);
        double y = pathCenterY[t] + r * sin(nextTheta);
        double z = pathZ[t - 1] + (pathZ[t] - pathZ[t - 1]) * i / numSegments;
        currentTheta = nextTheta;
        xyzCoords.push_back(x);
        xyzCoords.push_back(y);
        xyzCoords.push_back(z);
    }
    return xyzCoords;
}


/* wpMillCut

   Side Effects:
   changes the Geometry and sets new zCoords for timeStep t.

   Called By:
   CNCPlugin::setTimestep
*/
void CNCPlugin::wpMillCutVec(int t)
{
    osg::Vec3Array* piece = static_cast<osg::Vec3Array*>(wpDynamicGeom->getVertexArray());
    osg::Vec3Array* pieceX = static_cast<osg::Vec3Array*>(wpDynamicGeomX->getVertexArray());
    osg::Vec3Array* pieceY = static_cast<osg::Vec3Array*>(wpDynamicGeomY->getVertexArray());
    for (TreeNode* tree : timestepVec[t])
    {
        int primPos = tree->getPrimitivePos();

        piece->at(primPos)[2] = pathZ[t];  // unpraezise bezueglich Hoehe Z. tatsaechliche Fraeserhoehe an Stelle piece-at(i) eventuell abweichend!
        piece->at(primPos + 1)[2] = pathZ[t];
        piece->at(primPos + 2)[2] = pathZ[t];
        piece->at(primPos + 3)[2] = pathZ[t];
        pieceX->at(primPos)[2] = pathZ[t];  // unpräzise bezüglich Höhe Z. tatsächliche Fräserhöhe an Stelle piece-at(i) eventuell abweichend!
        pieceX->at(primPos + 1)[2] = pathZ[t];
        pieceX->at(primPos + 2)[2] = pathZ[t];
        pieceX->at(primPos + 3)[2] = pathZ[t];
        pieceY->at(primPos)[2] = pathZ[t];  // unpräzise bezüglich Höhe Z. tatsächliche Fräserhöhe an Stelle piece-at(i) eventuell abweichend!
        pieceY->at(primPos + 1)[2] = pathZ[t];
        pieceY->at(primPos + 2)[2] = pathZ[t];
        pieceY->at(primPos + 3)[2] = pathZ[t];
    }
}
/* wpResetCutsVec

   Side Effects:
   changes the Geometry and resets all zCoords to wpMaxZ (for t=0).
   Uses primitiveResetCounter to determine which Vertices need to be resettet.

   Called By:
   CNCPlugin::setTimestep
*/
void CNCPlugin::wpResetCutsVec()
{
    osg::Vec3Array* piece = static_cast<osg::Vec3Array*>(wpDynamicGeom->getVertexArray());
    osg::Vec3Array* pieceX = static_cast<osg::Vec3Array*>(wpDynamicGeomX->getVertexArray());
    osg::Vec3Array* pieceY = static_cast<osg::Vec3Array*>(wpDynamicGeomY->getVertexArray());
    for (int i = 0; i < primitiveResetCounterDynamic; i++)
    {
        piece->at(i)[2] = wpMaxZ;
        pieceX->at(i)[2] = wpMaxZ;
        pieceY->at(i)[2] = wpMaxZ;
    }
}