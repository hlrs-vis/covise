/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\
**                                                            (C)2005 HLRS  **
**                                                                          **
** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** April-05  v1	    				       		                         **
**                                                                          **
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
: ui::Owner("CNCPlugin", cover->ui)
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
    parentNode->addChild(geode.get());


    assert(thePlugin);

    wpGroup = new osg::Group;
    loadParent->addChild(wpGroup);
    if (filename != NULL)
    {   wpGroup->setName(filename);
    }
    thePlugin->createWorkpiece(wpGroup);

    
    return 0;
}

//--------------------------------------------------------------------
void CNCPlugin::setTimestep(int t)
{
    if (primitives)
    {
        primitives->at(0) = t+1;
        geom->dirtyDisplayList();
    }

    if (wpTopGeom && t == 0)
    {
    //    wpResetCuts(static_cast<osg::Vec3Array*>(wpTopGeom->getVertexArray()), t);
    //    wpTopGeom->dirtyDisplayList();
    }

    if (wpTopGeom && t>0)
    {   
        if (t % 1 == 0)
        {
            //wpMillCut(wpTopGeom, static_cast<osg::Vec3Array*>(wpTopGeom->getVertexArray()), t);
            if (pathG[t] == 2 || pathG[t] == 3) {}
  //              wpMillCutTreeCircle(wpTopGeom, static_cast<osg::Vec3Array*>(wpTopGeom->getVertexArray()), t);
  //          else
  //              wpMillCutTree(wpTopGeom, static_cast<osg::Vec3Array*>(wpTopGeom->getVertexArray()), t);

            wpTopGeom->dirtyDisplayList();
        }
    }

    if (t == 10)
    {
  /*      auto asde = wpGroup->getChild(0);
        int asdf = wpGroup->getNumChildren();
   //     auto asdg = wpBotGeode->getBoundingBox();
        auto asdh = wpTopGeode->getNumChildren();
        int i = 0;
    */
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
    lineWidth = new osg::LineWidth(2.0);
    linemtl.get()->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    linemtl.get()->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    linemtl.get()->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0));
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


void CNCPlugin::straightFeed(double x, double y, double z, double a, double b, double c, double feedRate)
{   
    pathX.push_back(x / 1000.0);
    pathY.push_back(y / 1000.0);
    pathZ.push_back(z / 1000.0);
    pathG.push_back(1);
    pathCenterX.push_back(0);
    pathCenterY.push_back(0);


    /* positions[frameNumber*3  ] = x;
         positions[frameNumber*3+1] = y;
         positions[frameNumber*3+2] = z;*/

    vert->push_back(Vec3(x / 1000.0, y / 1000.0, z / 1000.0));
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

void CNCPlugin::arcFeed(double x, double y, double z, double centerX, double centerY, int rotation, double feedRate)
{
    pathX.push_back(x / 1000.0);
    pathY.push_back(y / 1000.0);
    pathZ.push_back(z / 1000.0);
    pathCenterX.push_back(centerX / 1000.0);
    pathCenterY.push_back(centerY / 1000.0);
    //rotation gives the direction and the amount of 360° circles (offset by 1)
    if (rotation < 0)
        pathG.push_back(2); //clockwise
    else if (rotation > 0)
        pathG.push_back(3); //counter_clockwise
    else
        pathG.push_back(1); //no arc, straight z movement

    
    /* positions[frameNumber*3  ] = x;
         positions[frameNumber*3+1] = y;
         positions[frameNumber*3+2] = z;*/

    vert->push_back(Vec3(x / 1000.0, y / 1000.0, z / 1000.0));
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


//    for (int iy = 0; iy < wpTotalQuadsY; iy++)
//    {
//        for (int ix = 0; ix < wpTotalQuadsX; ix++)
//        {
//            cuttedFaces.push_back(-1);
//        }
//    }

    treeRoot = new TreeNode(Point(-1, -1), Point(wpTotalQuadsX, wpTotalQuadsY), 0, wpMaxZ, nullptr);
    wpAddQuadsToTree(treeRoot);

    wpTopGeom = wpTreeToGeometry();


 //   wpTopGeom = createWpTopTree(wpMinX, wpMaxX, wpMinY, wpMaxY, wpMaxZ);

//
    wpBotGeom = createWpBottom(wpMinX, wpMaxX, wpMinY, wpMaxY, wpMinZ, wpMaxZ);
    wpBotGeode = new osg::Geode();
    wpBotGeode->setName("wpBotGeode");
    parent->addChild(wpBotGeode);
    wpBotGeode->addDrawable(wpBotGeom);
    wpBotGeom->dirtyDisplayList();
   

 ////   wpTopGeom = createWpTop(wpMinX, wpMaxX, wpMinY, wpMaxY, wpMaxZ);
    wpTopGeode = new osg::Geode();
    wpTopGeode->setName("wpTopGeode");
    parent->addChild(wpTopGeode);
    wpTopGeode->addDrawable(wpTopGeom);
    wpTopGeom->dirtyDisplayList();

 /*   for (int i = 1; i < coVRAnimationManager::instance()->getNumTimesteps(); i++)
    {
        wpPrepareMillCut(wpTopGeom, static_cast<osg::Vec3Array*>(wpTopGeom->getVertexArray()), i);
    }
*/
//    auto testVec = wpTopGeom->getVertexArray();
//    auto testArray = dynamic_cast<osg::Vec3Array*>(testVec);
    //osgUtil::SmoothingVisitor::smooth(*wpTopGeom);
 /*   if (pointShader != nullptr)
    {
        pointShader->apply(m_currentGeode, m_pointCloud);
    }
    coVRAnimationManager::instance()->setNumTimesteps(dataTable.size(), this);
*/
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
        if (pathG[t] == 2 || pathG[t] == 3)
            wpAddQuadsG2G3(wpMaxZ, t, treeRoot);
        else
            wpAddQuadsG0G1(wpMaxZ, t, treeRoot);
    }
    treeRoot->traverseAndCallCC2Siblings();
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
        if (ixMin < 0)
            ixMin = 0;
        if (iyMin < 0)
            iyMin = 0;

        for (int iy = iyMin; iy < iyMax; iy++)
        {
            for (int ix = ixMin; ix < ixMax; ix++)
            {
                int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
                double wpQuadIXCenter = wpMinX + ix * wpResX + wpResX / 2;
                double wpQuadIYCenter = wpMinY + iy * wpResY + wpResY / 2;
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
        if (ixMin < 0)
            ixMin = 0;
        if (iyMin <= 0)
            iyMin = 0;

        if (ixMax > wpTotalQuadsX)
            ixMax = wpTotalQuadsX;
        if (iyMax > wpTotalQuadsY)
            iyMax = wpTotalQuadsY;

        double angleEnd = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t], pathY[t]);
        double angleStart = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t - 1], pathY[t - 1]);

        for (int iy = iyMin; iy < iyMax; iy++)
        {
            for (int ix = ixMin; ix < ixMax; ix++)
            {
                int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
                double wpQuadIXCenter = wpMinX + ix * wpResX + wpResX / 2;
                double wpQuadIYCenter = wpMinY + iy * wpResY + wpResY / 2;
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

/* wpTreeToGeometry

   Returned Value: topsurface Geometry

   Side Effects:
   Creates a Geometry: rectangular shape with multiple squares
   and uses wpResX, wpResY for length and wpTotalQuadsX, Y.

   Called By:
   CNCPlugin::createWorkpiece
*/
osg::ref_ptr<osg::Geometry> CNCPlugin::wpTreeToGeometry()
{
    //create geometry
    auto geo = new osg::Geometry();

    wpTopColors = new Vec4Array();
    auto points = new Vec3Array();
    wpTopPrimitives = new DrawArrayLengths(PrimitiveSet::QUADS);

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
            if (node->getMillTimesteps().size() == 0)
                for (int i = 0; i < 4; i++)
                    wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
            else
                for (int i = 0; i < 4; i++)
                    wpTopColors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.50f));
            auto tl = node->getTopLeft();
            auto br = node->getBotRight();
            wpAddVertexsForGeo(points, tl.x + 0, br.x, tl.y + 0, br.y, wpMaxZ);
        }
    }

    //wpTopColors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.50f));
    geo->setVertexArray(points);
    geo->setColorArray(wpTopColors);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    //geo->setColorBinding(osg::Geometry::BIND_OVERALL);

    geo->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    wpTopNormals = new osg::Vec3Array;
    wpTopNormals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
    //wpTopNormals->push_back(osg::Vec3(0.0f, 1.0f, 0.0f));
    //wpTopNormals->push_back(osg::Vec3(1.0f, 0.0f, 0.0f));
    geo->setNormalArray(wpTopNormals);// , osg::Array::BIND_OVERALL);

    wpTopPrimitives->push_back(points->size());
    wpTopPrimitives->setName("wpTopPrimitives");
    geo->addPrimitiveSet(wpTopPrimitives);

    geo->setStateSet(wpStateSet.get());
    geo->dirtyDisplayList();

    return geo;
}




/* createWpBottom

   Returned Value: bottom Geometry

   Side Effects:
   Creates a Geometry: rectangular shape as big as the given coords.

   Called By:
   CNCPlugin::createWpGeodes
*/
osg::ref_ptr<osg::Geometry> CNCPlugin::createWpBottom(double minX, double maxX, double minY, double maxY, double minZ, double maxZ)
{

    //create geometry
    auto geo = new osg::Geometry();
    //geo->setColorBinding(Geometry::BIND_OFF);
 //   geo->setUseDisplayList(false);
 //   geo->setSupportsDisplayList(false);
 //   geo->setUseVertexBufferObjects(true);
 //   auto vertexBufferArray = geo->getOrCreateVertexBufferObject();
    wpBotColors = new Vec4Array();
    wpBotNormals = new osg::Vec3Array;
    auto points = new Vec3Array();
    wpBotPrimitives = new DrawArrayLengths(PrimitiveSet::QUADS);

    points->push_back(Vec3(maxX, minY, minZ));
    points->push_back(Vec3(minX, minY, minZ));
    points->push_back(Vec3(minX, minY, maxZ));
    points->push_back(Vec3(maxX, minY, maxZ));
    for (int i = 0; i<4; i++)
        wpBotNormals->push_back(osg::Vec3(0.0f, 1.0f, 0.0f));
    points->push_back(Vec3(minX, maxY, minZ));
    points->push_back(Vec3(maxX, maxY, minZ));
    points->push_back(Vec3(maxX, maxY, maxZ));
    points->push_back(Vec3(minX, maxY, maxZ));
    for (int i = 0; i < 4; i++)
        wpBotNormals->push_back(osg::Vec3(0.0f, 1.0f, 0.0f));

    points->push_back(Vec3(minX, minY, minZ));
    points->push_back(Vec3(minX, maxY, minZ));
    points->push_back(Vec3(minX, maxY, maxZ));
    points->push_back(Vec3(minX, minY, maxZ));
    for (int i = 0; i < 4; i++)
        wpBotNormals->push_back(osg::Vec3(1.0f, 0.0f, 0.0f));
    points->push_back(Vec3(maxX, maxY, minZ));
    points->push_back(Vec3(maxX, minY, minZ));
    points->push_back(Vec3(maxX, minY, maxZ));
    points->push_back(Vec3(maxX, maxY, maxZ));
    for (int i = 0; i < 4; i++)
        wpBotNormals->push_back(osg::Vec3(1.0f, 0.0f, 0.0f));

    points->push_back(Vec3(minX, minY, minZ));
    points->push_back(Vec3(maxX, minY, minZ));
    points->push_back(Vec3(maxX, maxY, minZ));
    points->push_back(Vec3(minX, maxY, minZ));
    for (int i = 0; i < 4; i++)
        wpBotNormals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));

    wpBotPrimitives->push_back(points->size());
    wpBotColors->push_back(osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f));
    
    geo->setVertexArray(points);
    geo->setColorArray(wpBotColors);
    geo->setColorBinding(osg::Geometry::BIND_OVERALL);
    geo->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geo->setNormalArray(wpBotNormals);
    wpBotPrimitives->setName("wpBotPrimitives");
    geo->addPrimitiveSet(wpBotPrimitives);

    geo->setStateSet(wpStateSet.get());
    geo->dirtyDisplayList();
    //geo->setUseDisplayList(false);

    return geo;

}




/* createWpTopTree

   Returned Value: topsurface Geometry

   Side Effects:
   Creates a Geometry: rectangular shape with multiple squares
   and uses wpResX, wpResY for length and wpTotalQuadsX, Y.

   Called By:
   CNCPlugin::createWpGeodes
*/
osg::ref_ptr<osg::Geometry> CNCPlugin::createWpTopTree(double minX, double maxX, double minY, double maxY, double z)
{
    //create geometry
    auto geo = new osg::Geometry();
    //geo->setColorBinding(Geometry::BIND_OFF);
 //   geo->setUseDisplayList(false);
 //   geo->setSupportsDisplayList(false);
 //   geo->setUseVertexBufferObjects(true);
 //   auto vertexBufferArray = geo->getOrCreateVertexBufferObject();
    wpTopColors = new Vec4Array();
    auto points = new Vec3Array();
    wpTopPrimitives = new DrawArrayLengths(PrimitiveSet::QUADS);

/*    std::stack<TreeNode*> nodeStack;
    nodeStack.push(treeRoot);
    while (!nodeStack.empty())
    {
        // traversiere Baum
        TreeNode* node = nodeStack.top();
        nodeStack.pop();
        auto qwert = node->search(Point(1, 1));
        TreeNode* child1 = node->getTopLeftTree();
        TreeNode* child2 = node->getTopRightTree();
        TreeNode* child3 = node->getBotLeftTree();
        TreeNode* child4 = node->getBotRightTree();
        // 4 Children: alle 4 in Queue adden
        if (node->getNumChildren() == 4)
        {
            nodeStack.push(child4);
            nodeStack.push(child3);
            nodeStack.push(child2);
            nodeStack.push(child1);
        }
        auto tl = node->getTopLeft();
        auto br = node->getBotRight();
        // 3 Children: 3 in Queue adden, 4. als Quad setzen
        if (node->getNumChildren() == 3)
        {   
            if (child1 == nullptr)
            {   
                wpAddVertexsForGeo(points, tl.x +1, (tl.x+br.x)/2 +1, tl.y +1, (tl.y+br.y)/2 +1, z);
                nodeStack.push(child4);
                nodeStack.push(child3);
                nodeStack.push(child2);
            }
            else if (child2 == nullptr)
            {
                wpAddVertexsForGeo(points, (tl.x+br.x)/2 +1, br.x +1, tl.y +1, (tl.y+br.y)/2 +1, z);
                nodeStack.push(child4);
                nodeStack.push(child3);
                nodeStack.push(child1);
            }
            else if (child3 == nullptr)
            {
                wpAddVertexsForGeo(points, tl.x +1, (tl.x+br.x)/2 +1, (tl.y+br.y)/2 +1, br.y +1, z);
                nodeStack.push(child4);
                nodeStack.push(child2);
                nodeStack.push(child1);
            }
            else //child4 == NULL
            {
                wpAddVertexsForGeo(points, (tl.x+br.x)/2 +1, br.x +1, (tl.y+br.y)/2 +1, br.y +1, z);
                nodeStack.push(child3);
                nodeStack.push(child2);
                nodeStack.push(child1);
            }
            for (int j = 0; j < 4; j++)
                wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));

        }
        // 2 Children: 2 in Queue adden, checken ob boundary parent == boundary child1+child2, falls nicht 3.+4. als Quad setzen 
        else if (node->getNumChildren() == 2)
        {
            if (child1 != nullptr && child4 != nullptr)     //diagonal
            {
                nodeStack.push(child4);
                wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child3
                wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child2
                nodeStack.push(child1);
                for (int j = 0; j < 8; j++)
                    wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
            }
            else if (child2 != nullptr && child3 != nullptr)     //diagonal
            {
                wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child4
                nodeStack.push(child3);
                nodeStack.push(child2);
                wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child1
                for (int j = 0; j < 8; j++)
                    wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
            }
            else if (child1 != nullptr && child2 != nullptr)     //top
            {   
                nodeStack.push(child2);
                nodeStack.push(child1);
                if (br.y != child2->getBotRight().y)
                {
                    wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child4
                    wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child3
                    for (int j = 0; j < 8; j++)
                        wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                }
            }
            else if (child1 != nullptr && child3 != nullptr)     //left
            {
                nodeStack.push(child3);
                nodeStack.push(child1);
                if (br.x != child3->getBotRight().x)
                {
                    wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child4
                    wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child2
                    for (int j = 0; j < 8; j++)
                        wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                }
            }
            else if (child3 != nullptr && child4 != nullptr)     //bot
            {
                nodeStack.push(child4);
                nodeStack.push(child3);
                if (tl.y != child3->getTopLeft().y)
                {
                    wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child2
                    wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child1
                    for (int j = 0; j < 8; j++)
                        wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                }
            }
            else if (child2 != nullptr && child4 != nullptr)     //right
            {
                nodeStack.push(child4);
                nodeStack.push(child2);
                if (tl.x != child2->getTopLeft().x)
                {
                    wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child3
                    wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child1
                    for (int j = 0; j < 8; j++)
                        wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                }
            }
        }
        // 1 Child: 1 in Queue adden, boundary checken, 1bis3 Quads setzen
        else if (node->getNumChildren() == 1)
        {
            if (child4 != nullptr)
            {
                nodeStack.push(child4);
                if (tl.x != child4->getTopLeft().x)
                {
                    if (tl.y != child4->getTopLeft().y)
                    {
                        wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child3
                        wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child2
                        wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child1
                        for (int j = 0; j < 12; j++)
                            wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                    }
                    else
                    {
                        wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child3
                        for (int j = 0; j < 4; j++)
                            wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                    }
                }
                else
                {
                    wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child2
                    for (int j = 0; j < 4; j++)
                        wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                }
            }
            else if (child3 != nullptr)
            {
                nodeStack.push(child3);
                if (br.x != child3->getBotRight().x)
                {
                    if (tl.y != child3->getTopLeft().y)
                    {
                        wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child4
                        wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child2
                        wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child1
                        for (int j = 0; j < 12; j++)
                            wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                    }
                    else
                    {
                        wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child4
                        for (int j = 0; j < 4; j++)
                            wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                    }
                }
                else
                {
                    wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child1
                    for (int j = 0; j < 4; j++)
                        wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                }
            }
            else if (child2 != nullptr)
            {
                nodeStack.push(child2);
                if (tl.x != child2->getTopLeft().x)
                {
                    if (br.y != child2->getBotRight().y)
                    {
                        wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child4
                        wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child3
                        wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child1
                        for (int j = 0; j < 12; j++)
                            wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                    }
                    else
                    {
                        wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child1
                        for (int j = 0; j < 4; j++)
                            wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                    }
                }
                else
                {
                    wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child4
                    for (int j = 0; j < 4; j++)
                        wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                }
            }
            else if (child1 != nullptr)
            {
                nodeStack.push(child1);
                if (br.x != child1->getBotRight().x)
                {
                    if (br.y != child1->getBotRight().y)
                    {
                        wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child4
                        wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child3
                        wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child2
                        for (int j = 0; j < 12; j++)
                            wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                    }
                    else
                    {
                        wpAddVertexsForGeo(points, (tl.x + br.x) / 2 + 1, br.x + 1, tl.y + 1, (tl.y + br.y) / 2 + 1, z);    //child2
                        for (int j = 0; j < 4; j++)
                            wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                    }
                }
                else
                {
                    wpAddVertexsForGeo(points, tl.x + 1, (tl.x + br.x) / 2 + 1, (tl.y + br.y) / 2 + 1, br.y + 1, z);    //child3
                    for (int j = 0; j < 4; j++)
                        wpTopColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 0.50f));
                }
            }
        }
  
        // 0 Child: error?
        // -1Child: Blattknoten, in Queue adden
        else if (node->getNumChildren() == -1)
        {   
            node->setPrimitivePos(primitivePosCounter);
            wpAddVertexsForGeo(points, tl.x +1, br.x +1, tl.y +1, br.y +1, z);
            for (int j = 0; j < 4; j++)
                wpTopColors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.50f));
        }
    }
    // VerticalPrimitives setzen
*/
  /*  for (int iy = 0; iy < wpTotalQuadsY; iy++)
    {
        for (int ix = 0; ix < wpTotalQuadsX; ix++)
        {
            test1 = ix;
            test2 = iy;
            points->push_back(Vec3(minX + ix * wpResX, minY + iy * wpResY, z));
            points->push_back(Vec3(minX + (ix + 1) * wpResX, minY + iy * wpResY, z));
            points->push_back(Vec3(minX + (ix + 1) * wpResX, minY + (iy + 1) * wpResY, z));
            points->push_back(Vec3(minX + ix * wpResX, minY + (iy + 1) * wpResY, z));
            cuttedFaces.push_back(-1);
        }
    }
*//*
    wpTopPrimitives->push_back(points->size());
*/
    //wpTopColors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    //wpTopColors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    //wpTopColors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

        /*
            //calculate coords and color
            std::vector<float> scalarData(symbols.size());
            float minScalar = 0, maxScalar = 0;
            for (size_t i = 0; i < symbols.size(); i++)
            {
                osg::Vec3 coords;
                for (size_t j = 0; j < 3; j++)
                {
                    coords[j] = stringExpressions[j]().value();
                }
                points->push_back(coords);
                scalarData[i] = stringExpressions[3]().value();
                minScalar = std::min(minScalar, scalarData[i]);
                maxScalar = std::max(maxScalar, scalarData[i]);
                symbols.advance();
            }

            for (size_t i = 0; i < symbols.size(); i++)
                colors->push_back(m_colorMapSelector.getColor(scalarData[i], minScalar, maxScalar));

            osg::Vec3 bottomLeft, hpr, offset;
            if (coVRMSController::instance()->isMaster() && coVRConfig::instance()->numScreens() > 0) {
                auto hudScale = covise::coCoviseConfig::getFloat("COVER.Plugin.ColorBar.HudScale", 0.5); // half screen height
                const auto& s0 = coVRConfig::instance()->screens[0];
                hpr = s0.hpr;
                auto sz = osg::Vec3(s0.hsize, 0., s0.vsize);
                osg::Matrix mat;
                MAKE_EULER_MAT_VEC(mat, hpr);
                bottomLeft = s0.xyz - sz * mat * 0.5;
                auto minsize = std::min(s0.hsize, s0.vsize);
                bottomLeft += osg::Vec3(minsize, 0., minsize) * mat * 0.02;
                offset = osg::Vec3(s0.vsize / 2.5, 0, 0) * mat * hudScale;
            }
            m_colorBar->setName("Power");
            m_colorBar->show(true);
            m_colorBar->update(m_colorTerm->value(), minScalar, maxScalar, m_colorMapSelector.selectedMap().a.size(), m_colorMapSelector.selectedMap().r.data(), m_colorMapSelector.selectedMap().g.data(), m_colorMapSelector.selectedMap().b.data(), m_colorMapSelector.selectedMap().a.data());
            m_colorBar->setHudPosition(bottomLeft, hpr, offset[0] / 480);
            m_colorBar->show(true);
        */

 /*       //vertexBufferArray->setArray(0, points);
        //vertexBufferArray->setArray(1, colors);
        //points->setBinding(osg::Array::BIND_PER_VERTEX);
        //wpTopColors->setBinding(osg::Array::BIND_PER_VERTEX);
        // bind color per vertex
    geo->setVertexArray(points);
    geo->setColorArray(wpTopColors);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    //geo->setColorBinding(osg::Geometry::BIND_OVERALL);

    geo->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    wpTopNormals = new osg::Vec3Array;
    wpTopNormals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
    wpTopNormals->push_back(osg::Vec3(0.0f, 1.0f, 0.0f));
    wpTopNormals->push_back(osg::Vec3(1.0f, 0.0f, 0.0f));
    geo->setNormalArray(wpTopNormals);// , osg::Array::BIND_OVERALL);



    test1 = points->getNumElements();
    test2 = points->size();
    wpTopPrimitives->setName("wpTopPrimitives");
    geo->addPrimitiveSet(wpTopPrimitives);

    wpVerticalPrimitivesX = new DrawElementsUInt(PrimitiveSet::QUADS);
    wpVerticalPrimitivesX->push_back(0);
    wpVerticalPrimitivesX->push_back(1);
    wpVerticalPrimitivesX->push_back(2);
    wpVerticalPrimitivesX->push_back(3);
    wpVerticalPrimitivesY = new DrawElementsUInt(PrimitiveSet::QUADS);
    wpVerticalPrimitivesY->push_back(0);
    wpVerticalPrimitivesY->push_back(1);
    wpVerticalPrimitivesY->push_back(2);
    wpVerticalPrimitivesY->push_back(3);

    wpAddFacesTree();
   
    wpVerticalPrimitivesX->setName("wpVerticalPrimitivesX");
    wpVerticalPrimitivesY->setName("wpVerticalPrimitivesY");
    geo->addPrimitiveSet(wpVerticalPrimitivesX);
    geo->addPrimitiveSet(wpVerticalPrimitivesY);

    //setStateSet(geo, pointSize());
    geo->setStateSet(wpStateSet.get());
    geo->dirtyDisplayList();
    //geo->setUseDisplayList(false);
*/
    return geo;

}

void CNCPlugin::setWpSize()
{
    wpMinX = *std::min_element(pathX.begin(), pathX.end()-1) - wpAllowance;
    wpMaxX = *std::max_element(pathX.begin(), pathX.end()-1) + wpAllowance;
    wpMinY = *std::min_element(pathY.begin(), pathY.end()-1) - wpAllowance;
    wpMaxY = *std::max_element(pathY.begin(), pathY.end()-1) + wpAllowance;
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
    wpLengthX = wpMaxX - wpMinX;
    wpLengthY = wpMaxY - wpMinY;
    wpLengthZ = wpMaxZ - wpMinZ;
}

void CNCPlugin::setWpResolution()
{
    wpTotalQuadsX = ::round(wpLengthX / wpResolution);
    wpResX = wpLengthX / wpTotalQuadsX;

    wpTotalQuadsY = ::round(wpLengthY / wpResolution);
    wpResY = wpLengthY / wpTotalQuadsY;
}

void CNCPlugin::setWpMaterial()
{
    wpStateSet = new osg::StateSet();
    wpMaterial = new osg::Material;
    wpLineWidth = new osg::LineWidth(2.0);
    wpMaterial.get()->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    wpMaterial.get()->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    wpMaterial.get()->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(0.80f, 0.80f, 0.80f, 1.0));
    wpMaterial.get()->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(0.5f, 0.5f, 0.5f, 1.0));
    wpMaterial.get()->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    wpMaterial.get()->setShininess(osg::Material::FRONT_AND_BACK, 8.0f);
    
    wpStateSet->setAttributeAndModes(wpMaterial.get(), StateAttribute::ON);

    //wpStateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    wpStateSet->setAttributeAndModes(wpLineWidth.get(), StateAttribute::ON); //nur fur Linien. Ueberflussig?

    osg::ref_ptr<osg::PolygonMode> polyMode(new osg::PolygonMode());
    polyMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL); //LINE);
    wpStateSet->setAttribute(polyMode.get());
}



/* wpMillCut

   Side Effects:
   changes the Geometry and sets new zCoords for timeStep t.

   Called By:
   CNCPlugin::setTimestep
*/
void CNCPlugin::wpMillCut(osg::Geometry *geo, osg::Vec3Array *piece, int t)
{   
 /*   //bounding Box
    double boxMinX = std::min(pathX[t - 1], pathX[t]) - cuttingRad;
    double boxMaxX = std::max(pathX[t - 1], pathX[t]) + cuttingRad;
    double boxMinY = std::min(pathY[t - 1], pathY[t]) - cuttingRad;
    double boxMaxY = std::max(pathY[t - 1], pathY[t]) + cuttingRad;

    int ixMin = (boxMinX - wpMinX) / wpResX;
    int ixMax = (boxMaxX - wpMinX) / wpResX;
    int iyMin = (boxMinY - wpMinY) / wpResY;
    int iyMax = (boxMaxY - wpMinY) / wpResY;
    if (ixMin < 0)              //for last move, usually to (0,0,0)
        ixMin = 0;
    if (iyMin < 0)
        iyMin = 0;


    for (int iy = iyMin; iy <= iyMax; iy++)
    {   
        for (int ix = ixMin; ix <= ixMax; ix++)
        {   
            int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
            double dist = distancePointLineSegment(piece->at(iPoint)[0] + wpResX/2, piece->at(iPoint)[1] + wpResY / 2, pathX[t - 1], pathY[t - 1], pathX[t], pathY[t]);
            if (dist < cuttingRad && pathZ[t] < piece->at(iPoint)[2])
            {   
                //cuttedQuadsIX.push_back(ix);
                //cuttedQuadsIY.push_back(iy);
                piece->at(iPoint)[2] = pathZ[t];  // unpräzise bezüglich Höhe Z. tatsächliche Fräserhöhe an Stelle piece-at(i) eventuell abweichend!
                piece->at(iPoint+1)[2] = pathZ[t];
                piece->at(iPoint+2)[2] = pathZ[t];
                piece->at(iPoint+3)[2] = pathZ[t];
            }
        }
    }
    wpCutFaces(geo, piece);
*/
}

/* wpMillCutTree

   Side Effects:
   changes the Geometry and sets new zCoords for timeStep t.

   Called By:
   CNCPlugin::setTimestep
*/
void CNCPlugin::wpMillCutTree(osg::Geometry* geo, osg::Vec3Array* piece, int t)
{
/*    //bounding Box
    double boxMinX = std::min(pathX[t - 1], pathX[t]) - cuttingRad;
    double boxMaxX = std::max(pathX[t - 1], pathX[t]) + cuttingRad;
    double boxMinY = std::min(pathY[t - 1], pathY[t]) - cuttingRad;
    double boxMaxY = std::max(pathY[t - 1], pathY[t]) + cuttingRad;

    int ixMin = (boxMinX - wpMinX) / wpResX;
    int ixMax = (boxMaxX - wpMinX) / wpResX;
    int iyMin = (boxMinY - wpMinY) / wpResY;
    int iyMax = (boxMaxY - wpMinY) / wpResY;
    if (ixMin < 0)              //for last move, usually to (0,0,0)
        ixMin = 0;
    if (iyMin < 0)
        iyMin = 0;


    for (int iy = iyMin; iy <= iyMax; iy++)
    {
        for (int ix = ixMin; ix <= ixMax; ix++)
        {
            //int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
            auto coords = treeRoot->search(Point(ix, iy));
            if (coords != nullptr)
            {
                int iPoint = coords->primitivePos;
                double dist = distancePointLineSegment(piece->at(iPoint)[0] + wpResX / 2, piece->at(iPoint)[1] + wpResY / 2, pathX[t - 1], pathY[t - 1], pathX[t], pathY[t]);
                if (dist < cuttingRad)// && pathZ[t] < piece->at(iPoint)[2])
                {   
                    double zCut = pathZ[t] - tan(pointAngle / 2) * dist;
                    if (pathZ[t] < piece->at(iPoint)[2]) //if (zCut < piece->at(iPoint)[2])
                    {
                        //cuttedQuadsIX.push_back(ix);
                        //cuttedQuadsIY.push_back(iy);
                        piece->at(iPoint)[2] = pathZ[t]; // zCut;//   // unpräzise bezüglich Höhe Z. tatsächliche Fräserhöhe an Stelle piece-at(i) eventuell abweichend!
                        piece->at(iPoint + 1)[2] = pathZ[t]; // zCut;
                        piece->at(iPoint + 2)[2] = pathZ[t]; // zCut;
                        piece->at(iPoint + 3)[2] = pathZ[t]; // zCut;
                    }
                }
            }
        }
    }
    //wpCutFaces(geo, piece);
*/
}

void CNCPlugin::wpPrepareMillCut(osg::Geometry *geo, osg::Vec3Array *piece, int t)
{
/*    //bounding Box
    double boxMinX = std::min(pathX[t - 1], pathX[t]) - cuttingRad;
    double boxMaxX = std::max(pathX[t - 1], pathX[t]) + cuttingRad;
    double boxMinY = std::min(pathY[t - 1], pathY[t]) - cuttingRad;
    double boxMaxY = std::max(pathY[t - 1], pathY[t]) + cuttingRad;

    int ixMin = (boxMinX - wpMinX) / wpResX;
    int ixMax = (boxMaxX - wpMinX) / wpResX;
    int iyMin = (boxMinY - wpMinY) / wpResY;
    int iyMax = (boxMaxY - wpMinY) / wpResY;
    if (ixMin < 0)
        ixMin = 0;
    if (iyMin < 0)
        iyMin = 0;

    for (int iy = iyMin; iy <= iyMax; iy++)
    {
        for (int ix = ixMin; ix <= ixMax; ix++)
        {
            int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
            double dist = distancePointLineSegment(piece->at(iPoint)[0] + wpResX / 2, piece->at(iPoint)[1] + wpResY / 2, pathX[t - 1], pathY[t - 1], pathX[t], pathY[t]);
            if (dist < cuttingRad && pathZ[t] < piece->at(iPoint)[2])
            {
                cuttedQuadsIX.push_back(ix);
                cuttedQuadsIY.push_back(iy);
                piece->at(iPoint)[2] = pathZ[t];  // unpräzise bezüglich Höhe Z. tatsächliche Fräserhöhe an Stelle piece-at(i) eventuell abweichend!
                piece->at(iPoint + 1)[2] = pathZ[t];
                piece->at(iPoint + 2)[2] = pathZ[t];
                piece->at(iPoint + 3)[2] = pathZ[t];
            }
        }
    }
    wpCutFaces(geo, piece);
*/
}

void CNCPlugin::wpPrepareMillCutTree(double minX, double maxX, double minY, double maxY, double z, int t)
{
/*    //bounding Box
    double boxMinX = std::min(pathX[t - 1], pathX[t]) - cuttingRad;
    double boxMaxX = std::max(pathX[t - 1], pathX[t]) + cuttingRad;
    double boxMinY = std::min(pathY[t - 1], pathY[t]) - cuttingRad;
    double boxMaxY = std::max(pathY[t - 1], pathY[t]) + cuttingRad;

    int ixMin = (boxMinX - wpMinX) / wpResX;
    int ixMax = (boxMaxX - wpMinX) / wpResX;
    int iyMin = (boxMinY - wpMinY) / wpResY;
    int iyMax = (boxMaxY - wpMinY) / wpResY;
    if (ixMin < 0)
        ixMin = 0;
    if (iyMin < 0)
        iyMin = 0;
    if (pathZ[t] < z)
    {
        for (int iy = iyMin; iy <= iyMax; iy++)
        {
            for (int ix = ixMin; ix <= ixMax; ix++)
            {
                int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
                double wpQuadIXCenter = minX + ix * wpResX + wpResX / 2;
                double wpQuadIYCenter = minY + iy * wpResY + wpResY / 2;
                double dist = distancePointLineSegment(wpQuadIXCenter, wpQuadIYCenter, pathX[t - 1], pathY[t - 1], pathX[t], pathY[t]);
                if (dist < cuttingRad) // && pathZ[t] < z)
                {
                    cuttedQuadsIX.push_back(ix);
                    cuttedQuadsIY.push_back(iy);

                    treeRoot->insert(ix, iy, z);
                    treeRoot->insert(ix - 1, iy, z);
                    treeRoot->insert(ix, iy - 1, z);
                    treeRoot->insert(ix + 1, iy, z);
                    treeRoot->insert(ix, iy + 1, z);

                    treeRoot->updateZ(ix, iy, pathZ[t]);
*/
                    /*    piece->at(iPoint)[2] = pathZ[t];  // unpräzise bezüglich Höhe Z. tatsächliche Fräserhöhe an Stelle piece-at(i) eventuell abweichend!
                        piece->at(iPoint + 1)[2] = pathZ[t];
                        piece->at(iPoint + 2)[2] = pathZ[t];
                        piece->at(iPoint + 3)[2] = pathZ[t];
                    */
 /*               }
            }
        }
        wpCutFacesTree(minX, maxX, minY, maxY, z);
    }
    */
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
/*
void CNCPlugin::wpResetCuts(osg::Vec3Array *piece, int t)
{
    for (int i = 0; i < piece->getNumElements(); i++)
    {
        piece->at(i)[2] = wpMaxZ;
    }
}*/
/*
void CNCPlugin::wpCutFaces(osg::Geometry *geo, osg::Vec3Array *piece)
{
    while (!cuttedQuadsIX.empty())
    {
        int ix = cuttedQuadsIX.back();
        int iy = cuttedQuadsIY.back();
        int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
        double zPoint = piece->at(iPoint)[2];

        //Neighbor 1, left/west
        int nb = (ix - 1) * 4 + iy * wpTotalQuadsX * 4;
        double zNb = piece->at(nb)[2];
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[iPoint/4] == 1 || cuttedFaces[iPoint/4] == 5)
            { 
            }
            else
            {
                //add Vertical Quad
                wpVerticalPrimitivesX->push_back(iPoint + 3);
                wpVerticalPrimitivesX->push_back(iPoint);
                wpVerticalPrimitivesX->push_back(nb + 1);
                wpVerticalPrimitivesX->push_back(nb + 2);
                if (cuttedFaces[iPoint / 4] == 2)
                    cuttedFaces[iPoint / 4] = 5;
                else
                    cuttedFaces[iPoint / 4] = 1;
            }

        }

        //Neighbor 2, down/south
        nb = ix * 4 + (iy - 1) * wpTotalQuadsX * 4;
        zNb = piece->at(nb)[2];
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[iPoint / 4] == 2 || cuttedFaces[iPoint / 4] == 5)
            {
            }
            else
            {
                //add Vertical Quad
                wpVerticalPrimitivesY->push_back(iPoint);
                wpVerticalPrimitivesY->push_back(iPoint + 1);
                wpVerticalPrimitivesY->push_back(nb + 2);
                wpVerticalPrimitivesY->push_back(nb + 3);
                if (cuttedFaces[iPoint / 4] == 1)
                    cuttedFaces[iPoint / 4] = 5;
                else
                    cuttedFaces[iPoint / 4] = 2;
            }

        }

        //Neighbor 3, right/east
        nb = (ix + 1) * 4 + iy * wpTotalQuadsX * 4;
        zNb = piece->at(nb)[2];
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[nb / 4] == 1 || cuttedFaces[nb / 4] == 5)
            {
            }
            else
            {
                //add Vertical Quad
                wpVerticalPrimitivesX->push_back(iPoint + 1);
                wpVerticalPrimitivesX->push_back(iPoint + 2);
                wpVerticalPrimitivesX->push_back(nb + 3);
                wpVerticalPrimitivesX->push_back(nb);
                if (cuttedFaces[nb / 4] == 2)
                    cuttedFaces[nb / 4] = 5;
                else
                    cuttedFaces[nb / 4] = 1;
            }

        }

        //Neighbor 4, up/nord
        nb = ix * 4 + (iy + 1) * wpTotalQuadsX * 4;
        zNb = piece->at(nb)[2];
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[nb / 4] == 2 || cuttedFaces[nb / 4] == 5)
            {
            }
            else
            {
                //add Vertical Quad
                wpVerticalPrimitivesY->push_back(iPoint + 2);
                wpVerticalPrimitivesY->push_back(iPoint + 3);
                wpVerticalPrimitivesY->push_back(nb);
                wpVerticalPrimitivesY->push_back(nb + 1);
                if (cuttedFaces[nb / 4] == 1)
                    cuttedFaces[nb / 4] = 5;
                else
                    cuttedFaces[nb / 4] = 2;
            }

        }
*/
        /*
        for (int i = 1; i < 5; i++)
        {
            //get NeighborQuads, can not be a edge quad
            if (i==1)
                int nb = (ix - 1) * 4 + iy * wpTotalQuadsX * 4;
            else if (i==2)
                int nb = ix * 4 + (iy - 1) * wpTotalQuadsX * 4;
            else if (i==3)
                int nb = (ix + 1) * 4 + iy * wpTotalQuadsX * 4;
            else if (i==4)
                int nb = ix * 4 + (iy + 1) * wpTotalQuadsX * 4;

            //get zHeight 
            double zNb = piece->at(nb)[2];

            //check for height difference
            if (zPoint < zNb)
            {
                //check if vertical Quad exists

                //add Vertical Quads
                wpVerticalPrimitives->push_back(iPoint);
                wpVerticalPrimitives->push_back(iPoint + 3);
                wpVerticalPrimitives->push_back(nb + 2);
                wpVerticalPrimitives->push_back(nb + 1);
                cuttedFaces[iPoint / 4] = 1;

            }
        }
        */

/*
        //remove vertical Quads?


        cuttedQuadsIX.pop_back();
        cuttedQuadsIY.pop_back();
    }
}
*/
/*
void CNCPlugin::wpCutFacesTree(double minX, double maxX, double minY, double maxY, double z)
{
    while (!cuttedQuadsIX.empty())
    {
        int ix = cuttedQuadsIX.back();
        int iy = cuttedQuadsIY.back();
        int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
        double zPoint = treeRoot->search(Point(ix, iy))->z;

        //Neighbor 1, left/west
        int nb = (ix - 1) * 4 + iy * wpTotalQuadsX * 4;
        double zNb = treeRoot->search(Point(ix - 1, iy))->z;
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[iPoint / 4] == 1 || cuttedFaces[iPoint / 4] == 5)
            {
            }
            else
            {
*/            /*    //add Vertical Quad
                wpVerticalPrimitivesX->push_back(iPoint + 3);
                wpVerticalPrimitivesX->push_back(iPoint);
                wpVerticalPrimitivesX->push_back(nb + 1);
                wpVerticalPrimitivesX->push_back(nb + 2);
            */
/*                if (cuttedFaces[iPoint / 4] == 2)
                    cuttedFaces[iPoint / 4] = 5;
                else
                    cuttedFaces[iPoint / 4] = 1;
            }

        }

        //Neighbor 2, down/south
        nb = ix * 4 + (iy - 1) * wpTotalQuadsX * 4;
        zNb = treeRoot->search(Point(ix, iy - 1))->z;
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[iPoint / 4] == 2 || cuttedFaces[iPoint / 4] == 5)
            {
            }
            else
            {
*/            /*    //add Vertical Quad
                wpVerticalPrimitivesY->push_back(iPoint);
                wpVerticalPrimitivesY->push_back(iPoint + 1);
                wpVerticalPrimitivesY->push_back(nb + 2);
                wpVerticalPrimitivesY->push_back(nb + 3);
            */
/*                if (cuttedFaces[iPoint / 4] == 1)
                    cuttedFaces[iPoint / 4] = 5;
                else
                    cuttedFaces[iPoint / 4] = 2;
            }

        }

        //Neighbor 3, right/east
        nb = (ix + 1) * 4 + iy * wpTotalQuadsX * 4;
        zNb = treeRoot->search(Point(ix + 1, iy))->z;
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[nb / 4] == 1 || cuttedFaces[nb / 4] == 5)
            {
            }
            else
            {
 */           /*    //add Vertical Quad
                wpVerticalPrimitivesX->push_back(iPoint + 1);
                wpVerticalPrimitivesX->push_back(iPoint + 2);
                wpVerticalPrimitivesX->push_back(nb + 3);
                wpVerticalPrimitivesX->push_back(nb);
            */
 /*               if (cuttedFaces[nb / 4] == 2)
                    cuttedFaces[nb / 4] = 5;
                else
                    cuttedFaces[nb / 4] = 1;
            }

        }

        //Neighbor 4, up/nord
        nb = ix * 4 + (iy + 1) * wpTotalQuadsX * 4;
        zNb = treeRoot->search(Point(ix, iy + 1))->z;
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[nb / 4] == 2 || cuttedFaces[nb / 4] == 5)
            {
            }
            else
            {
  */          /*    //add Vertical Quad
                wpVerticalPrimitivesY->push_back(iPoint + 2);
                wpVerticalPrimitivesY->push_back(iPoint + 3);
                wpVerticalPrimitivesY->push_back(nb);
                wpVerticalPrimitivesY->push_back(nb + 1);
            */
  /*              if (cuttedFaces[nb / 4] == 1)
                    cuttedFaces[nb / 4] = 5;
                else
                    cuttedFaces[nb / 4] = 2;
            }

        }
  
        //remove vertical Quads?

        cuttedQuadsIX.pop_back();
        cuttedQuadsIY.pop_back();
    }
}
*/

void CNCPlugin::wpAddVertexsForGeo(osg::Vec3Array* points, int minIX, int maxIX, int minIY, int maxIY, double z)
{
    points->push_back(Vec3(wpMinX + minIX * wpResX, wpMinY + minIY * wpResY, z));
    points->push_back(Vec3(wpMinX + maxIX * wpResX, wpMinY + minIY * wpResY, z));
    points->push_back(Vec3(wpMinX + maxIX * wpResX, wpMinY + maxIY * wpResY, z));
    points->push_back(Vec3(wpMinX + minIX * wpResX, wpMinY + maxIY * wpResY, z));
    primitivePosCounter += 4;
    return;
}

/*
void CNCPlugin::wpAddFacesTree()
{   
    for (int iy = 0; iy < wpTotalQuadsY; iy++)
    {
        for (int ix = 0; ix < wpTotalQuadsX; ix++)
        {   
            int i = ix + iy * wpTotalQuadsX;
            if (cuttedFaces[i] == -1)
            {
            }
            else if (cuttedFaces[i] == 1)
            {
                int primPosI = treeRoot->search(Point(ix, iy))->primitivePos;
                int primPosNb = treeRoot->search(Point(ix - 1, iy))->primitivePos;
                wpVerticalPrimitivesX->push_back(primPosI + 3);
                wpVerticalPrimitivesX->push_back(primPosI);
                wpVerticalPrimitivesX->push_back(primPosNb + 1);
                wpVerticalPrimitivesX->push_back(primPosNb + 2);
            }
            else if (cuttedFaces[i] == 2)
            {
                int primPosI = treeRoot->search(Point(ix, iy))->primitivePos;
                int primPosNb = treeRoot->search(Point(ix, iy - 1))->primitivePos;
                wpVerticalPrimitivesY->push_back(primPosI);
                wpVerticalPrimitivesY->push_back(primPosI + 1);
                wpVerticalPrimitivesY->push_back(primPosNb + 2);
                wpVerticalPrimitivesY->push_back(primPosNb + 3);
            }
            else if (cuttedFaces[i] == 5)
            {   
                int primPosI = treeRoot->search(Point(ix, iy))->primitivePos;
                int primPosNb = treeRoot->search(Point(ix - 1, iy))->primitivePos;
                wpVerticalPrimitivesX->push_back(primPosI + 3);
                wpVerticalPrimitivesX->push_back(primPosI);
                wpVerticalPrimitivesX->push_back(primPosNb + 1);
                wpVerticalPrimitivesX->push_back(primPosNb + 2);
                primPosI = treeRoot->search(Point(ix, iy))->primitivePos;
                primPosNb = treeRoot->search(Point(ix, iy - 1))->primitivePos;
                wpVerticalPrimitivesY->push_back(primPosI);
                wpVerticalPrimitivesY->push_back(primPosI + 1);
                wpVerticalPrimitivesY->push_back(primPosNb + 2);
                wpVerticalPrimitivesY->push_back(primPosNb + 3);
            }
        }
    }
  */       
  /*
    while (!cuttedQuadsIX.empty())
    {
        int ix = cuttedQuadsIX.back();
        int iy = cuttedQuadsIY.back();
        int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
        double zPoint = piece->at(iPoint)[2];

        //Neighbor 1, left/west
        int nb = (ix - 1) * 4 + iy * wpTotalQuadsX * 4;
        double zNb = piece->at(nb)[2];
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[iPoint / 4] == 1 || cuttedFaces[iPoint / 4] == 5)
            {
            }
            else
            {
                //add Vertical Quad
                wpVerticalPrimitivesX->push_back(iPoint + 3);
                wpVerticalPrimitivesX->push_back(iPoint);
                wpVerticalPrimitivesX->push_back(nb + 1);
                wpVerticalPrimitivesX->push_back(nb + 2);
                if (cuttedFaces[iPoint / 4] == 2)
                    cuttedFaces[iPoint / 4] = 5;
                else
                    cuttedFaces[iPoint / 4] = 1;
            }

        }

        //Neighbor 2, down/south
        nb = ix * 4 + (iy - 1) * wpTotalQuadsX * 4;
        zNb = piece->at(nb)[2];
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[iPoint / 4] == 2 || cuttedFaces[iPoint / 4] == 5)
            {
            }
            else
            {
                //add Vertical Quad
                wpVerticalPrimitivesY->push_back(iPoint);
                wpVerticalPrimitivesY->push_back(iPoint + 1);
                wpVerticalPrimitivesY->push_back(nb + 2);
                wpVerticalPrimitivesY->push_back(nb + 3);
                if (cuttedFaces[iPoint / 4] == 1)
                    cuttedFaces[iPoint / 4] = 5;
                else
                    cuttedFaces[iPoint / 4] = 2;
            }

        }

        //Neighbor 3, right/east
        nb = (ix + 1) * 4 + iy * wpTotalQuadsX * 4;
        zNb = piece->at(nb)[2];
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[nb / 4] == 1 || cuttedFaces[nb / 4] == 5)
            {
            }
            else
            {
                //add Vertical Quad
                wpVerticalPrimitivesX->push_back(iPoint + 1);
                wpVerticalPrimitivesX->push_back(iPoint + 2);
                wpVerticalPrimitivesX->push_back(nb + 3);
                wpVerticalPrimitivesX->push_back(nb);
                if (cuttedFaces[nb / 4] == 2)
                    cuttedFaces[nb / 4] = 5;
                else
                    cuttedFaces[nb / 4] = 1;
            }

        }

        //Neighbor 4, up/nord
        nb = ix * 4 + (iy + 1) * wpTotalQuadsX * 4;
        zNb = piece->at(nb)[2];
        if (zPoint < zNb)
        {
            //check if vertical Quad exists
            if (cuttedFaces[nb / 4] == 2 || cuttedFaces[nb / 4] == 5)
            {
            }
            else
            {
                //add Vertical Quad
                wpVerticalPrimitivesY->push_back(iPoint + 2);
                wpVerticalPrimitivesY->push_back(iPoint + 3);
                wpVerticalPrimitivesY->push_back(nb);
                wpVerticalPrimitivesY->push_back(nb + 1);
                if (cuttedFaces[nb / 4] == 1)
                    cuttedFaces[nb / 4] = 5;
                else
                    cuttedFaces[nb / 4] = 2;
            }

        }

    }
  */
//}


// Quad Tree


/*

TreeNode* CNCPlugin::createTree(int minIX, int maxIX, int minIY, int maxIY, double z)
{
    treeRoot = new TreeNode(Point(minIX, minIY), Point(maxIX, maxIY),0);
    for (int i = 0; i < 5; i++)
    {
        try {
            //MillCoords* _coords = new MillCoords(Point(i, i), i * i);  // wird uberschrieben, nur ein Millcoords Objekt!
            //treeRoot->insert(_coords);
            treeRoot->insert(new MillCoords(Point(i, i), i*i));
        }
        catch (const std::invalid_argument& e) {
            cout << "invalid_argument: " << i << "\n";
        }
        MillCoords* _coords2 = new MillCoords(Point(i+10, i+10), i * i);  // wird uberschrieben, nur ein Millcoords Objekt!
        //treeRoot->insert(_coords2);
        treeRoot->insert(i + 10, i + 10, i * i);
    }
    return treeRoot;
}


void CNCPlugin::wpPrepareMillCutTreeCircle(double minX, double maxX, double minY, double maxY, double z, int t)
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
        if (ixMin <= 1)
            ixMin = 2;
        if (iyMin <= 1)
            iyMin = 2;

        if (ixMax >= wpTotalQuadsX)
            ixMax = wpTotalQuadsX-1;
        if (iyMax >= wpTotalQuadsY)
            iyMax = wpTotalQuadsY-1;

        double angleEnd = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t], pathY[t]);
        double angleStart = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t-1], pathY[t-1]);

        for (int iy = iyMin; iy <= iyMax; iy++)
        {
            for (int ix = ixMin; ix <= ixMax; ix++)
            {
                int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
                double wpQuadIXCenter = minX + ix * wpResX + wpResX / 2;
                double wpQuadIYCenter = minY + iy * wpResY + wpResY / 2;
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
                        cuttedQuadsIX.push_back(ix);
                        cuttedQuadsIY.push_back(iy);

                        treeRoot->insert(ix, iy, z);
                        treeRoot->insert(ix - 1, iy, z);
                        treeRoot->insert(ix, iy - 1, z);
                        treeRoot->insert(ix + 1, iy, z);
                        treeRoot->insert(ix, iy + 1, z);

                        treeRoot->updateZ(ix, iy, pathZ[t]);

 */                       /*    piece->at(iPoint)[2] = pathZ[t];  // unpräzise bezüglich Höhe Z. tatsächliche Fräserhöhe an Stelle piece-at(i) eventuell abweichend!
                            piece->at(iPoint + 1)[2] = pathZ[t];
                            piece->at(iPoint + 2)[2] = pathZ[t];
                            piece->at(iPoint + 3)[2] = pathZ[t];
                        */
 /*                   }
                }
            }
        }
        wpCutFacesTree(minX, maxX, minY, maxY, z);
    }
}
*/
/*
void CNCPlugin::wpMillCutTreeCircle(osg::Geometry* geo, osg::Vec3Array* piece, int t)
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
    if (ixMin < 0)              //for last move, usually to (0,0,0)
        ixMin = 0;
    if (iyMin < 0)
        iyMin = 0;

    if (ixMax >= wpTotalQuadsX)
        ixMax = wpTotalQuadsX-1;
    if (iyMax >= wpTotalQuadsY)
        iyMax = wpTotalQuadsY-1;

    double angleEnd = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t], pathY[t]);
    double angleStart = anglePointPoint(pathCenterX[t], pathCenterY[t], pathX[t - 1], pathY[t - 1]);

    for (int iy = iyMin; iy <= iyMax; iy++)
    {
        for (int ix = ixMin; ix <= ixMax; ix++)
        {
            //int iPoint = ix * 4 + iy * wpTotalQuadsX * 4;
            auto coords = treeRoot->search(Point(ix, iy));
            if (coords != nullptr)
            {
                int iPoint = coords->primitivePos;

                double wpQuadIXCenter = piece->at(iPoint)[0] + wpResX / 2;
                double wpQuadIYCenter = piece->at(iPoint)[1] + wpResX / 2;

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

                    // double zCut = pathZ[t] - tan(pointAngle / 2) * dist;
                    if (insert && pathZ[t] < piece->at(iPoint)[2]) //if (zCut < piece->at(iPoint)[2])
                    {
                        //cuttedQuadsIX.push_back(ix);
                        //cuttedQuadsIY.push_back(iy);
                        piece->at(iPoint)[2] = pathZ[t]; // zCut;//   // unpräzise bezüglich Höhe Z. tatsächliche Fräserhöhe an Stelle piece-at(i) eventuell abweichend!
                        piece->at(iPoint + 1)[2] = pathZ[t]; // zCut;
                        piece->at(iPoint + 2)[2] = pathZ[t]; // zCut;
                        piece->at(iPoint + 3)[2] = pathZ[t]; // zCut;
                    }
                }
            }
        }
    }
    //wpCutFaces(geo, piece);
}
*/