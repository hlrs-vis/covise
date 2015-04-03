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
#define USE_MATH_DEFINES
#include <math.h>
#include <QDir>
#include <config/coConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/LineWidth>

#include <osg/LineSegment>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osgUtil/IntersectVisitor>
#include <cover/coVRAnimationManager.h>
#define MAXSAMPLES 1200
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
{
    //positions=NULL;
    thePlugin = this;
}

static FileHandler handlers[] = {
    { NULL,
      CNCPlugin::sloadGCode,
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
    ;
    return 0;
}

//--------------------------------------------------------------------
void CNCPlugin::setTimestep(int t)
{
    if (primitives)
        primitives->at(0) = t;
}

int CNCPlugin::unloadGCode(const char *filename, const char *)
{
    (void)filename;

    return 0;
}

void CNCPlugin::deleteColorMap(const QString &name)
{
    float *mval = mapValues.value(name);
    mapSize.remove(name);
    mapValues.remove(name);
    delete[] mval;
}

bool CNCPlugin::init()
{
    fprintf(stderr, "CNCPlugin::CNCPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);

    length = 1;
    recordRate = 1;
    filename = NULL;
    primitives = NULL;
    currentMap = 0;

    coConfig *config = coConfig::getInstance();

    // read the name of all colormaps in file
    QStringList list;
    list = config->getVariableList("Colormaps");

    for (int i = 0; i < list.size(); i++)
        mapNames.append(list[i]);

    // read the values for each colormap
    for (int k = 1; k < mapNames.size(); k++)
    {
        // get all definition points for the colormap
        QString cmapname = "Colormaps." + mapNames[k];
        QStringList variable = config->getVariableList(cmapname);

        mapSize.insert(mapNames[k], variable.size());
        float *cval = new float[variable.size() * 5];
        mapValues.insert(mapNames[k], cval);

        // read the rgbax values
        int it = 0;
        for (int l = 0; l < variable.size() * 5; l = l + 5)
        {
            QString tmp = cmapname + ".Point:" + QString::number(it);
            cval[l] = config->getFloat("x", tmp, -1.0);
            if (cval[l] == -1)
            {
                cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
            }
            cval[l + 1] = config->getFloat("r", tmp, 1.0);
            cval[l + 2] = config->getFloat("g", tmp, 1.0);
            cval[l + 3] = config->getFloat("b", tmp, 1.0);
            cval[l + 4] = config->getFloat("a", tmp, 1.0);
            it++;
        }
    }

    // read values of local colormap files in .covise
    QString place = coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "colormaps";

    QDir directory(place);
    if (directory.exists())
    {
        QStringList filters;
        filters << "colormap_*.xml";
        directory.setNameFilters(filters);
        directory.setFilter(QDir::Files);
        QStringList files = directory.entryList();

        // loop over all found colormap xml files
        for (int j = 0; j < files.size(); j++)
        {
            coConfigGroup *colorConfig = new coConfigGroup("ColorMap");
            colorConfig->addConfig(place + "/" + files[j], "local", true);

            // read the name of the colormaps
            QStringList list;
            list = colorConfig->getVariableList("Colormaps");

            // loop over all colormaps in one file
            for (int i = 0; i < list.size(); i++)
            {

                // remove global colormap with same name
                int index = mapNames.indexOf(list[i]);
                if (index != -1)
                {
                    mapNames.removeAt(index);
                    deleteColorMap(list[i]);
                }
                mapNames.append(list[i]);

                // get all definition points for the colormap
                QString cmapname = "Colormaps." + mapNames.last();
                QStringList variable = colorConfig->getVariableList(cmapname);

                mapSize.insert(list[i], variable.size());
                float *cval = new float[variable.size() * 5];
                mapValues.insert(list[i], cval);

                // read the rgbax values
                int it = 0;
                for (int l = 0; l < variable.size() * 5; l = l + 5)
                {
                    QString tmp = cmapname + ".Point:" + QString::number(it);
                    cval[l] = (colorConfig->getValue("x", tmp, " -1.0")).toFloat();
                    if (cval[l] == -1)
                    {
                        cval[l] = (1.0 / (variable.size() - 1)) * (l / 5);
                    }
                    cval[l + 1] = (colorConfig->getValue("r", tmp, "1.0")).toFloat();
                    cval[l + 2] = (colorConfig->getValue("g", tmp, "1.0")).toFloat();
                    cval[l + 3] = (colorConfig->getValue("b", tmp, "1.0")).toFloat();
                    cval[l + 4] = (colorConfig->getValue("a", tmp, "1.0")).toFloat();
                    it++;
                }
            }
            config->removeConfig(place + "/" + files[j]);
        }
    }
    mapNames.sort();

    PathTab = new coTUITab("CNC", coVRTui::instance()->mainFolder->getID());
    record = new coTUIToggleButton("Record", PathTab->getID());
    stop = new coTUIButton("Stop", PathTab->getID());
    play = new coTUIButton("Play", PathTab->getID());
    reset = new coTUIButton("Reset", PathTab->getID());
    saveButton = new coTUIButton("Save", PathTab->getID());

    mapChoice = new coTUIComboBox("mapChoice", PathTab->getID());
    mapChoice->setEventListener(this);
    int i;
    for (i = 0; i < mapNames.count(); i++)
    {
        mapChoice->addEntry(mapNames[i].toStdString());
    }
    mapChoice->setSelectedEntry(currentMap);
    mapChoice->setPos(6, 0);

    viewPath = new coTUIToggleButton("View Path", PathTab->getID());
    viewDirections = new coTUIToggleButton("Viewing Directions", PathTab->getID());
    viewlookAt = new coTUIToggleButton("View Target", PathTab->getID());

    lengthLabel = new coTUILabel("Length", PathTab->getID());
    lengthLabel->setPos(0, 4);
    lengthEdit = new coTUIEditFloatField("length", PathTab->getID());
    lengthEdit->setValue(1);
    lengthEdit->setPos(1, 4);

    radiusLabel = new coTUILabel("Radius", PathTab->getID());
    radiusLabel->setPos(2, 4);
    radiusEdit = new coTUIEditFloatField("radius", PathTab->getID());
    radiusEdit->setValue(1);
    radiusEdit->setEventListener(this);
    radiusEdit->setPos(3, 4);
    renderMethod = new coTUIComboBox("renderMethod", PathTab->getID());
    renderMethod->addEntry("renderMethod CPU Billboard");
    renderMethod->addEntry("renderMethod Cg Shader");
    renderMethod->addEntry("renderMethod Point Sprite");
    renderMethod->setSelectedEntry(0);
    renderMethod->setEventListener(this);
    renderMethod->setPos(0, 5);

    recordRateLabel = new coTUILabel("recordRate", PathTab->getID());
    recordRateLabel->setPos(0, 3);
    recordRateTUI = new coTUIEditIntField("Fps", PathTab->getID());
    recordRateTUI->setEventListener(this);
    recordRateTUI->setValue(1);
    //recordRateTUI->setText("Fps:");
    recordRateTUI->setPos(1, 3);

    fileNameBrowser = new coTUIFileBrowserButton("File", PathTab->getID());
    fileNameBrowser->setMode(coTUIFileBrowserButton::SAVE);
    fileNameBrowser->setFilterList("*.txt");
    fileNameBrowser->setPos(0, 7);
    fileNameBrowser->setEventListener(this);

    numSamples = new coTUILabel("SampleNum: 0", PathTab->getID());
    numSamples->setPos(0, 6);
    PathTab->setPos(0, 0);
    record->setPos(0, 0);
    record->setEventListener(this);
    stop->setPos(1, 0);
    stop->setEventListener(this);
    play->setPos(2, 0);
    play->setEventListener(this);
    reset->setPos(3, 0);
    reset->setEventListener(this);
    saveButton->setPos(4, 0);
    saveButton->setEventListener(this);
    //positions = new float [3*MAXSAMPLES+3];
    lookat[0] = new float[MAXSAMPLES + 1];
    lookat[1] = new float[MAXSAMPLES + 1];
    lookat[2] = new float[MAXSAMPLES + 1];
    objectName = new const char *[MAXSAMPLES + 3];
    viewPath->setPos(0, 2);
    viewPath->setEventListener(this);
    viewlookAt->setPos(1, 2);
    viewlookAt->setEventListener(this);
    viewDirections->setPos(2, 2);
    viewDirections->setEventListener(this);
    frameNumber = 0;
    record->setState(false);
    playing = false;

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

    delete record;
    delete stop;
    delete play;
    delete reset;
    delete saveButton;
    delete viewPath;
    delete viewDirections;
    delete viewlookAt;
    delete lengthLabel;
    delete lengthEdit;
    delete radiusLabel;
    delete radiusEdit;
    delete renderMethod;
    delete recordRateLabel;
    delete recordRateTUI;
    delete numSamples;
    delete PathTab;
    delete[] filename;

    //delete[] positions;
    delete[] lookat[0];
    delete[] lookat[1];
    delete[] lookat[2];
    delete[] objectName;
    if (geode->getNumParents() > 0)
    {
        parentNode = geode->getParent(0);
        if (parentNode)
            parentNode->removeChild(geode.get());
    }
}

void
CNCPlugin::preFrame()
{
    if (record->getState())
    {
    }
}

void CNCPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == lengthEdit)
    {
        length = lengthEdit->getValue();
    }
    else if (tUIItem == recordRateTUI)
    {
        recordRate = 1.0 / recordRateTUI->getValue();
    }
    else if (tUIItem == fileNameBrowser)
    {
        std::string fn = fileNameBrowser->getSelectedPath();
        delete filename;
        filename = new char[fn.length()];
        strcpy(filename, fn.c_str());

        if (filename[0] != '\0')
        {
            char *pchar;
            if ((pchar = strstr(filename, "://")) != NULL)
            {
                pchar += 3;
                strcpy(filename, pchar);
            }
        }
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

void CNCPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == play)
    {
        playing = true;
    }

    if (tUIItem == saveButton)
    {
        save();
    }

    else if (tUIItem == record)
    {
        playing = false;
    }
    else if (tUIItem == stop)
    {
        record->setState(false);
        playing = false;
    }
    else if (tUIItem == reset)
    {
        frameNumber = 0;
        record->setState(false);
        playing = false;
    }
    else if (tUIItem == lengthEdit)
    {
        length = lengthEdit->getValue();
    }
    else if (tUIItem == viewPath)
    {
        char label[100];
        sprintf(label, "numSamples: %d", frameNumber);
        numSamples->setLabel(label);
        if (viewPath->getState())
        {

            if (parentNode == NULL)
                parentNode = cover->getObjectsRoot();
            parentNode->addChild(geode.get());
            ;
        }
        else
        {
            parentNode = geode->getParent(0);
            parentNode->removeChild(geode.get());
        }
    }
}

void CNCPlugin::straightFeed(double x, double y, double z, double a, double b, double c, double feedRate)
{
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
        char label[100];
        sprintf(label, "numSamples: %d", frameNumber);
        numSamples->setLabel(label);
    }
}
void CNCPlugin::tabletReleaseEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
}

osg::Vec4 CNCPlugin::getColor(float pos)
{

    osg::Vec4 actCol;
    int idx = 0;
    //cerr << "name: " << (const char *)mapNames[currentMap].toAscii() << endl;
    float *map = mapValues.value(mapNames[currentMap]);
    int mapS = mapSize.value(mapNames[currentMap]);
    if (map == NULL)
    {
        return actCol;
    }
    while (map[(idx + 1) * 5] <= pos)
    {
        idx++;
        if (idx > mapS - 2)
        {
            idx = mapS - 2;
            break;
        }
    }
    double d = (pos - map[idx * 5]) / (map[(idx + 1) * 5] - map[idx * 5]);
    actCol[0] = (float)((1 - d) * map[idx * 5 + 1] + d * map[(idx + 1) * 5 + 1]);
    actCol[1] = (float)((1 - d) * map[idx * 5 + 2] + d * map[(idx + 1) * 5 + 2]);
    actCol[2] = (float)((1 - d) * map[idx * 5 + 3] + d * map[(idx + 1) * 5 + 3]);
    actCol[3] = (float)((1 - d) * map[idx * 5 + 4] + d * map[(idx + 1) * 5 + 4]);

    return actCol;
}

COVERPLUGIN(CNCPlugin)
