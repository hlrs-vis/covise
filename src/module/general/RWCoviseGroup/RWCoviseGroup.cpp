/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   (C)2001 VirCinity    **
 **                                                                        **
 ** Description: Read/Write module for multiple COVISE Files               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Sven Kufer                                 **
 **                      VirCinity IT-Consulting GmbH                      **
 **                            Nobelstrasse 15                             **
 **                            70569 Stuttgart                             **
 **                                                                        **
 ** Date:  03.08.01                                                        **
\**************************************************************************/

#include "RWCoviseGroup.h"
#include <algorithm>
#include <string>

#include <errno.h>
#include <util/coviseCompat.h>
#include <api/coFeedback.h>
#include <do/coDoGeometry.h>
#include <do/coDoSet.h>

#undef VERBOSE
#undef DEBUGFILES
#undef DEBUGGER

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            C O N S T R U C T O R
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

RWCoviseGroup::RWCoviseGroup(int argc, char *argv[])
    : coModule(argc, argv, "Read/Write group of COVISE files")
{
    const char *PossibleTypes = { "UniformGrid|Text|Points|Spheres|UnstructuredGrid|RectilinearGrid|StructuredGrid|Float|Vec3|Polygons|TriangleStrips|Geometry|Lines|PixelImage|Spheres|Texture|IntArr|RGBA" };

    // Mesh and Data file names
    char *cov_path = getenv("COVISEDIR");
    if (cov_path)
        sprintf(GroupFileName, "%s/nofile", cov_path);
    else
        sprintf(GroupFileName, "./nofile");

    p_groupFile = addFileBrowserParam("group_file", "Group file path");
    p_groupFile->setValue(GroupFileName, "*.covgrp/*");

    // These are one per Output port
    static const char *deffileVal[] = { "---" };
    for (int i = 0; i < NUMPORTS; i++)
    {
        char buf[400];
        sprintf(buf, "file_%d", i);
        p_file[i] = addChoiceParam(buf, "COVISE file");
        p_file[i]->setValue(1, deffileVal, 0);

        sprintf(buf, "DataIn_%d", i);
        p_data_in[i] = addInputPort(buf, PossibleTypes, "Input");
        p_data_in[i]->setRequired(0);

        sprintf(buf, "DataOut_%d", i);
        p_data_out[i] = addOutputPort(buf, PossibleTypes, "Output");
    }
    for (int i = 0; i < NUMPORTS; i++)
    {
        char buf[400];
        sprintf(buf, "description_%d", i);
        p_desc[i] = addStringParam(buf, "COVISE file description");
        p_desc[i]->setValue("no description yet");
    }

    choice = NULL;
    doupdate = 1;
    mapLoading = true; // set to true to handle ports right
    in_exec = 0;
    num_in = 0;
    groupfile = NULL;
    mode = READ;
    mapLoad = NULL;
}

RWCoviseGroup::~RWCoviseGroup()
{
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            I M M E D I A T E   C A L L B A C K
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

void RWCoviseGroup::param(const char *paramname, bool inMapLoading)
{
    ////// Selected new input group file /////////////////////
    if (strcmp(p_groupFile->getName(), paramname) == 0)
    {
        checkMapLoading(inMapLoading);

        if (mode == WRITE)
        {
            parse_name(p_groupFile->getValue());
            return;
        }
        else
        {
            const char *path = p_groupFile->getValue();
            int newFiles = handleChangedGroupFile(path, mapLoading, 0);

            if (mapLoad && !mapLoading)
            {
                delete[] mapLoad;
                mapLoad = NULL;
            }

            if (newFiles > 0)
            {
                updateChoices();
                doupdate = 1;
            }
            else if (newFiles == -1)
            {
                if (!mapLoading)
                    sendError("Could not open your specified group file!");
                groupfile = NULL;
                return;
            }
        }
    }

    ////// Select new files ////////////////////

    else if (strstr(paramname, "file_") == paramname)
    {
        checkMapLoading(inMapLoading);
        int whichPort;
        if (sscanf(paramname + 5, "%d", &whichPort) != 1)
        {
            fprintf(stderr, "RWCoviseGroup::param: sscanf1 failed\n");
        }
        int choiceSel = p_file[whichPort]->getValue();

        if (choiceSel != 0)
        {
            if (choice)
                p_data_out[whichPort]->setInfo(choice->getString(choiceSel));
        }
        else
            p_data_out[whichPort]->setInfo("no data");
    }

    //////// description of one port has changed, don't update fill the description lines
    //////// with original object names

    else if (strstr(paramname, "description_") == paramname)
    {
        checkMapLoading(inMapLoading);
        int whichPort;
        if (sscanf(paramname + 12, "%d", &whichPort) != 1)
        {
            fprintf(stderr, "RWCoviseGroup::param: sscanf2 failed\n");
        }
        if (mapLoading)
            doupdate = 1;
        if (!strcmp("no description yet", p_desc[whichPort]->getValue()))
        {
            if (mapLoading)
            {
                if (mode == WRITE)
                    doupdate = 0;
            }
            else
            {
                p_desc[whichPort]->show();
                const char *str = p_desc[whichPort]->getValue();
                int i = 0;
                for (i = 0; i < NUMPORTS && i != whichPort; i++)
                {
                    if (!strcmp(str, p_desc[i]->getValue()))
                    {
                        if (mode == WRITE)
                        {
                            sendError("You used the description '%s' for the Port %d and %d. Please change the description of one of these ports.", str, i, whichPort);
                            break;
                        }
                    }
                }
                if (i == NUMPORTS)
                {
                    doupdate = 0;
                    p_desc[whichPort]->show();
                }
            }
        }
    }

    //////// port connection of an inport has changed

    else if (strstr(paramname, "DataIn_") == paramname)
    {
        getMode();
        if (!mapLoading)
        {
            int oldmode = mode;
            int old_num_in = num_in;
            updateDescription(oldmode, old_num_in);
            getObjectNames();
            doupdate = 1;
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// split filename into pathname and short filename
//
////////////////////////////////////////////////////////////////////////

void RWCoviseGroup::parse_name(const char *filename)
{
    int add_ending = 0;
    const char *cPtr = strrchr(filename, '/');
    if (cPtr)
    {
        strcpy(GroupFileName, cPtr);
        if (GroupFileName[0] == '/')
            strcpy(GroupFileName, GroupFileName + 1);
        std::string str(filename);
        if (mode == WRITE && str.find(".covgrp") == std::string::npos)
            add_ending = 1;
        std::string::size_type pos = str.find(GroupFileName);
        if (pos != std::string::npos)
        {
            str.erase(pos, strlen(GroupFileName));
        }
        strcpy(GroupFilePath, str.c_str());
    }
    else
    {
        strcpy(GroupFileName, filename);
        std::string string(filename);
        if (mode == WRITE && string.find(".covgrp") == std::string::npos)
            add_ending = 1;
        if (GroupFileName[0] == '/')
            strcpy(GroupFileName, GroupFileName + 1);
        GroupFilePath[0] = 0;
    }

    if (add_ending)
    {
        char buf[400];
        sprintf(buf, "RWGroup:%s", GroupFileName);
        setTitle(buf);
        strcat(GroupFileName, ".covgrp");
    }
    else
    {
        std::string header("RWGroup:");
        std::string title(GroupFileName);
        std::string::size_type pos = title.find(".covgrp");
        if (pos != std::string::npos)
            title.erase(pos, strlen(".covgrp"));
        setTitle((header + title).c_str());
    }

    strcpy(GroupFileString, GroupFilePath);
    strcat(GroupFileString, GroupFileName);
    Covise::send_ui_message("MODULE_DESC", GroupFileName);
}

////////////////////////////////////////////////////////////////////////
//
// get value of the SPECIES attribute of the object
//
////////////////////////////////////////////////////////////////////////

const char *RWCoviseGroup::getSpecies(const coDistributedObject *obj)
{
    const coDistributedObject *tmp_obj;
    if (obj == NULL)
        return NULL;
    if (obj->isType("GEOMET"))
    {
        tmp_obj = ((const coDoGeometry *)obj)->getGeometry();
        return getSpecies(tmp_obj);
    }
    else if (obj->isType("SETELE"))
    {
        int num_elem;
        const coDistributedObject *const *set;
        set = ((const coDoSet *)obj)->getAllElements(&num_elem);
        return getSpecies(set[0]);
    }
    else
        return obj->getAttribute("SPECIES");
}

////////////////////////////////////////////////////////////////////////
//
//  file the description field with the incoming object names
//
////////////////////////////////////////////////////////////////////////

int RWCoviseGroup::getObjectNames()
{
    for (int i = 0; i < NUMPORTS; i++)
    {
        if (!p_data_in[i]->isConnected())
            p_desc[i]->hide();
        else if (in_exec)
        {
            const coDistributedObject *tmp_obj = p_data_in[i]->getCurrentObject();
            if (tmp_obj)
            {
                const char *name = getSpecies(tmp_obj);
                if (name != NULL)
                    p_desc[i]->setValue(name);
                else
                    p_desc[i]->setValue(tmp_obj->getName());
                p_desc[i]->show();
            }
            else if (p_data_in[i]->isConnected())
            {
                sendError("Missing object at port DataIn_%d. Please check pipeline.", i);
                for (int j = 0; j < i; j++)
                {
                    p_desc[j]->setValue("no description yet");
                    p_desc[j]->hide();
                }
                return 1;
            }
        }
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////
//
// In READ mode: update choices and port descriptions
//
////////////////////////////////////////////////////////////////////////

void RWCoviseGroup::updateChoices()
{
    if (!choice)
        return;
    const char *const *choiceLabels = choice->get_strings();
    int numChoices = choice->get_num();

    for (int i = 0; i < NUMPORTS; i++)
    {

        //p_desc[i]->hide();
        if (i + 1 < numChoices)
        {
            p_file[i]->setValue(numChoices, choiceLabels, i + 1);
            p_data_out[i]->setInfo(choice->getString(i + 1));
            p_file[i]->show();
        }
        else
        {
            p_file[i]->hide();
            p_file[i]->setValue(numChoices, choiceLabels, 0);
            p_data_out[i]->setInfo("no data");
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// decide whether READ or WRITE is the mode
//
// rule: if one input port is connected take the mode WRITE, otherwise: READ
//
////////////////////////////////////////////////////////////////////////

void RWCoviseGroup::getMode()
{
    num_in = 0;

    for (int i = 0; i < NUMPORTS; i++)
    {
        if (p_data_in[i]->isConnected())
            num_in++;
    }
    if (num_in > 0)
        mode = WRITE;
    else
        mode = READ;
}

////////////////////////////////////////////////////////////////////////
//
// check if something has changed
//
// if yes, update description
//
////////////////////////////////////////////////////////////////////////

void RWCoviseGroup::updateDescription(int oldmode, int old_num_in)
{
    // mode has changed: update control panel
    if (mode == WRITE && oldmode != mode && !mapLoading)
    {
        for (int i = 0; i < NUMPORTS; i++)
        {
            p_file[i]->hide();
            p_data_out[i]->setInfo("no data");
        }
    }
    else if (mode == WRITE && old_num_in != num_in && !mapLoading)
    {
        for (int i = 0; i < NUMPORTS; i++)
        {
            p_desc[i]->hide();
            p_desc[i]->setValue("no description yet");
        }
    }
    else if (oldmode != mode && mode == READ && !mapLoading)
    {
        for (int i = 0; i < NUMPORTS; i++)
        {
            p_desc[i]->hide();
            p_desc[i]->setValue("no description yet");
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
//  check if we are in loading a map
//
////////////////////////////////////////////////////////////////////////

void RWCoviseGroup::checkMapLoading(bool inMapLoading)
{
    bool oldmapLoading = mapLoading;

    mapLoading = inMapLoading;

    if (mapLoading != oldmapLoading && mode == READ)
    {
        for (int i = 0; i < NUMPORTS; i++)
        {
            int num = p_file[i]->getValue();
            if (num == 1)
                p_data_out[i]->setInfo("no data");
            else
                p_data_out[i]->setInfo(p_file[i]->getLabel(i));
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// generate ouput file name
//
////////////////////////////////////////////////////////////////////////

void RWCoviseGroup::gen_filename(char *desc, char *file)
{
    strcpy(file, GroupFilePath);
    std::string str(desc);
    for (int i = 0; i < 100; ++i)
    {
        std::string::size_type pos = str.find(" ");
        if (pos != std::string::npos)
            str.replace(pos, 1, "_");
    }
    strcat(file, str.c_str());
    strcat(file, ".covise");
}

////////////////////////////////////////////////////////////////////////
//
// what to do if the group file name has changed
//
////////////////////////////////////////////////////////////////////////

int RWCoviseGroup::handleChangedGroupFile(const char *newpath,
                                          int inMapLoading, int ignoreErr)
{
    (void)ignoreErr;

    int newFiles = -1;

    if (inMapLoading)
    {
        mapLoad = strcpy(new char[strlen(newpath) + 1], newpath);
        return -1;
    }

    if (strcmp(GroupFileString, newpath) != 0)
    {
        if (mode == READ)
        {
            delete groupfile;
            groupfile = new GroupFile(newpath, 1);
            if (groupfile->isValid())
            {
                choice = groupfile->get_choice(&filenames);
                parse_name(newpath);
                newFiles = 1;

                delete[] mapLoad;
                mapLoad = NULL;
            }
        }
        else
        {
            newFiles = -1;
        }
    }

    return newFiles;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            C O M P U T E
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

int RWCoviseGroup::compute(const char *)
{
#ifdef DEBUGFILES
    static int instance = 0;
    int pid = getpid();
    char filename[50];
    sprintf(filename, "Data.%d.%03d", pid, instance);
    FILE *dataFile = fopen(filename, "w");
    instance++;
#endif

    in_exec = 1;
    getMode();
    // =========================================================
    // if we got our data from the map, we have to read it now
    // =========================================================

    if (mapLoad)
        handleChangedGroupFile(mapLoad, 0, 1);

    ////////////////////////////////////////////////////////////////////////
    //
    // in mode WRITE during first execution: just read object names
    //
    ////////////////////////////////////////////////////////////////////////

    if (doupdate && mode == WRITE)
    {
        doupdate = getObjectNames();
        in_exec = 0;
        return CONTINUE_PIPELINE;
    }

    char cfile[400]; // filename without ending newline
    char current_file[400];
    char *desc[NUMPORTS], *files[NUMPORTS];
    int num_desc = 0;

    if (mode == READ)
    {
        if (!choice && groupfile)
        {
            choice = groupfile->get_choice(&filenames);
        }

        for (int i = 0; i < NUMPORTS; i++)
        {
            int choiceVal = p_file[i]->getValue();
            if (choiceVal > 0 && choice)
            {
                if (choiceVal >= choice->get_num())
                    choiceVal = 0;
                p_data_out[i]->setInfo(choice->getString(choiceVal));

                strcpy(current_file, filenames->getString(choiceVal));
                if (current_file[0] == '/')
                    strcpy(cfile, current_file);
                else
                {
                    getname(cfile, current_file, GroupFilePath);
                }
                coDistributedObject *obj = ReadFile(cfile, p_data_out[i]->getObjName());
                if (obj && i == 0)
                {
                    // Create file browser feedback
                    coFeedback feedback("FileBrowserParam");
                    feedback.addPara(p_groupFile);
                    feedback.apply(obj);
                }
                p_data_out[i]->setCurrentObject(obj);
            }
            else
            {
                p_data_out[i]->setCurrentObject(NULL);
                p_data_out[i]->setInfo("no data");
            }
        }
    }
    else
    {
        parse_name(p_groupFile->getValue());

        groupfile = new GroupFile(GroupFileString, WRITE);
        if (groupfile->isValid())
        {
            for (int i = 0; i < NUMPORTS; i++)
            {
                const coDistributedObject *obj = p_data_in[i]->getCurrentObject();
                if (obj)
                {
                    desc[num_desc] = new char[400];
                    files[num_desc] = new char[400];
                    strcpy(desc[num_desc], p_desc[i]->getValue());
                    gen_filename(desc[num_desc], files[num_desc]);
                    WriteFile(files[num_desc], obj);
                    num_desc++;
                }
            }
            groupfile->put_choices(num_desc, desc, files);
            for (int i = 0; i < num_desc; i++)
            {
                delete[] desc[i];
                delete[] files[i];
            }
            delete groupfile;
            groupfile = NULL;
        }
        else
        {
            Covise::sendError("Could not open your specified group file '%s'.", GroupFileString);
            in_exec = 0;
            return STOP_PIPELINE;
        }
    }

    ///////////////////////////////////////////////////////////////////
    //                     Check output settings
    ///////////////////////////////////////////////////////////////////

    for (int i = 0; i < NUMPORTS; i++)
    {
        if (p_file[i]->getValue() == 0 && p_data_out[i]->isConnected())
        {
            sendError("Port %d connected but no file selected", i);
            in_exec = 0;
            return STOP_PIPELINE;
        }
    }

    in_exec = 0;
    return CONTINUE_PIPELINE;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
//
// Class GroupFile: read and write group file information
//
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

GroupFile::GroupFile(const char *groupfile, int open_to_read)
{
    if (open_to_read)
        fd = Covise::fopen(groupfile, "r");
    else
        fd = Covise::fopen(groupfile, "w");
}

int GroupFile::isValid()
{
    return (fd != NULL);
}

GroupFile::~GroupFile()
{
    fclose(fd);
}

////////////////////////////////////////////////////////////////////////
//
//  Read file in format: <description>:<filename> \n
//
////////////////////////////////////////////////////////////////////////

ChoiceList *GroupFile::get_choice(ChoiceList **files)
{
    delete *files;
    *files = NULL;
    int num = 1;
    char buffer[MAXLINE];
    char *help;
    const char *deffileVal = { "---" };

    ChoiceList *choice = new ChoiceList(deffileVal, 0);
    *files = new ChoiceList(deffileVal, 0);

    while (fgets(buffer, MAXLINE, fd))
    {
        help = strtok(buffer, ":");
        if (help == NULL)
        {
            Covise::sendError("Error in line %d of groupfile: Not in <description>:<filename> format!", num);
            return NULL;
        }
        std::string description(help);
        help = strtok(NULL, "\n");
        if (help == NULL)
        {
            Covise::sendError("Error in line %d of groupfile: Not in <description>:<filename> format!", num);
            return NULL;
        }
        std::string filename(help);

        choice->add(description.c_str(), num);
        (*files)->add(filename.c_str(), num);
        num++;
    }
    return choice;
}

////////////////////////////////////////////////////////////////////////
//
// Write group file
//
////////////////////////////////////////////////////////////////////////

int GroupFile::put_choices(int num, char **desc, char **files)
{
    int i;
    for (i = 0; i < num; i++)
        fprintf(fd, "%s:%s\n", desc[i], files[i]);
    return 1;
}

MODULE_MAIN(IO, RWCoviseGroup)
