/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for STAR  Files              	              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Andreas Werner                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  03.01.97  V1.0                                                  **
\**************************************************************************/

#include <util/coviseCompat.h>
#include <do/coDoSet.h>
#include <appl/ApplInterface.h>
#include <util/ChoiceList.h>
#include <star/File29.h>
#include <star/File09.h>
#include <star/File16.h>
#include "ReadStar.h"

#include <sys/types.h>
#include <sys/stat.h>

#undef VERBOSE
#undef DEBUGFILES
#undef DEBUGGER

#define ERR0(cond, text, action)     \
    {                                \
        if (cond)                    \
        {                            \
            Covise::sendError(text); \
            {                        \
                action               \
            }                        \
        }                            \
    }

#define ERR1(cond, text, arg1, action)          \
    {                                           \
        if (cond)                               \
        {                                       \
            Covise::sendError(buf, text, arg1); \
            {                                   \
                action                          \
            }                                   \
        }                                       \
    }

#define ERR2(cond, text, arg1, arg2, action)          \
    {                                                 \
        if (cond)                                     \
        {                                             \
            Covise::sendError(buf, text, arg1, arg2); \
            {                                         \
                action                                \
            }                                         \
        }                                             \
    }

/////////////////////////////////////////////////////////////////////
// commodity function

inline const char *show_name(const char *filename)
{
    const char *cPtr = strrchr(filename, '/');
    return cPtr ? cPtr + 1 : filename;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            C O N S T R U C T O R
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

ReadStar::ReadStar(int argc, char *argv[])
    : coModule(argc, argv, "Read grid from Star-CD File09")
{
    starfile = NULL;
    choice = NULL;
    //cells_used = NULL;
    //num_all_cells = 0;
    //all_cells_used = NULL;
    //fieldnoBak[0] = fieldnoBak[1] = fieldnoBak[2] = 0;

    // Mesh and Data file names
    char buffer[128];
    char *cov_path = getenv("READSTAR_DIR");
    if (!cov_path)
        cov_path = getenv("COVISEDIR");
    if (cov_path)
        sprintf(buffer, "%s/nofile", cov_path);
    else
        sprintf(buffer, "./nofile");

    p_meshFile = addFileBrowserParam("mesh_path", "Mesh file path");
    p_meshFile->setValue(buffer, "*16;*mdl;*MDL");

    p_dataFile = addFileBrowserParam("data_path", "Data file path");
    p_dataFile->setValue(buffer, "*09;*29*;*pst*;*PST*");

    // Parameter inputs
    p_fromToStep = addInt32VectorParam("from_to_step", "Read from/to/by step ");
    p_fromToStep->setValue(1, 1, 1);

    p_timestep = addIntSliderParam("timestep", "current timestep to be read (0 for static)");
    p_timestep->setValue(-1, 0, 0);

    static const char *defCellVal[] = { "Create VERTEX Data", "Create CELL Data" };
    p_cellVert = addChoiceParam("cellVert", "Create Cell- or Vertex-based data");
    p_cellVert->setValue(2, defCellVal, 0);

    // The output grid
    p_mesh = addOutputPort("mesh", "UnstructuredGrid", "Mesh output");

    // These are one per Output port
    int i;
    static const char *defFieldVal[] = { "---" };
    for (i = 0; i < NUMPORTS; i++)
    {
        sprintf(buffer, "field_%d", i);
        p_field[i] = addChoiceParam(buffer, "Field to read for output");
        p_field[i]->setValue(1, defFieldVal, 0);

        sprintf(buffer, "data_%d", i);
        p_data[i] = addOutputPort(buffer,
                                  "Float|Vec3|Points", "Data Output");

        field_no[i] = 0;
        //choiceSel[i]=1;
    }

    // Output
    p_type = addOutputPort("type", "IntArr", "Cell types");
    p_celltab = addOutputPort("cellTable", "IntArr", "Cell Table");
    p_cpPoly = addOutputPort("cpPolygons", "Polygons", "CP Matching Poly");

    // Do the setup
    // Set internal object pointers to Files and Filenames

    num_timesteps = 0;
    timestep = 0;

    file16 = NULL;
    file09 = NULL;
    file29 = NULL;

    file16_name[0] = '\0';
    file9_name[0] = '\0';

    fromStep = toStep = 0;
    byStep = 1;

    // filenames received at map loading : do not immediately read it.
    mapLoad9 = NULL;
    mapLoad16 = NULL;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            D E S T R U C T O R
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

ReadStar::~ReadStar()
{
    delete file16;
    delete file09;
    delete file29;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            I M M E D I A T E   C A L L B A C K
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

void ReadStar::param(const char *paramname, bool inMapLoading)
{
    const char *path;
    long newFrom, newTo, newBy;
    int newFiles = 0;

    ////// Selected new input data file /////////////////////

    if (strcmp(p_dataFile->getName(), paramname) == 0)
    {
        path = p_dataFile->getValue();

        // we assemble our own path if we have transient data
        if (num_timesteps > 0)
        {
            fprintf(stderr, "path is: %s Timestep is %ld out of %ld\n", path, timestep, num_timesteps);
            //fprintf(stderr, "hmmm\n");
        }
        newFiles = handleChangedDataPath(path, inMapLoading, 0);

        if (!inMapLoading)
        {
            delete[] mapLoad9;
            mapLoad9 = NULL;
            delete[] mapLoad16;
            mapLoad16 = NULL;
        }

        if (newFiles == -1)
            return;
    }

    ////// Selected new input mesh file /////////////////////

    else if (strcmp(p_meshFile->getName(), paramname) == 0)
    {
        path = p_meshFile->getValue();
        handleChangedMeshPath(path, inMapLoading);
        if (inMapLoading)
            return;
        else
        {
            delete[] mapLoad9;
            mapLoad9 = NULL;
            delete[] mapLoad16;
            mapLoad16 = NULL;
        }

        newFiles = 1;
    }

    ////// Selected new stepping /////////////////////

    else if (strcmp(p_fromToStep->getName(), paramname) == 0)
    {
        p_fromToStep->getValue(newFrom, newTo, newBy);

        // if this is a startup call-back, it's called last in map loading, so
        // all files are  initialized if existent
        // changed aw: startup reads no files!

        if (inMapLoading || mapLoad9 || mapLoad16) // in Map loading we accept everything
        {
            fromStep = newFrom;
            toStep = newTo;
            byStep = newBy;
        }
        else
        {
            // if it's not transient, we have no steps
            if (!file29)
            {
                fromStep = toStep = byStep = 0;
                p_fromToStep->setValue(0, 0, 0);
                p_fromToStep->hide();
            }
            else
            {
                int maxStep = file29->get_num_steps();
                if ((newFrom < 1) || (newFrom > maxStep) || (newTo < newFrom))
                    newFrom = 1;
                if ((newTo < 1) || (newTo > maxStep) || (newTo < newFrom))
                    newTo = maxStep;
                if (newBy < 1)
                    newBy = 1;
                fromStep = newFrom;
                toStep = newTo;
                byStep = newBy;
                p_fromToStep->setValue(fromStep, toStep, byStep);
                p_fromToStep->show();
            }
        }
    }

    ////// Select new fields //////////////////// @@@@@@@@@@@@@@@

    else if (strstr(paramname, "field_") == paramname)
    {
        int whichPort;
        if (sscanf(paramname + 6, "%d", &whichPort) != 1)
        {
            cerr << "ReadStar::param: sscanf1 failed" << endl;
        }
        //choiceSel[whichPort] = p_field[whichPort]->getValue();
        int choiceSel = p_field[whichPort]->getValue();

        if (choice)
        { // if we already created a choice list
            const char *const *choiceLabels = choice->get_strings();
            int numChoices = choice->get_num();
            p_data[whichPort]->setInfo(choice->getString(choiceSel));
            p_field[whichPort]->setValue(numChoices, choiceLabels, choiceSel);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    ////// Never get here...

    else if (!inMapLoading && strcmp(paramname, "SetModuleTitle"))
    {
        // all params are now immediate
        return;
    }

    else
        return;

    ///////////////////////////////////////////////////////////////////////////
    ////// If the files changed, take care of the Choices

    if ((newFiles) && (file16) && (starfile) && (!inMapLoading))
    {
        delete choice;
        choice = starfile->get_choice((const char **)file16->scalName, file16->maxscl);

        int numChoices = choice->get_num();

        // make sure we can see Cell/Vertex switch
        p_cellVert->show();

        // Check whether current value is ok, if not set to empty
        int i;
        const char *const *choiceLabels = choice->get_strings();
        for (i = 0; i < NUMPORTS; i++)
        {
            int currentChoice = p_field[i]->getValue();
            if (currentChoice >= numChoices)
            {
                p_field[i]->setValue(numChoices, choiceLabels, 0);
                p_data[i]->setInfo("no data");
            }
            else
            {
                p_field[i]->setValue(numChoices, choiceLabels, currentChoice);
                p_data[i]->setInfo(choice->getString(currentChoice));
            }
            if (i < 3)
                p_field[i]->show();
        }

        ///// and care about the stepping ////////////////////////////////////

        if (!file29) // if it's not transient, we have no steps
        {
            fromStep = toStep = byStep = 0;
        }
        else
        {
            int maxStep = file29->get_num_steps();
            if ((fromStep < 1) || (fromStep > maxStep) || (fromStep < newFrom))
                fromStep = 1;
            if ((toStep < 1) || (toStep > maxStep) || (toStep < newFrom))
                toStep = maxStep;
            if (byStep < 1)
                byStep = 1;
        }
        p_fromToStep->setValue(fromStep, toStep, byStep);
    }
}

////////////////////////////////////////////////////////////////////////

void dumpFunction(const char *s)
{
    Covise::sendInfo("%s", s);
}

void ReadStar::handleChangedMeshPath(const char *newpath, int inMapLoading)
{
    char bfr[500];

    if (strcmp(newpath, "/") == 0)
        return;
    if (inMapLoading)
    {
        mapLoad16 = strcpy(new char[strlen(newpath) + 1], newpath);
        return;
    }

    if (strcmp(file16_name, newpath) != 0)
    {

        delete file16;

        //// send busy message (toDo)

        int fd = Covise::open(newpath, O_RDONLY);
        if (fd >= 0)
            file16 = new File16(fd, dumpFunction);
        else
            file16 = NULL;

        //// send non-busy message (toDo)
        if ((!file16) || (!file16->isValid()))
        {
            if (fd < 0)
                sendError("Error reading '%s' : %s", newpath, strerror(errno));
            else
                sendError("Error reading '%s' : not a Star valid model file", newpath);

            delete file16;
            file16 = NULL;
            strcpy(file16_name, "");
            delete[] mapLoad16;
            mapLoad16 = NULL;
            return;
        }
        strcpy(file16_name, newpath);
        Covise::sendInfo("File '%s': %i cells, %i vertices, %i SAMM",
                         newpath, file16->maxe, file16->maxn, file16->getNumSAMM());

        int numCPpoly, numCPconn, numCPvert;
        file16->getCPsizes(numCPvert, numCPconn, numCPpoly);
        if (numCPpoly)
            Covise::sendInfo(bfr, "attached boundaries: %d polygons, %i Conn, %i Vert",
                             numCPpoly, numCPconn, numCPvert);
    }

    delete[] mapLoad16;
    mapLoad16 = NULL;

    return;
}

////////////////////////////////////////////////////////////////////////

int ReadStar::handleChangedDataPath(const char *newpath,
                                    int inMapLoading, int ignoreErr)
{
    int newFiles = 0;
    const char *path;
    char bfr[500];
    char tmpfile[500];
    int i;
    //int hdl;
    struct stat statbfr;
    if (strcmp(newpath, "/") == 0)
        return -1;

    if (inMapLoading)
    {
        mapLoad9 = strcpy(new char[strlen(newpath) + 1], newpath);
        return -1;
    }

    //fprintf(stderr, "handleChangedDataPath (%s)\n", newpath);
    i = strlen(newpath);
    if (newpath[i - 1] == 'z' && newpath[i - 2] == 'g')
    {
        // gzipped file

        // see if we have it unzipped there
        strcpy(bfr, newpath);
        bfr[i - 3] = '\0';
        if (!stat(bfr, &statbfr))
        {
            // yes we have it, so use the unzipped instead
            strcpy(tmpfile, newpath);
            tmpfile[i - 3] = '\0';
            path = tmpfile;
        }
        else
        {
            // it is not there, so unzip it (if gzip is not allready on it)
            sprintf(tmpfile, "%s.tmp", newpath);
            if (!stat(tmpfile, &statbfr))
            {
                // unzip is on it -> wait
                while (stat(bfr, &statbfr))
                    ;
                strcpy(tmpfile, newpath);
                tmpfile[i - 3] = '\0';
                path = tmpfile;
            }
            else
            {
                // we have to unzip it ourselves
                strcpy(tmpfile, newpath);
                tmpfile[i - 3] = '\0';
                sprintf(bfr, "gzip -d -c %s > %s", newpath, tmpfile);
                if (system(bfr) == -1)
                {
                    cerr << "ReadStar::handleChangedDataPath: exec of " << bfr << " failed" << endl;
                }
                path = tmpfile;
            }
        }
    }
    else if (newpath[i - 1] == 'Z')
    {
        // compressed file
        fprintf(stderr, "compressed files not yet supported\n");
        path = "";
    }
    else
        path = newpath;

    if (strcmp(file9_name, path) != 0)
    {

        // delete old Files
        delete file09;
        file09 = NULL;
        delete file29;
        file29 = NULL;

        // try to read as File29

        //send busy message (toDo)
        starfile = NULL;

        file29 = new File29(Covise::open(path, O_RDONLY), dumpFunction);

        if ((!file29) || (!file29->isValid()))
        {
            delete file29;
            file29 = NULL;

            // Not 29, try 09
            int fd = Covise::open(path, O_RDONLY);
            if (fd > 0)
                file09 = new File09(fd);
            else
                file09 = NULL;

            if ((!file09) || (!file09->isValid()))
            {
                if (fd < 0) // file opening faied
                {
                    // we ignore 'file does not exist' for map-loading:
                    // needed if loading a map with ReadStar reading grid only
                    if (ignoreErr == 0 || errno != ENOENT)
                    {
                        Covise::sendError("Error %d reading '%s': %s", errno, path, strerror(errno));
                    }
                }
                else
                {
                    Covise::sendError("Could not read '%s' as STAR Data File (09/29)", path);
                }
                delete file09;
                file09 = NULL;
                strcpy(file9_name, "");
                delete[] mapLoad9;
                mapLoad9 = NULL;

                //send non-busy message (toDo)
                return -1;
            }

            //////// This was a File09
            else
            {
                sprintf(bfr, "File '%s': %i cells, %i vertices, stationary",
                        show_name(path), file09->ncell, file09->nnode);
                starfile = file09;
                p_fromToStep->setValue(0, 0, 0);
                p_fromToStep->hide();
                fromStep = 0;
                toStep = 0;
                byStep = 0;
            }
        }

        //////// This was a File29
        else
        {

            sprintf(bfr, "File '%s': %i cells, %i vertices, transient, %i steps ",
                    show_name(path), file29->ncell, file29->nnode,
                    file29->get_num_steps());
            starfile = file29;
            int numSteps = file29->get_num_steps();

            // If we got legal values, keep it, else set good new values
            if (fromStep > numSteps)
                fromStep = 1;

            if (toStep > numSteps)
                toStep = numSteps;

            if (byStep <= 0)
                byStep = 1;

            // whether or not it changed, we'll send it and show it in the panel

            p_fromToStep->setValue(fromStep, toStep, byStep);
            p_fromToStep->show();
        }
        newFiles = 1;
        Covise::send_ui_message("MODULE_DESC", show_name(path));
    }

    delete[] mapLoad9;
    mapLoad9 = NULL;

    ////// Selected new input mesh file /////////////////////
    return newFiles;
}

void ReadStar::parseStartStep(const char *str, int *start, int *step)
{
    int i, j;
    char buf[300];

    //fprintf(stderr, "parseStartStep: %s\n", str);

    // we love parsing
    for (i = 0; str[i] && !(str[i] == '.' && str[i + 1] == '.'); i++)
        ;
    if (str[i])
    {
        i += 2;
        // we have multiple stationary files
        for (j = 0; str[i] && !(str[i] == '.' && str[i + 1] == '.'); i++)
        {
            buf[j] = str[i];
            j++;
        }
        buf[j] = '\0';
        if (sscanf(buf, "%d", start) != 1)
        {
            cerr << "ReadStar::parseStartStep: sscanf1 failed" << endl;
        }
        if (str[i])
        {
            i += 2;
            // here comes the hot-stepper, trammladamm
            for (j = 0; str[i]; i++)
            {
                buf[j] = str[i];
                j++;
            }
            buf[j] = '\0';
            if (sscanf(buf, "%d", step) != 1)
            {
                cerr << "ReadStar::parseStartStep: sscanf2 failed" << endl;
            }
        }
        else
            *step = 1;
    }
    else
    {
        *start = -1;
        *step = -1;
    }

    return;
}

// Add timestep Info for multi-step reading and BlockCollect
void ReadStar::addBlockInfo(coDistributedObject *obj,
                            int timestep, int num_timesteps)
{
    char buf[64];
    sprintf(buf, "%d %d", timestep, num_timesteps);
    obj->addAttribute("BLOCKINFO", buf);
    sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
    obj->addAttribute("READ_MODULE", buf);
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            C O M P U T E
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

int ReadStar::compute(const char *)
{
    ///////////////////////////////////////////////////////////////////
    //                     common variables
    ///////////////////////////////////////////////////////////////////

    int i, n;

#ifdef DEBUGFILES
    static int instance = 0;
    int pid = getpid();
    char filename[50];
    sprintf(filename, "Data.%d.%03d", pid, instance);
    FILE *dataFile = fopen(filename, "w");
    instance++;
#endif

    // Make sure the requested parameters are there... in case of edited map file
    //int field_no[3];
    const char *mesh_path, *data_path;
    char buf[300], buf2[300];
    char tmpfile[500];

    mesh_path = p_meshFile->getValue();
    data_path = p_dataFile->getValue();

    //int act_timestep, num_timesteps;
    long waste;
    p_timestep->getValue(waste, num_timesteps, timestep);

    // =========================================================
    // if we got our meshes from the map, we have to read it now
    // =========================================================

    if (mapLoad9 && mapLoad16)
    {
        int doUpdate = 0;
        handleChangedDataPath(mapLoad9, 0, 1);
        handleChangedMeshPath(mapLoad16, 0);
        if (file09 && (fromStep || toStep || byStep))
        {
            fromStep = toStep = byStep = 0;
            doUpdate = 1;
        }
        else if (file29)
        {
            int maxStep = file29->get_num_steps();
            if ((fromStep < 1) || (fromStep > maxStep))
            {
                fromStep = 1;
                doUpdate = 1;
            }
            if ((toStep < 1) || (toStep > maxStep) || (toStep < fromStep))
            {
                toStep = maxStep;
                doUpdate = 1;
            }
            if (byStep < 1)
            {
                byStep = 1;
                doUpdate = 1;
            }
        }
        for (i = 0; i < NUMPORTS; i++)
            p_field[i]->show();

        if (choice)
        {
            int whichPort;
            for (whichPort = 0; whichPort < NUMPORTS; whichPort++)
            {
                char buffer[64];
                sprintf(buffer, "data_%d", i);
                strcat(buffer, "\n");
                strcat(buffer, choice->getString(p_field[whichPort]->getValue()));
                // remove trailing \n
                char *bPtr = buffer + strlen(buffer) - 1;
                if (*bPtr == '\n')
                    *bPtr = '\0';
                cerr << "PORT_DESC\n------\n" << buffer << "\n------\n" << endl;
                Covise::send_ui_message("PORT_DESC", buffer);
            }
        }

        if (doUpdate)
        {
            p_fromToStep->setValue(fromStep, toStep, byStep);
            Covise::sendError("incorrect stepping: corrected");
            return STOP_PIPELINE;
        }
    }

    // =========================================================
    // Lars' Timesteps -> wie auch immer...
    // =========================================================

    if (0) // if (num_timesteps)
    {
        int start, step;
        // we love parsing
        parseStartStep(mesh_path, &start, &step);
        if ((start != -1) && step)
        {
            // load the current mesh
            for (i = 0; !(mesh_path[i] == '.' && mesh_path[i + 1] == '.'); i++)
                buf[i] = mesh_path[i];
            buf[i] = '\0';
            sprintf(buf2, buf, start + step * timestep);
            handleChangedMeshPath(buf2, 0);
        }

        parseStartStep(data_path, &start, &step);

        if ((start != -1) && step)
        {
            // first see if we previously used a compressed file and
            // uncompressed it - we should free up diskspace
            if (timestep)
            {
                // compute the last data-name
                for (i = 0; !(data_path[i] == '.' && data_path[i + 1] == '.'); i++)
                    buf[i] = data_path[i];
                buf[i] = '\0';
                sprintf(buf2, buf, start + step * (timestep - 1));

                i = strlen(buf2);

                if (buf2[i - 1] == 'z' && buf2[i - 2] == 'g')
                {
                    // gzipped file - so delete the uncompressed one
                    buf2[i - 3] = '\0';
                    unlink(buf2);
                }
                // add support for other compression-types here
            }

            // launch an unzip for the 7th step in the future (?!)
            if (timestep < num_timesteps - 8)
            {
                if (!timestep)
                {
                    // we come here for the first time, so start up to 7 uncompressors
                    for (n = 0; n < (num_timesteps > 7 ? 7 : num_timesteps); n++)
                    {
                        for (i = 0; !(data_path[i] == '.' && data_path[i + 1] == '.'); i++)
                            buf[i] = data_path[i];
                        buf[i] = '\0';
                        sprintf(buf2, buf, start + step * (timestep + n));
                        i = strlen(buf2);
                        if (buf2[i - 1] == 'z' && buf2[i - 2] == 'g')
                        {
                            // gzipped file
                            buf2[i - 3] = '\0';
                            strcpy(tmpfile, buf2);
                            buf2[i - 3] = '.';
                            sprintf(buf, "gzip -d -c %s > %s.tmp ; mv %s.tmp %s &",
                                    buf2, buf2, buf2, tmpfile);
                            if (system(buf) == -1)
                            {
                                cerr << "ReadStar::compute: exec1 of " << buf << " failed" << endl;
                            }
                        }
                        // add support for other compr. here
                    }
                }
                else
                {
                    // not here for the first time, so maybe just start current+7th step
                    for (i = 0; !(data_path[i] == '.' && data_path[i + 1] == '.'); i++)
                        buf[i] = data_path[i];
                    buf[i] = '\0';
                    sprintf(buf2, buf, start + step * (timestep + 7));
                    i = strlen(buf2);
                    if (buf2[i - 1] == 'z' && buf2[i - 2] == 'g')
                    {
                        // gzipped file
                        buf2[i - 3] = '\0';
                        strcpy(tmpfile, buf2);
                        buf2[i - 3] = '.';
                        sprintf(buf, "gzip -d -c %s > %s.tmp ; mv %s.tmp %s &",
                                buf2, buf2, buf2, tmpfile);
                        if (system(buf) == -1)
                        {
                            cerr << "ReadStar::compute: exec2 of " << buf << " failed" << endl;
                        }
                    }
                    // add support for other compr. here
                }
            }

            // load the current data
            for (i = 0; !(data_path[i] == '.' && data_path[i + 1] == '.'); i++)
                buf[i] = data_path[i];
            buf[i] = '\0';
            sprintf(buf2, buf, start + step * timestep);
            if (handleChangedDataPath(buf2, 0, 0) == -1)
                Covise::sendError("What now ?!");
        }
    }

    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    // Someone just changed our Files ??

    if (file16 && !file16->isFile(file16_name))
    {
        handleChangedMeshPath(file16_name, 0);
        sendInfo("File '%s' just changed its content: Re-read", file16_name);
    }

    if ((file09 && !file09->isFile(file9_name))
        || (file29 && !file29->isFile(file9_name)))
    {
        handleChangedDataPath(file9_name, 0, 0);
        sendInfo("File '%s' just changed its content: Re-read", file9_name);
    }

    ///////////////////////////////////////////////////////////////////
    //                     Everything there?
    ///////////////////////////////////////////////////////////////////

    ERR0((!file16), "No valid mesh file", return STOP_PIPELINE;)
    //ERR0( ((!file09)&&(!file29)),"No valid data file", return STOP_PIPELINE; )

    ///////////////////////////////////////////////////////////////////
    //                     Input parameters
    ///////////////////////////////////////////////////////////////////

    if (starfile)
    {
        if (!choice)
        {
            choice = starfile->get_choice((const char **)file16->scalName, file16->maxscl);
        }

        for (i = 0; i < NUMPORTS; i++)
        {
            int choiceVal = p_field[i]->getValue();
            if (choiceVal >= choice->get_num())
                choiceVal = 0;

            fieldName[i] = choice->getString(choiceVal);

            field_no[i] = choice->get_orig_num(choiceVal);
        }
    }
    else
    {
        for (i = 0; i < NUMPORTS; i++)
            field_no[i] = 0;
    }

    ///////////////////////////////////////////////////////////////////
    //                     Check output settings
    ///////////////////////////////////////////////////////////////////

    for (i = 0; i < NUMPORTS; i++)
    {
        if (field_no[i] == 0 && p_field[i]->isConnected())
        {
            sendError("Port '%s' connected but no field selected on no data file", p_field[i]->getName());
            return -1;
        }
    }

    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    /////
    /////        Data -> call subroutines for static/dynamic
    /////
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////

    if (!file29) // stationary or no data
        return statData();
    else
        return transData();
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///    Handle static data
///
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

int ReadStar::statData()
{
    int i;

    // ---- create internal Map of File16 ----
    if (file09)
        file16->createMap(file09->ntcell < file09->ncell);
    else
        file16->createMap(1);

    // ---- create Mesh from File16 info ----

    coDoUnstructuredGrid *mesh = new coDoUnstructuredGrid(p_mesh->getObjName(),
                                                          file16->numCovCells,
                                                          file16->numCovConn,
                                                          file16->maxn, 1);
    ERR0((mesh == NULL), "Error allocating Mesh", return STOP_PIPELINE;);
    p_mesh->setCurrentObject(mesh);

    int *clPtr, *tlPtr, *elPtr;
    float *xPtr, *yPtr, *zPtr;
    mesh->getAddresses(&elPtr, &clPtr, &xPtr, &yPtr, &zPtr);
    mesh->getTypeList(&tlPtr);

    // ---- Cell Table ----
    int size[2];
    size[1] = 10;
    size[0] = file16->mxtb;
    coDoIntArr *celltab = new coDoIntArr(p_celltab->getObjName(),
                                         2, size, (int *)(file16->cellType));
    p_celltab->setCurrentObject(celltab);

    // ---- Cell types ----
    int *typ;
    size[0] = file16->numCovCells;
    coDoIntArr *type = new coDoIntArr(p_type->getObjName(), 1, size);

    ERR0((type == NULL), "Error allocating Type Field", return STOP_PIPELINE;);
    type->getAddress(&typ);

    // ---- Read the Mesh and the type info from File16

    file16->getMesh(elPtr, clPtr, tlPtr, xPtr, yPtr, zPtr,
                    type->getAddress());

    // ---- Convert from ProSTAR units to Star units (after v3000)
    float fact = file16->getScale();
    sprintf(buf, "%f", fact);
    mesh->addAttribute("STAR_SCALE8", buf);

    if (file16->getVersion() >= 3000)
    {
        int i;
        if (fact > 0)
        {
            for (i = 0; i < file16->maxn; i++)
            {
                xPtr[i] *= fact;
                yPtr[i] *= fact;
                zPtr[i] *= fact;
            }
        }
        else
            sendInfo("Model file with 0 or negative Scale factor");
    }

    checkUSG(mesh);

    // --- set attributes for BlockCollect
    if (num_timesteps)
        addBlockInfo(mesh, timestep, num_timesteps);

    // --- create CP matching Poygons if matching exists
    int cpNumVert, cpNumConn, cpNumPoly;
    file16->getCPsizes(cpNumVert, cpNumConn, cpNumPoly);

    coDoPolygons *poly = new coDoPolygons(p_cpPoly->getObjName(),
                                          cpNumVert, cpNumConn, cpNumPoly);
    int *polyList, *connList;
    float *xPoly, *yPoly, *zPoly;
    poly->getAddresses(&xPoly, &yPoly, &zPoly, &connList, &polyList);
    if (cpNumPoly)
        file16->getCPPoly(xPtr, yPtr, zPtr,
                          xPoly, yPoly, zPoly, polyList, connList);
    poly->addAttribute("vertexOrder", "2");
    p_cpPoly->setCurrentObject(poly);

    // read data if we have a data file
    if (file09)
    {
        float *vdata[3] = { NULL, NULL, NULL };
        float *sdata[3] = { NULL, NULL, NULL };
        int createVertexData = (p_cellVert->getValue() == 0);
        int numData = file16->numCovCells;

        vdata[0] = new float[numData]; // always need at least 1st component
        sdata[0] = vdata[0]; // re-use 1st component in both

        // conversion Map: covise cells -> Star Cells (SAMM)
        const int *covToStar = file16->getCovToStar();

        for (i = 0; i < NUMPORTS; i++)
        {
            const char *Name = p_data[i]->getObjName();

            if (field_no[i] > 0)
            {
                int numRead;
                coDistributedObject *resData = NULL;
                if (field_no[i] == 1) // Velocity -> Vector
                {
                    //////// prepare field for reading
                    if (!vdata[1])
                    {
                        vdata[1] = new float[numData];
                        vdata[2] = new float[numData];
                    }

                    /// Now read it
                    numRead = file09->readField(field_no[i], vdata[0], vdata[1], vdata[2]);
                    //cerr << "Read " << numRead << " Elements" << endl;

                    //////// read Fields
                    ERR1((numRead < 0),
                         "Cannot read Velocity for Port Data%i", i + 1,
                         return STOP_PIPELINE;)

                    //////// if necessary, cellToVert and create object in here
                    if (createVertexData)
                        resData = cellToVert(mesh, vdata, Name, covToStar);

                    // otherwise copy and apply SAMM-trafo
                    else /// cell data: read directly
                    {
                        coDoVec3 *data
                            = new coDoVec3(Name, numData);
                        ERR0((data == NULL), "Error allocating Data Field", return STOP_PIPELINE;);
                        resData = data;
                        float *d0, *d1, *d2;
                        data->getAddresses(&d0, &d1, &d2);
                        int j;
                        //cerr << "Convert " << file16->numCovCells << " Elements" << endl;
                        //ofstream debug("TRANS");
                        for (j = 0; j < file16->numCovCells; j++)
                        {
                            //debug << j << '\t' << covToStar[j] << endl;
                            //
                            d0[j] = vdata[0][covToStar[j]];
                            //
                            d1[j] = vdata[1][covToStar[j]];
                            //
                            d2[j] = vdata[2][covToStar[j]];
                        }
                    }
                }

                else /// any scalar field
                {
                    /// Now read it
                    numRead = file09->readField(field_no[i], sdata[0], sdata[1], sdata[2]);

                    //////// read Fields
                    ERR1((numRead < 0),
                         "Cannot read Velocity for Port Data%i", i + 1,
                         return STOP_PIPELINE;)

                    //////// if necessary, cellToVert and create object in here
                    if (createVertexData)
                        resData = cellToVert(mesh, sdata, Name, covToStar);
                    else
                    {
                        coDoFloat *data
                            = new coDoFloat(Name, numData);
                        ERR0((data == NULL), "Error allocating Data Field", return STOP_PIPELINE;);
                        resData = data;
                        float *d;
                        data->getAddress(&d);
                        int j;
                        for (j = 0; j < file16->numCovCells; j++)
                        {
                            //debug << j << '\t' << covToStar[j] << endl;
                            //
                            d[j] = sdata[0][covToStar[j]];
                        }
                    }
                }

                //////// add attributes
                // remove trailing \n
                char *buffer = new char[strlen(fieldName[i]) + 1];
                strcpy(buffer, fieldName[i]);
                char *bPtr = buffer + strlen(buffer) - 1;
                if (*bPtr == '\n')
                    *bPtr = '\0';
                resData->addAttribute("SPECIES", buffer);
                delete[] buffer;

                if (num_timesteps)
                    addBlockInfo(resData, timestep, num_timesteps);
            }
        }

        delete[] vdata[0];
        delete[] vdata[1];
        delete[] vdata[2];
    }

    // done

    return CONTINUE_PIPELINE;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
///
///    Handle transient data
///
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

int ReadStar::transData()
{
    int i;

    // --- Cell Type Table : Always constant
    int size[2];
    size[1] = 10;
    size[0] = file16->mxtb;
    //coDoIntArr *celltab =
    //new coDoIntArr(p_celltab->getObjName(), 2, size, (int*) (file16->cellType) );
    //delete celltab;

    // --- Get stepping
    if (toStep > file29->get_num_steps())
        toStep = file29->get_num_steps();

    int num_elem = (toStep - fromStep) / byStep + 1;

#ifdef VERBOSE
    cerr << "Stepping from " << fromStep
         << " to " << toStep
         << " by steps of " << byStep
         << " giving " << num_elem << "Elemants" << endl;
#endif

    // --- Allocate object lists
    coDistributedObject **meshObjList
        = new coDistributedObject *[num_elem + 1];
    coDistributedObject **typeObjList
        = new coDistributedObject *[num_elem + 1];
    coDistributedObject **polyObjList
        = new coDistributedObject *[num_elem + 1];
    coDistributedObject **dataObjList[NUMPORTS];

    meshObjList[num_elem] = NULL;
    typeObjList[num_elem] = NULL;
    polyObjList[num_elem] = NULL;

    const char *Type = p_type->getObjName();
    const char *Mesh = p_mesh->getObjName();
    const char *Poly = p_cpPoly->getObjName();

    const char *Data[NUMPORTS];
    for (i = 0; i < NUMPORTS; i++)
    {
        dataObjList[i] = new coDistributedObject *[num_elem + 1];
        dataObjList[i][num_elem] = NULL;
        Data[i] = p_data[i]->getObjName();
    }

    char namebuf[512];
    int step_no, set_elem;

    // --- Allocation size for USG
    int new_cl, new_el, new_vl;

    // --- setup internal maps for File16
    file16->createMap(file29->ntcell < file29->ncell);

    // --- Mapping from (moving-grid/reduced) covise indices to star data indices
    int *redCovToStar = new int[file16->numCovCells];

    coDoUnstructuredGrid *usg;

    // --- Loop over timesteps
    for (step_no = fromStep, set_elem = 0;
         step_no <= toStep;
         step_no += byStep, set_elem++)
    {

#ifdef VERBOSE
        cerr << "Step " << step_no << "   element " << set_elem << endl;
        cerr << "NumStarCells: " << file16->numCovCells << endl;
#endif

        Covise::sendInfo("Step %d ", step_no);

        //////////////////// MESH ////////////////////

        // create Map and check consistency
        ERR2((file16->numOrigStarCells != file29->ncell),
             "Mesh and Data files not consistent:  Cells %i vs. %i",
             file16->numOrigStarCells, file29->ncell,
             return STOP_PIPELINE;)

        if (file29->lmvgrd) /// moving-Grid
        {
            // --- create temporary (full) mesh
            int *cl, *tl, *el;
            float *xv, *yv, *zv;
            cl = new int[file16->numCovConn];
            tl = new int[file16->numCovCells];
            el = new int[file16->numCovCells];
            xv = new float[file16->maxn];
            yv = new float[file16->maxn];
            zv = new float[file16->maxn];

            file29->getVertexCoordinates(step_no, xv, yv, zv, &new_vl);

            // --- create CP matching Poygons if matching exists
            int cpNumVert, cpNumConn, cpNumPoly;
            file16->getCPsizes(cpNumVert, cpNumConn, cpNumPoly);

            if (num_elem > 1)
            {
                sprintf(namebuf, "%s_%d", Poly, step_no);
            }
            else
            {
                strcpy(namebuf, Poly);
            }

            coDoPolygons *poly = new coDoPolygons(namebuf,
                                                  cpNumVert, cpNumConn, cpNumPoly);
            int *polyList, *connList;
            float *xPoly, *yPoly, *zPoly;
            poly->getAddresses(&xPoly, &yPoly, &zPoly, &connList, &polyList);
            if (cpNumPoly)
                file16->getCPPoly(xv, yv, zv,
                                  xPoly, yPoly, zPoly, polyList, connList);

            polyObjList[set_elem] = poly;

            // --- Cell Types ---
            int *tmpType = new int[file16->numCovCells];

            file16->getReducedMesh(el, cl, tl, redCovToStar, xv, yv, zv,
                                   &new_el, &new_cl, &new_vl,
                                   tmpType);

            // --- Write Type Array ---
            if (num_elem > 1)
            {
                sprintf(namebuf, "%s_%d", Type, step_no);
            }
            else
            {
                strcpy(namebuf, Type);
            }

            size[0] = new_el;
            typeObjList[set_elem] = new coDoIntArr(namebuf, 1, size, tmpType);
            ERR1((typeObjList[set_elem] == NULL),
                 "Error allocating Type '%s'", namebuf, return STOP_PIPELINE;)
            delete[] tmpType;

            // --- Mesh Object ---
            if (num_elem > 1)
            {
                sprintf(namebuf, "%s_%d", Mesh, step_no);
            }
            else
            {
                strcpy(namebuf, Mesh);
            }

            usg = new coDoUnstructuredGrid(namebuf, new_el, new_cl, new_vl,
                                           el, cl, xv, yv, zv, tl);
            ERR1((usg == NULL),
                 "Error allocating Mesh '%s'", namebuf, return STOP_PIPELINE;);
            meshObjList[set_elem] = usg;

            // --- remove unneeded and duplicate vertices
            checkUSG(usg);

            // --- Now place an attribute which realtime is read right now
            sprintf(buf, "%30.15f", file29->getRealTime(step_no));
            meshObjList[set_elem]->addAttribute("REALTIME", buf);

            // --- erase temporary fields
            delete[] cl;
            delete[] tl;
            delete[] el;
            delete[] xv;
            delete[] yv;
            delete[] zv;
        }

        else
        {
            /// ======== non-moving-Grid =======

            if (set_elem == 0)
            {
                // --- set sizes for data elements
                new_el = file16->numCovCells;
                new_cl = file16->numCovConn;
                new_vl = file16->maxn;

                // --- no movin-grid : all possible Cells are active ... best we know

                memcpy(redCovToStar, file16->getCovToStar(), new_el * sizeof(int));

                // --- create Map and check consistency
                file16->createMap(file29->ntcell < file29->ncell);

                // --- stationary case -> use grid from File16
                char *Mesh = Covise::get_object_name("mesh");
                ERR0((Mesh == NULL), "Error getting name 'mesh'", return STOP_PIPELINE;)

                if (num_elem > 1)
                {
                    sprintf(namebuf, "%s_%d", Mesh, step_no);
                }
                else
                {
                    strcpy(namebuf, Mesh);
                }
                usg = new coDoUnstructuredGrid(namebuf, file16->numCovCells,
                                               file16->numCovConn,
                                               file16->maxn, 1);
                ERR1((usg == NULL), "Error allocating '%s'", Mesh, return STOP_PIPELINE;);
                int *clPtr, *tlPtr, *elPtr;
                float *xPtr, *yPtr, *zPtr;
                usg->getAddresses(&elPtr, &clPtr, &xPtr, &yPtr, &zPtr);
                usg->getTypeList(&tlPtr);

                // --- cellTable
                char *Table = Covise::get_object_name("cellTable");
                if (num_elem > 1)
                {
                    sprintf(namebuf, "%s_%d", Table, step_no);
                }
                else
                {
                    strcpy(namebuf, Table);
                }

                ERR0((Table == NULL), "Error getting name 'cellTable'", return STOP_PIPELINE;)
                int size[2];
                size[1] = 10;
                size[0] = file16->mxtb;
                coDoIntArr *celltab = new coDoIntArr(namebuf, 2, size, (int *)(file16->cellType));
                delete celltab;

                // --- create CP matching Poygons if matching exists
                int cpNumVert, cpNumConn, cpNumPoly;
                file16->getCPsizes(cpNumVert, cpNumConn, cpNumPoly);

                const char *polyName = p_cpPoly->getObjName();
                if (num_elem > 1)
                {
                    sprintf(namebuf, "%s_%d", polyName, step_no);
                }
                else
                {
                    strcpy(namebuf, polyName);
                }

                coDoPolygons *poly = new coDoPolygons(namebuf,
                                                      cpNumVert, cpNumConn, cpNumPoly);
                int *polyList, *connList;
                float *xPoly, *yPoly, *zPoly;
                poly->getAddresses(&xPoly, &yPoly, &zPoly, &connList, &polyList);
                if (cpNumPoly)
                    file16->getCPPoly(xPtr, yPtr, zPtr,
                                      xPoly, yPoly, zPoly, polyList, connList);
                poly->addAttribute("vertexOrder", "2");
                polyObjList[0] = poly;

                // --- cell types;

                int *typ;
                char *Type = Covise::get_object_name("type");
                ERR0((Type == NULL), "Error getting name 'type'", return STOP_PIPELINE;)
                size[0] = file16->numCovCells;

                if (num_elem > 1)
                {
                    sprintf(namebuf, "%s_%d", Type, step_no);
                }
                else
                {
                    strcpy(namebuf, Type);
                }

                coDoIntArr *type = new coDoIntArr(namebuf, 1, size);
                ERR1((type == NULL), "Error allocating '%s'", Type, return STOP_PIPELINE;);
                type->getAddress(&typ);

                // --- get the mesh

                file16->getMesh(elPtr, clPtr, tlPtr, xPtr, yPtr, zPtr,
                                type->getAddress());

                // ---- Convert from ProSTAR units to Star units (after v3000)
                float fact = file16->getScale();
                sprintf(buf, "%f", fact);
                usg->addAttribute("STAR_SCALE8", buf);

                if (file16->getVersion() >= 3000)
                {
                    int i;
                    if (fact > 0)
                    {
                        for (i = 0; i < file16->maxn; i++)
                        {
                            xPtr[i] *= fact;
                            yPtr[i] *= fact;
                            zPtr[i] *= fact;
                        }
                    }
                    else
                        sendInfo("Model file with 0 or negative Scale factor");
                }

                meshObjList[0] = usg;
                typeObjList[0] = type;

                // --- remove unused and duplicate vertices
                checkUSG(usg);
            }
            else
            {
                meshObjList[set_elem] = meshObjList[0];
                meshObjList[0]->incRefCount();
                typeObjList[set_elem] = typeObjList[0];
                typeObjList[0]->incRefCount();
                polyObjList[set_elem] = polyObjList[0];
                polyObjList[0]->incRefCount();
                usg = NULL;
            }
        }

        //////////////////// DATA ////////////////////

        float *vdata[3] = { NULL, NULL, NULL };
        float *sdata[3] = { NULL, NULL, NULL };
        int createVertexData = (p_cellVert->getValue() == 0);

        vdata[0] = new float[new_el]; // always need at least 1st component
        sdata[0] = vdata[0]; // re-use 1st component in both

        for (i = 0; i < NUMPORTS; i++)
        {
            // --- create internal object name
            if (num_elem > 1)
            {
                sprintf(namebuf, "%s_%i", Data[i], step_no);
            }
            else
            {
                strcpy(namebuf, Data[i]);
            }

            if (field_no[i] > 0)
            {
                int numRead;
                coDistributedObject *resData = NULL;

                // --- Velocity -> Vector
                if (field_no[i] == 1)
                {
                    // --- prepare field for reading
                    if (!vdata[1])
                    {
                        vdata[1] = new float[new_el];
                        vdata[2] = new float[new_el];
                    }
                    // if (file29->numDrop(step_no) >numData)
                    //  re-alloc
                    numRead = file29->readField(step_no, field_no[i], redCovToStar, new_el,
                                                vdata[0], vdata[1], vdata[2]);

                    // --- read Fields
                    ERR1((numRead < 0),
                         "Cannot read Velocity for Port Data%i", i + 1,
                         return STOP_PIPELINE;)

                    // --- if necessary, convert to vertex and create object in here
                    if (createVertexData)
                        resData = cellToVert(usg, vdata, namebuf);

                    // otherwise copy (Cell-Data) and apply SAMM-trafo
                    else
                    {
                        coDoVec3 *data
                            = new coDoVec3(namebuf, new_el);
                        ERR0((data == NULL), "Error allocating Data Field", return STOP_PIPELINE;);
                        resData = data;
                        float *d0, *d1, *d2;
                        data->getAddresses(&d0, &d1, &d2);
                        int j;
                        for (j = 0; j < new_el; j++)
                        {
                            d0[j] = vdata[0][j]; //
                            d1[j] = vdata[1][j]; //
                            d2[j] = vdata[2][j]; //
                        }
                    }
                }

                /////////////////   DROPLET

                // --- Droplet Coordinates -> Points
                else if (field_no[i] == StarFile::DROP_COORD)
                {
                    int numDrops = file29->getNumDrops(step_no);
                    coDoPoints *data
                        = new coDoPoints(namebuf, numDrops);

                    // remove trailing \n
                    char *buffer = new char[strlen(fieldName[i]) + 1];
                    strcpy(buffer, fieldName[i]);
                    char *bPtr = buffer + strlen(buffer) - 1;
                    if (*bPtr == '\n')
                        *bPtr = '\0';
                    data->addAttribute("SPECIES", (char *)fieldName[i]);
                    delete[] buffer;

                    dataObjList[i][set_elem] = data;
                    ERR1((data == NULL), "Error allocating '%s'", buf, return STOP_PIPELINE;);

                    float *vx, *vy, *vz;
                    data->getAddresses(&vx, &vy, &vz);
                    int numRead = file29->readField(step_no, field_no[i],
                                                    redCovToStar, numDrops, vx, vy, vz);

                    ERR1((numRead < 0),
                         "Cannot read Drop Velocity for Port Data%i", i,
                         return STOP_PIPELINE;)

                    data->setSize(numRead);
                    resData = data;
                }

                // --- Droplet Velocity -> Vector
                else if (field_no[i] == StarFile::DROP_VEL)
                {
                    int numDrops = file29->getNumDrops(step_no);
                    coDoVec3 *data
                        = new coDoVec3(namebuf, numDrops);

                    // remove trailing \n
                    char *buffer = new char[strlen(fieldName[i]) + 1];
                    strcpy(buffer, fieldName[i]);
                    char *bPtr = buffer + strlen(buffer) - 1;
                    if (*bPtr == '\n')
                        *bPtr = '\0';
                    data->addAttribute("SPECIES", (char *)fieldName[i]);
                    delete[] buffer;

                    dataObjList[i][set_elem] = data;
                    ERR1((data == NULL), "Error allocating '%s'", buf, return STOP_PIPELINE;);

                    float *vx, *vy, *vz;
                    data->getAddresses(&vx, &vy, &vz);
                    int numRead = file29->readField(step_no, field_no[i],
                                                    redCovToStar, numDrops, vx, vy, vz);
                    ERR1((numRead < 0),
                         "Cannot read Drop Velocity for Port Data%i", i, return STOP_PIPELINE;)
                    data->setSize(numRead);
                    resData = data;
                }

                // --- Droplet Scalar Data -> Scalars
                else if (field_no[i] >= StarFile::DROP_DENS
                         && field_no[i] <= StarFile::DROP_MASS)
                {
                    int numDrops = file29->getNumDrops(step_no);
                    coDoFloat *data
                        = new coDoFloat(namebuf, numDrops);
                    dataObjList[i][set_elem] = data;

                    // remove trailing \n
                    char *buffer = new char[strlen(fieldName[i]) + 1];
                    strcpy(buffer, fieldName[i]);
                    char *bPtr = buffer + strlen(buffer) - 1;
                    if (*bPtr == '\n')
                        *bPtr = '\0';
                    data->addAttribute("SPECIES", (char *)fieldName[i]);
                    delete[] buffer;

                    ERR1((data == NULL), "Error allocating '%s'", buf, return STOP_PIPELINE;);
                    float *v;
                    data->getAddress(&v);
                    int numRead = file29->readField(step_no, field_no[i],
                                                    redCovToStar, numDrops, v, NULL, NULL);
                    ERR1((numRead < 0),
                         "cannot read Data for Port Data%i", i + 1, return STOP_PIPELINE;)
                    data->setSize(numRead);
                    resData = data;
                }

                //////////////////////////////////////////////////////////////////
                else /// any scalar field
                {
                    numRead = file29->readField(step_no, field_no[i], redCovToStar, new_el,
                                                sdata[0], sdata[1], sdata[2]);

                    // --- read Fields
                    ERR1((numRead < 0),
                         "Cannot read Velocity for Port Data%i", i + 1,
                         return STOP_PIPELINE;)

                    // --- if necessary, cellToVert and create object in here
                    //cerr << namebuf << "  " << new_el << endl;
                    if (createVertexData)
                        resData = cellToVert(usg, sdata, namebuf);

                    // --- otherwise copy and apply SAMM-trafo
                    else /// cell data: read directly
                    {
                        coDoFloat *data
                            = new coDoFloat(namebuf, new_el);
                        ERR0((data == NULL), "Error allocating Data Field", return STOP_PIPELINE;);
                        resData = data;
                        float *d;
                        data->getAddress(&d);
                        int j;
                        for (j = 0; j < new_el; j++)
                        {
                            //debug << j << '\t' << trans[j] << endl;
                            d[j] = sdata[0][j]; //
                        }
                    }
                }

                // --- add attributes
                // remove trailing \n
                char *buffer = new char[strlen(fieldName[i]) + 1];
                strcpy(buffer, fieldName[i]);
                char *bPtr = buffer + strlen(buffer) - 1;
                if (*bPtr == '\n')
                    *bPtr = '\0';
                resData->addAttribute("SPECIES", (char *)fieldName[i]);
                delete[] buffer;

                // --- Now place an attribute which realtime is read right now
                sprintf(buf, "%30.15f", file29->getRealTime(step_no));
                resData->addAttribute("REALTIME", buf);

                if (num_timesteps)
                {
                    addBlockInfo(resData, timestep, num_timesteps);
                }
                dataObjList[i][set_elem] = resData;
            }
        }

        delete[] vdata[0];
        delete[] vdata[1];
        delete[] vdata[2];
    }

    /////////////////////////////////////////////////////////////////
    /////////// You got the objects - Now build the sets  ///////////

    /// build sets only for >1 step
    if (num_elem > 1)
    {
        char timestepattr[256];
        sprintf(timestepattr, "%d %d", 0, num_elem - 1);

        // --- Mesh
        coDoSet *set = new coDoSet(Mesh, meshObjList);
        ERR0((set == NULL), "Error getting name 'mesh'", return STOP_PIPELINE;)

        // --- the droplet reader needs this factor
        if (file16->getVersion() >= 3000)
        {
            sprintf(buf, "%f", file16->getScale());
            set->addAttribute("STAR_SCALE8", buf);
        }

        if (num_elem > 1)
        {
            set->addAttribute("TIMESTEP", timestepattr);
        }

        if (file29->lmvgrd) // --- moving grid - delete all
        {
            for (i = 0; i < num_elem; i++)
            {
                delete meshObjList[i];
            }
        }
        else
        {
            delete meshObjList[0]; // --- stationary grid - delete 1st, others are same
        }
        delete[] meshObjList;

        if (num_timesteps)
        {
            addBlockInfo(set, timestep, num_timesteps);
        }

        //delete set;
        p_mesh->setCurrentObject(set);

        // --- Typelist
        set = new coDoSet(Type, typeObjList);
        ERR0((set == NULL), "Error getting name 'mesh'", return STOP_PIPELINE;)
        if (num_elem > 1)
        {
            set->addAttribute("TIMESTEP", timestepattr);
        }

        if (file29->lmvgrd)
        {
            for (i = 0; i < num_elem; i++)
            {
                delete typeObjList[i];
            }
        }
        else
        {
            delete typeObjList[0];
        }
        delete[] typeObjList;
        //delete set;
        p_type->setCurrentObject(set);

        // --- Polygons for Coupled Sets
        set = new coDoSet(Poly, polyObjList);
        ERR0((set == NULL), "Error getting name 'mesh'", return STOP_PIPELINE;)
        if (num_elem > 1)
        {
            set->addAttribute("TIMESTEP", timestepattr);
        }

        if (file29->lmvgrd)
        {
            for (i = 0; i < num_elem; i++)
            {
                delete polyObjList[i];
            }
        }
        else
        {
            delete polyObjList[0];
        }
        delete[] polyObjList;
        //delete set;
        p_cpPoly->setCurrentObject(set);

        // --- Data

        int port;

        for (port = 0; port < NUMPORTS; port++)
        {
            if (field_no[port] > 0)
            {
                coDoSet *set = new coDoSet(Data[port], dataObjList[port]);
                ERR1((set == NULL), "Error creating '%s'", Data[port], return STOP_PIPELINE;)
                if (num_elem > 1)
                {
                    set->addAttribute("TIMESTEP", timestepattr);
                }

                for (i = 0; i < num_elem; i++)
                {
                    delete (dataObjList[port])[i];
                }
                delete[](dataObjList[port]);
                //delete set;
                p_data[port]->setCurrentObject(set);
            }
        }
    }

    //// single step of transient data set
    else
    {
        // --- Mesh
        if (file16->getVersion() >= 3000)
        {
            sprintf(buf, "%f", file16->getScale());
            meshObjList[0]->addAttribute("STAR_SCALE8", buf);
        }
        //delete meshObjList[0];
        p_mesh->setCurrentObject(meshObjList[0]);
        delete[] meshObjList;

        // --- Typelist
        //delete typeObjList[0];
        p_type->setCurrentObject(typeObjList[0]);
        delete[] typeObjList;

        // --- Polygons for Coupled Sets
        //delete polyObjList[0];
        p_cpPoly->setCurrentObject(polyObjList[0]);
        delete[] polyObjList;

        // --- Data
        int port;

        for (port = 0; port < NUMPORTS; port++)
        {
            if (field_no[port] > 0)
            {
                //delete (dataObjList[port])[0];
                p_data[port]->setCurrentObject((dataObjList[port])[0]);
                delete (dataObjList[port]);
            }
        }
    }

#ifdef DEBUGFILES
    fclose(dataFile);
#endif

    return CONTINUE_PIPELINE;
}

/////////////////////////////////////////////////////////////////////
//
//   CheckUSG auf internen Feldern -> es kann nur kleiner werden...
//   Code von CellToVert kopiert.
//
//   Returns Map newCellId [ oldCellID ] -> to be deleted by prog
//
/////////////////////////////////////////////////////////////////////

/////// find the bounding box

inline void boundingBox(float **x, float **y, float **z, int *c, int n,
                        float *bbx1, float *bby1, float *bbz1,
                        float *bbx2, float *bby2, float *bbz2)
{
    int i;
    float cx, cy, cz;

    *bbx1 = *bbx2 = (*x)[c[0]];
    *bby1 = *bby2 = (*y)[c[0]];
    *bbz1 = *bbz2 = (*z)[c[0]];

    for (i = 0; i < n; i++)
    {
        cx = (*x)[c[i]];
        cy = (*y)[c[i]];
        cz = (*z)[c[i]];

        // x
        if (cx < *bbx1)
            *bbx1 = cx;
        else if (cx > *bbx2)
            *bbx2 = cx;

        // y
        if (cy < *bby1)
            *bby1 = cy;
        else if (cy > *bby2)
            *bby2 = cy;

        // z
        if (cz < *bbz1)
            *bbz1 = cz;
        else if (cz > *bbz2)
            *bbz2 = cz;
    }
    return;
}

/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////

inline int getOctant(float x, float y, float z, float ox, float oy, float oz)
{
    int r;

    // ... the origin

    if (x < ox) // behind

        if (y < oy) // below
            if (z < oz) // left
                r = 6;
            else // right
                r = 7;
        else // above
            if (z < oz) // left
            r = 4;
        else // right
            r = 5;

    else // in front of
        if (y < oy) // below
        if (z < oz) // left
            r = 2;
        else // right
            r = 3;
    else // above
        if (z < oz) // left
        r = 0;
    else // right
        r = 1;

    // done
    return (r);
}

inline void getOctantBounds(int o, float ox, float oy, float oz,
                            float bbx1, float bby1, float bbz1,
                            float bbx2, float bby2, float bbz2,
                            float *bx1, float *by1, float *bz1,
                            float *bx2, float *by2, float *bz2)
{
    switch (o)
    {
    case 0: // left, above, front
        *bx1 = bbx1;
        *by1 = oy;
        *bz1 = oz;
        *bx2 = ox;
        *by2 = bby2;
        *bz2 = bbz2;
        break;
    case 1: // right, above, front
        *bx1 = ox;
        *by1 = oy;
        *bz1 = oz;
        *bx2 = bbx2;
        *by2 = bby2;
        *bz2 = bbz2;
        break;
    case 2: // left, below, front
        *bx1 = bbx1;
        *by1 = bby1;
        *bz1 = oz;
        *bx2 = ox;
        *by2 = oy;
        *bz2 = bbz2;
        break;
    case 3: // right, below, front
        *bx1 = ox;
        *by1 = bby1;
        *bz1 = oz;
        *bx2 = bbx2;
        *by2 = oy;
        *bz2 = bbz2;
        break;
    case 4: // left, above, behind
        *bx1 = bbx1;
        *by1 = oy;
        *bz1 = bbz1;
        *bx2 = ox;
        *by2 = bby2;
        *bz2 = oz;
        break;
    case 5: // right, above, behind
        *bx1 = ox;
        *by1 = oy;
        *bz1 = bbz1;
        *bx2 = bbx2;
        *by2 = bby2;
        *bz2 = oz;
        break;
    case 6: // left, below, behind
        *bx1 = bbx1;
        *by1 = bby1;
        *bz1 = bbz1;
        *bx2 = ox;
        *by2 = oy;
        *bz2 = oz;
        break;
    case 7: // right, below, behind
        *bx1 = ox;
        *by1 = bby1;
        *bz1 = bbz1;
        *bx2 = bbx2;
        *by2 = oy;
        *bz2 = oz;
        break;
    }

    return;
}

void ReadStar::computeCell(float *xcoord, float *ycoord, float *zcoord,
                           int *coordInBox, int numCoordInBox,
                           float bbx1, float bby1, float bbz1,
                           float bbx2, float bby2, float bbz2,
                           int maxCoord, int *replBy, int &numCoordToRemove)
{
    int i, j;
    int v, w;
    float rx, ry, rz;
    float obx1 = 0.f, oby1 = 0.f, obz1 = 0.f;
    float obx2 = 0.f, oby2 = 0.f, obz2 = 0.f;
    int numCoordInCell[8];
    int *coordInCell[8];

    // too many Coords in my box -> split octree box deeper
    if (numCoordInBox > maxCoord)
    {
        // yes we have
        rx = (bbx1 + bbx2) / 2.0f;
        ry = (bby1 + bby2) / 2.0f;
        rz = (bbz1 + bbz2) / 2.0f;

        // go through the coordinates and sort them in the right cell

        for (i = 0; i < 8; i++)
        {
            coordInCell[i] = new int[numCoordInBox];
            numCoordInCell[i] = 0;
        }

        for (i = 0; i < numCoordInBox; i++)
        {
            v = coordInBox[i];
            w = getOctant(xcoord[v], ycoord[v], zcoord[v], rx, ry, rz);
            coordInCell[w][numCoordInCell[w]] = v;
            numCoordInCell[w]++;
        }

        // we're recursive - hype
        for (i = 0; i < 8; i++)
        {
            if (numCoordInCell[i])
            {
                if (numCoordInCell[i] > numCoordInBox / 4)
                {
                    // we decide to compute the new BoundingBox instead of
                    // just splitting the parent-Box
                    boundingBox(&xcoord, &ycoord, &zcoord, coordInCell[i],
                                numCoordInCell[i], &obx1, &oby1, &obz1,
                                &obx2, &oby2, &obz2);
                }
                else
                    getOctantBounds(i, rx, ry, rz, bbx1, bby1, bbz1,
                                    bbx2, bby2, bbz2,
                                    &obx1, &oby1, &obz1, &obx2, &oby2, &obz2);

                computeCell(xcoord, ycoord, zcoord, coordInCell[i],
                            numCoordInCell[i], obx1, oby1, obz1,
                            obx2, oby2, obz2, maxCoord, replBy, numCoordToRemove);
            }
            delete[] coordInCell[i];
        }
    }

    //// directly compare in box
    else if (numCoordInBox > 1)
    {
        // check these vertices
        for (i = 0; i < numCoordInBox - 1; i++)
        {
            v = coordInBox[i];
            rx = xcoord[v];
            ry = ycoord[v];
            rz = zcoord[v];
            // see if this one is doubled
            for (j = i + 1; j < numCoordInBox; j++)
            {
                w = coordInBox[j];

                if (xcoord[w] == rx && // @@@@ add distance fkt here if necessary
                    ycoord[w] == ry && zcoord[w] == rz)
                {
                    // this one is double
                    if (v < w)
                        replBy[w] = v;
                    else
                        replBy[v] = w;
                    numCoordToRemove++;
                    // break out
                    j = numCoordInBox;
                }
            }
        }
    }

    // done
    return;
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

void ReadStar::computeReplaceLists(int num_coord, int *replBy,
                                   int *&src2filt, int *&filt2src)
{
    int i, k;

    // now unlink the temporary list
    for (i = 0; i < num_coord; i++)
    {
        k = replBy[i];
        if (k >= 0)
        {
            // this one will be replaced, so unlink the list
            while (replBy[k] >= 0)
                k = replBy[k];

            if (replBy[k] == -1)
                // remove this one
                replBy[i] = -1;
            else
                // replace this one
                replBy[i] = k;
        }
    }

    // allocate mem
    src2filt = new int[num_coord]; // original vertex i is replaced by s2f[i]

    // forward filter
    int numFiltered = 0;
    for (i = 0; i < num_coord; i++)
    {
        // vertex untouched
        if (replBy[i] == -2)
        {
            src2filt[i] = numFiltered;
            numFiltered++;
        }
        // vertex replaced: replacer < replacee
        else if (replBy[i] >= 0)
        {
            src2filt[i] = src2filt[replBy[i]];
        }
        else
        {
            src2filt[i] = -1;
        }
    }

    // backward filter
    filt2src = new int[numFiltered];
    for (i = 0; i < num_coord; i++)
        if (src2filt[i] >= 0)
            filt2src[src2filt[i]] = i;

    // done
    return;
}

//////////////////////////////////////////////////////////////////

int ReadStar::checkUSG(coDoUnstructuredGrid *grid)
{
    int i, num_elem, num_conn, num_coord, *elemList, *connList;
    float *xcoord, *ycoord, *zcoord;

    grid->getGridSize(&num_elem, &num_conn, &num_coord);
    grid->getAddresses(&elemList, &connList, &xcoord, &ycoord, &zcoord);

    /// create a Replace-List
    int *replBy = new int[num_coord];
    for (i = 0; i < num_coord; i++)
        replBy[i] = -2;

    int numCoordToRemove = 0;

    int *coordInBox = new int[num_coord];
    if (!coordInBox)
    {
        coModule::sendError("Could not allocate memory");
        return 0;
    }
    int numCoordInBox = 0;

    // the "starting" cell contains all USED coordinates
    // clear all flags -> no coordinates used at all
    for (i = 0; i < num_coord; i++)
        coordInBox[i] = 0;

    for (i = 0; i < num_conn; i++)
        coordInBox[connList[i]] = 1;

    // now remove the unused coordinates
    for (i = 0; i < num_coord; i++)
    {
        if (coordInBox[i])
        {
            // this one is used
            coordInBox[numCoordInBox] = i;
            numCoordInBox++;
        }
        else
        {
            // unused coordinate
            replBy[i] = -1;
            numCoordToRemove++;
        }
    }

    float bbx1, bby1, bbz1;
    float bbx2, bby2, bbz2;

    // find the bounding box
    boundingBox(&xcoord, &ycoord, &zcoord, coordInBox, numCoordInBox,
                &bbx1, &bby1, &bbz1, &bbx2, &bby2, &bbz2);

    const int maxCoord = 50; // elements for direct sort @@@

    computeCell(xcoord, ycoord, zcoord,
                coordInBox, numCoordInBox, bbx1, bby1, bbz1,
                bbx2, bby2, bbz2, maxCoord, replBy, numCoordToRemove);

    // partially clean up
    delete[] coordInBox;

    // obly if we found vertices to remove...
    if (numCoordToRemove)
    {
        // compute the lists of replacements (both directions)
        int *src2filt, *filt2src;
        computeReplaceLists(num_coord, replBy, src2filt, filt2src);
        delete[] replBy;

        ////// ---------- Filter Grid ----------
        int newNumCoord = num_coord - numCoordToRemove;
        int newIdx = 0;

        // skip thru first changed idx (we DO replace at least one, see above)
        while (filt2src[newIdx] == newIdx)
            newIdx++;

        // and now replace coordinates
        do
        {
            int oldIdx = filt2src[newIdx];
            xcoord[newIdx] = xcoord[oldIdx];
            ycoord[newIdx] = ycoord[oldIdx];
            zcoord[newIdx] = zcoord[oldIdx];
            newIdx++;
        } while (newIdx < newNumCoord);

        // and replace connectivities
        for (i = 0; i < num_conn; i++)
            connList[i] = src2filt[connList[i]];

        // set the new sizes
        grid->setSizes(num_elem, num_conn, newNumCoord);

        // clean up
        delete[] src2filt;
        delete[] filt2src;

        // status report
        //char buffer[64];
        //sprintf(buffer,"removed %d points",numCoordToRemove);
        //Covise::sendInfo(buffer);

        return newNumCoord;
    }
    else
        return num_coord;
}

/////////////////////////////////////////////////////////////////
// the 'included' CellToVert module : return new object
coDistributedObject *ReadStar::cellToVert(coDoUnstructuredGrid *grid,
                                          float *elemData[3],
                                          const char *name,
                                          const int *trans)
{
    int numCell, numVert, numConn;
    int *vertexList, *vertCellList;
    coDistributedObject *dataObj;
    float *vd0, *vd1, *vd2;
    float *&ed0 = elemData[0];
    float *&ed1 = elemData[1];
    float *&ed2 = elemData[2];

    grid->getGridSize(&numCell, &numConn, &numVert);
    grid->getNeighborList(&numConn, &vertCellList, &vertexList);

    int i, j;
    int vstart, vend;
    vstart = vertexList[0];
    int *vListPtr = vertexList + 1;

    // build new list for cell trafo
    // int *dataIdx = new int[numCell];
    // for (i=0;i<numCell;i++)
    //   dataIdx[i] = trans[vertCellList[j]];

    /// split here for speed vector/scalar
    if (elemData[1])
    {
        coDoVec3 *v3d = new coDoVec3(name, numVert);

        ERR1((v3d == NULL), "Error allocating '%s'", name,
             return NULL;);
        v3d->getAddresses(&vd0, &vd1, &vd2);

        for (i = 0; i < numVert; i++)
        {
            vd0[i] = 0.0;
            vd1[i] = 0.0;
            vd2[i] = 0.0;
            vend = *vListPtr;
            for (j = vstart; j < vend; j++)
            {
                vd0[i] += ed0[trans[vertCellList[j]]]; //  ];       dataIdx[j]
                vd1[i] += ed1[trans[vertCellList[j]]]; //  ];       dataIdx[j]
                vd2[i] += ed2[trans[vertCellList[j]]]; //  ];       dataIdx[j]
            }
            float fact = 1.0f / (vend - vstart);
            vd0[i] *= fact;
            vd1[i] *= fact;
            vd2[i] *= fact;

            vstart = vend;
            vListPtr++;
        }
        dataObj = v3d;
    }
    else
    {
        coDoFloat *s3d = new coDoFloat(name, numVert);

        ERR1((s3d == NULL), "Error allocating '%s'", name,
             return NULL;);
        s3d->getAddress(&vd0);

        for (i = 0; i < numVert; i++)
        {
            vd0[i] = 0.0;
            vend = *vListPtr;
            for (j = vstart; j < vend; j++)
            {
                vd0[i] += ed0[trans[vertCellList[j]]]; //  ];dataIdx[j]
            }
            vd0[i] /= vend - vstart;

            vstart = vend;
            vListPtr++;
        }

        dataObj = s3d;
    }

    return dataObj;
}

//////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
// the 'included' CellToVert module : return new object
coDistributedObject *ReadStar::cellToVert(coDoUnstructuredGrid *grid,
                                          float *elemData[3],
                                          const char *name)
{
    int numCell, numVert, numConn;
    int *vertexList, *vertCellList;
    coDistributedObject *dataObj;
    float *vd0, *vd1, *vd2;
    float *&ed0 = elemData[0];
    float *&ed1 = elemData[1];
    float *&ed2 = elemData[2];

    grid->getGridSize(&numCell, &numConn, &numVert);
    grid->getNeighborList(&numConn, &vertCellList, &vertexList);

    int i, j;
    int vstart, vend;
    vstart = vertexList[0];
    int *vListPtr = vertexList + 1;

    // build new list for cell trafo
    // int *dataIdx = new int[numCell];
    // for (i=0;i<numCell;i++)
    //   dataIdx[i] = trans[vertCellList[j]];

    /// split here for speed vector/scalar
    if (elemData[1])
    {
        coDoVec3 *v3d = new coDoVec3(name, numVert);

        ERR1((v3d == NULL), "Error allocating '%s'", name,
             return NULL;);
        v3d->getAddresses(&vd0, &vd1, &vd2);

        for (i = 0; i < numVert; i++)
        {
            vd0[i] = 0.0;
            vd1[i] = 0.0;
            vd2[i] = 0.0;
            vend = *vListPtr;
            for (j = vstart; j < vend; j++)
            {
                vd0[i] += ed0[vertCellList[j]]; //  ];       dataIdx[j]
                vd1[i] += ed1[vertCellList[j]]; //  ];       dataIdx[j]
                vd2[i] += ed2[vertCellList[j]]; //  ];       dataIdx[j]
            }
            float fact = 1.0f / (vend - vstart);
            vd0[i] *= fact;
            vd1[i] *= fact;
            vd2[i] *= fact;

            vstart = vend;
            vListPtr++;
        }
        dataObj = v3d;
    }
    else
    {
        coDoFloat *s3d = new coDoFloat(name, numVert);

        ERR1((s3d == NULL), "Error allocating '%s'", name,
             return NULL;);
        s3d->getAddress(&vd0);

        for (i = 0; i < numVert; i++)
        {
            vd0[i] = 0.0;
            vend = *vListPtr;
            for (j = vstart; j < vend; j++)
            {
                vd0[i] += ed0[vertCellList[j]]; //  ];dataIdx[j]
            }
            vd0[i] /= vend - vstart;

            vstart = vend;
            vListPtr++;
        }

        dataObj = s3d;
    }

    return dataObj;
}

MODULE_MAIN(Reader, ReadStar)
