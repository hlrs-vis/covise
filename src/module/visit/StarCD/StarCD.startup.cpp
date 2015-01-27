/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoText.h>
#include "StarCD.h"
#include "Mesh.h"
#include <ctype.h>
#include <string.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#undef VERBOSE

/// get a line chars from a filebuffer
//     - skip indents
//     - remove end-comments
//     - skip empty lines
//     - remove end-blanks
//     - advances fileBuffer pointer
//     - converts \n in \0 chars in buffer
// returns pointer to null-terminated string
static const char *getLine(char *&fileBuffer)
{
    char *res;
    do
    {

        // we are at end-of-file or have no file at all
        if (!fileBuffer || !*fileBuffer)
            return NULL;

        // remove leading blanks
        while ((*fileBuffer) && (isspace(*fileBuffer)))
            fileBuffer++;

        // This is what we return
        res = fileBuffer;

        // advance fileBuffer to end-of-file or next \n
        while (*fileBuffer && *fileBuffer != '\n')
            fileBuffer++;

        // teminate string and advance if not at end-of-file
        if (*fileBuffer)
        {
            *fileBuffer = '\0';
            fileBuffer++;
        }

        // remove comments
        char *cPtr = strchr(res, '#');
        if (cPtr)
            *cPtr = '\0';
        else
            cPtr = fileBuffer;

        // remove trailing blanks
        cPtr--;
        int len = strlen(res);
        if (res)
        {
            cPtr = res + len - 1;
            while ((*res) && isspace(*res))
            {
                *res = '\0';
                res--;
            }
        }
    } while (*res == '\0');

#ifdef VERBOSE
    cerr << "getLine() returns '" << res << "'" << endl;
#endif

    return res;
}

// get the result of an option line or NULL if the option wasn't found
static void getOption(const char *option, const char *buffer, char *&resVal)
{
    int len = strlen(option);
    if (strncasecmp(option, buffer, len) == 0)
    {
        buffer += len;
        while ((*buffer) && (isspace(*buffer)))
            buffer++;
        if (*buffer)
        {
            delete[] resVal;
            resVal = strcpy(new char[strlen(buffer) + 1], buffer);
#ifdef VERBOSE
            cerr << "Found Option " << option << " as '" << resVal << "'" << endl;
#endif
        }
        else
            StarCD::getModule()->sendWarning("Defined %s to empty string", option);
    }
    return;
}

///////////////////////////////////////////////////////////////////////////////
/// copy a complete slider and de-activate it
static void sliderCopy(coFloatSliderParam *newSlider,
                       const coFloatSliderParam *oldSlider)
{
    float x, y, z;
    oldSlider->getValue(x, y, z);
    newSlider->setValue(x, y, z);
    newSlider->setActive(1);
    newSlider->enable();
}

///////////////////////////////////////////////////////////////////////////////
// setup the general case parameters : returns the first 'region' line in 'line'

int StarCD::setupCase(char *&filebuffer, const char *&line)
{
#ifdef VERBOSE
    cerr << "Called setupCase(filebuffer,line)\n -----------\n"
         << filebuffer << "\n-----------\n"
         << endl;
#endif
    // read one line
    line = getLine(filebuffer);

    // first loop: get the options
    while (line && *line
           && (strncasecmp("region", line, 6)) != 0)
    {
        // getOption tries to retrieve option lines and doesn't change anything
        // if the option wasn't found
        getOption("USER", line, d_user);
        getOption("HOST", line, d_host);
        getOption("COMPDIR", line, d_compDir);
        getOption("CASE", line, d_case);
        getOption("SCRIPT", line, d_script);
        getOption("MESHDIR", line, d_meshDir);
        getOption("CREATOR", line, d_creator);
        getOption("USR0", line, d_usr[0]);
        getOption("USR1", line, d_usr[1]);
        getOption("USR2", line, d_usr[2]);
        getOption("USR3", line, d_usr[3]);
        getOption("USR4", line, d_usr[4]);
        getOption("USR5", line, d_usr[5]);
        getOption("USR6", line, d_usr[6]);
        getOption("USR7", line, d_usr[7]);
        getOption("USR8", line, d_usr[8]);
        getOption("USR8", line, d_usr[9]);

        // read next line
        line = getLine(filebuffer);
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// setup the region parameters
// get the first line in 'line' and then read the rest from filebuffer

int StarCD::setupRegions(char *&filebuffer, const char *&line)
{
#ifdef VERBOSE
    cerr << "Called setupRegions(filebuffer,'"
         << line << "')\n -----------\n"
         << filebuffer << "\n-----------\n"
         << endl;
#endif

    // This is our 'regions' choice: first label is always the title
    choices[0] = strcpy(new char[20], "Select Region");

    // Flag: forget lat region when set
    bool killLastRegion = false;

    int actRegion = flags.iconum - 1; // C counting: start with 0, FTN starts with 1
    while (line)
    {
        // This command starts a new region
        if (strncasecmp("region", line, 6) == 0)
        {
            if (!killLastRegion)
                actRegion++;
            else
                killLastRegion = false;

            char *name = const_cast<char *>(strchr(line, ':'));
            char *like = const_cast<char *>(strrchr(line, ':'));

#ifdef VERBOSE
            cerr << "Found Region: '" << name << endl;
            ;
#endif

            if (!name)
            {
                sendWarning("Illegal region line '%s'", line);
                break;
            }
            *name = '\0';

            if (like == name)
                like = NULL;
            else
            {
                *like = '\0';
                like++;
            }

            // skip the ':' and all following blanks
            name++;
            while (*name && isspace(*name))
                name++;

            // set the choice label
            choices[actRegion + 1] = strcpy(new char[strlen(name) + 3], name);

            // get the region number
            flags.icoreg[actRegion] = atoi(line + 6);

            // if there was a second ':', we have a 'like' region
            if (like)
            {
#ifdef VERBOSE
                cerr << " like: '" << like << endl;
#endif
                while (*like && !isdigit(*like))
                    like++;

                int likeNo = atoi(like);
                int old = 0;
                while (old < actRegion && flags.icoreg[old] != likeNo)
                    old++;
                if (flags.icoreg[old] != likeNo)
                {
                    sendWarning("Tried to set region '%s' like unknown #%s", name, like);
                }
                else
                {
                    // copy flags blocks
                    int i;
                    flags.icovel[actRegion] = flags.icovel[old];
                    flags.icot[actRegion] = flags.icot[old];
                    flags.icoden[actRegion] = flags.icoden[old];
                    flags.icotur[actRegion] = flags.icotur[old];
                    flags.icop[actRegion] = flags.icop[old];
                    for (i = 0; i < MAX_SCALARS; i++)
                        flags.icosca[actRegion][i] = flags.icosca[old][i];

                    // copy all values and activate the parameters
                    float x, y, z;
                    if (flags.icovel[actRegion] > 0) // Velocity
                    {
                        p_v[old]->getValue(x, y, z);
                        p_v[actRegion]->setValue(x, y, z);
                        p_v[actRegion]->setActive(1);
                        p_v[actRegion]->enable();
                        sliderCopy(p_vmag[actRegion], p_vmag[old]);
                    }

                    if (flags.icot[actRegion] > 0) // Temperature
                        sliderCopy(p_t[actRegion], p_t[old]);

                    if (flags.icoden[actRegion] > 0) // Density
                        sliderCopy(p_den[actRegion], p_den[old]);

                    if (flags.icop[actRegion] > 0) // Pressure
                        sliderCopy(p_p[actRegion], p_p[old]);

                    if (flags.icotur[actRegion] < 0) // Turbulence, k-eps
                    {
                        sliderCopy(p_k[actRegion], p_k[old]);
                        sliderCopy(p_eps[actRegion], p_eps[old]);
                    }

                    if (flags.icotur[actRegion] > 0) // Turbulence, Int-len
                    {
                        sliderCopy(p_tin[actRegion], p_tin[old]);
                        sliderCopy(p_len[actRegion], p_len[old]);
                    }

                    for (i = 0; i < MAX_SCALARS; i++)
                        if (flags.icosca[actRegion][i])
                            sliderCopy(p_scal[actRegion][i], p_scal[old][i]);
                }
            }

            if (flags.icoreg[actRegion] < 0)
            {
                killLastRegion = true;
            }
        }

        ////////////////////////////////////////////////////////////////////////////

        /// REGION SETTINGS
        /// this is inside some region: we do not get here without a 'region' before

        // velocity magnitude
        else if (strncasecmp("vmag", line, 4) == 0 && isspace(line[4]))
        {
            float min, max, val, u, v, w, scale;
            sscanf(line + 4, "%f %f %f", &min, &max, &val);
            p_vmag[actRegion]->setValue(min, max, val);

            // correct the velocity if given before, else set to (1,0,0)*magnitude
            p_v[actRegion]->getValue(u, v, w);
            scale = sqrt(u * u + v * v + w * w);
            if (scale)
            {
                scale = val / scale;
                p_v[actRegion]->setValue(u * scale, v * scale, w * scale);
            }
            else
                p_v[actRegion]->setValue(val, 0, 0);

            // we want to see these parameters
            p_vmag[actRegion]->setActive(1);
            p_vmag[actRegion]->enable();
            p_v[actRegion]->setActive(1);
            p_v[actRegion]->enable();

            // and transmit it to the simulation
            flags.icovel[actRegion] = 1;

#ifdef VERBOSE
            cerr << " vmag: " << line + 4 << endl;
#endif
        }

        // velocity direction:
        else if (strncasecmp("vdir", line, 4) == 0 && isspace(line[4]))
        {
#ifdef VERBOSE
            cerr << " vdir: " << line + 4 << endl;
#endif
            float u, v, w, scale;
            sscanf(line + 4, "%f %f %f", &u, &v, &w);

            // if magnitude was given before: scale with magnitude
            scale = p_vmag[actRegion]->getValue();
            if (scale)
            {
                scale = scale / sqrt(u * u + v * v + w * w);
                p_v[actRegion]->setValue(u * scale, v * scale, w * scale);
            }
            // if not: set magnitude
            else
            {
                scale = sqrt(u * u + v * v + w * w);
                p_v[actRegion]->setValue(u, v, w);
                p_vmag[actRegion]->setValue(0, 10 * scale, scale);
            }

            // we want to see these parameters
            p_vmag[actRegion]->setActive(1);
            p_vmag[actRegion]->enable();
            p_v[actRegion]->setActive(1);
            p_v[actRegion]->enable();

            // and transmit it to the simulation
            flags.icovel[actRegion] = 1;
        }

        // Euler angles
        else if (strncasecmp("local", line, 5) == 0 && isspace(line[5]))
        {
            float u, v, w;
            sscanf(line + 5, "%f %f %f", &u, &v, &w);
            p_euler[actRegion]->setValue(u, v, w);
#ifdef VERBOSE
            cerr << " local: " << line + 5
                 << " u=" << u << " v=" << v << " w=" << w << endl;
#endif
            // we want to see these parameters
            p_euler[actRegion]->setActive(1);
            p_euler[actRegion]->enable();

            // and transmit it to the simulation
            flags.icovel[actRegion] = 1;
        }

        // Temperature:
        else if (strncasecmp("t", line, 1) == 0 && isspace(line[1]))
        {
#ifdef VERBOSE
            cerr << " t: " << line + 2 << endl;
#endif
            // set parameter
            float min, max, val;
            sscanf(line + 2, "%f %f %f", &min, &max, &val);
            p_t[actRegion]->setValue(min, max, val);
            p_t[actRegion]->setActive(1);
            p_t[actRegion]->enable();

            // and transmit it to the simulation
            flags.icot[actRegion] = 1;
        }

        // Pressure:
        else if (strncasecmp("p", line, 1) == 0 && isspace(line[1]))
        {
#ifdef VERBOSE
            cerr << " p: " << line + 2 << endl;
#endif
            // set parameter
            float min, max, val;
            sscanf(line + 2, "%f %f %f", &min, &max, &val);
            p_p[actRegion]->setValue(min, max, val);
            p_p[actRegion]->setActive(1);
            p_p[actRegion]->enable();

            // and transmit it to the simulation
            flags.icop[actRegion] = 1;
        }

        // Turbulence: k
        else if (strncasecmp("k", line, 1) == 0 && isspace(line[1]))
        {
#ifdef VERBOSE
            cerr << " k: " << line + 2 << endl;
#endif
            // set parameter
            float min, max, val;
            sscanf(line + 2, "%f %f %f", &min, &max, &val);
            p_k[actRegion]->setValue(min, max, val);
            p_k[actRegion]->setActive(1);
            p_k[actRegion]->enable();

            // and transmit it to the simulation
            if (flags.icotur[actRegion] == -1)
                Covise::sendWarning("Warning: mixed k/eps and l/I setting");
            flags.icotur[actRegion] = 1;
        }

        // Turbulence: eps
        else if (strncasecmp("eps", line, 3) == 0 && isspace(line[3]))
        {
#ifdef VERBOSE
            cerr << " eps: " << line + 4 << endl;
#endif
            // set parameter
            float min, max, val;
            sscanf(line + 4, "%f %f %f", &min, &max, &val);
            p_eps[actRegion]->setValue(min, max, val);
            p_eps[actRegion]->setActive(1);
            p_eps[actRegion]->enable();

            // and transmit it to the simulation
            if (flags.icotur[actRegion] == -1)
                Covise::sendWarning("Warning: mixed k/eps and l/I setting");
            flags.icotur[actRegion] = 1;
        }

        // Turbulence: int
        else if (strncasecmp("tin", line, 1) == 0 && isspace(line[3]))
        {
#ifdef VERBOSE
            cerr << " tin: " << line + 4 << endl;
#endif
            // set parameter
            float min, max, val;
            sscanf(line + 4, "%f %f %f", &min, &max, &val);
            p_tin[actRegion]->setValue(min, max, val);
            p_tin[actRegion]->setActive(1);
            p_tin[actRegion]->enable();

            // and transmit it to the simulation
            if (flags.icotur[actRegion] == 1)
                Covise::sendWarning("Warning: mixed k/eps and l/I setting");
            flags.icotur[actRegion] = -1;
        }

        // Turbulence: eps
        else if (strncasecmp("tlen", line, 4) == 0 && isspace(line[4]))
        {
#ifdef VERBOSE
            cerr << " tlen: " << line + 5 << endl;
#endif
            // set parameter
            float min, max, val;
            sscanf(line + 5, "%f %f %f", &min, &max, &val);
            p_len[actRegion]->setValue(min, max, val);
            p_len[actRegion]->setActive(1);
            p_len[actRegion]->enable();

            // and transmit it to the simulation
            if (flags.icotur[actRegion] == 1)
                Covise::sendWarning("Warning: mixed k/eps and l/I setting");
            flags.icotur[actRegion] = -1;
        }

        // Scalars
        else if (strncasecmp("scal", line, 4) == 0 && isdigit(line[4]))
        {
#ifdef VERBOSE
            cerr << " scal: " << line + 5 << endl;
#endif
            // retrieve scalar number
            int scalNo = 0;
            const char *lPtr = NULL;
            if (isspace(line[5]))
            {
                scalNo = (line[4] - '0');
                lPtr = line + 5;
            }
            else if (isdigit(line[5]) && isspace(line[6]))
            {
                scalNo = 10 * (line[4] - '0') + (line[5] - '0');
                lPtr = line + 6;
            }
            else if (isdigit(line[5]) && isdigit(line[6]) && isspace(line[7]))
            {
                scalNo = 100 * (line[4] - '0') + 10 * (line[5] - '0') + (line[6] - '0');
                lPtr = line + 7;
            }

            // find first unused Scalar element number
            int scIndex = 0;
            while (scIndex < MAX_SCALARS && flags.icosca[actRegion][scIndex])
                scIndex++;

            if (scIndex == MAX_SCALARS)
            {
                sendError("Using more Scalars than allowed");
                scIndex = MAX_SCALARS - 1;
            }

            // set parameter
            float min, max, val;
            sscanf(lPtr, "%f %f %f", &min, &max, &val);
            p_scal[actRegion][scIndex]->setValue(min, max, val);
            p_scal[actRegion][scIndex]->setActive(1);
            p_scal[actRegion][scIndex]->enable();

            // make sure our base switch works, too
            //p_scalSw[actRegion]->setActive(1);
            //p_scalSw[actRegion]->enable();

            flags.icosca[actRegion][scIndex] = scalNo;
        }

        // User data
        else if (strncasecmp("user", line, 4) == 0 && isdigit(line[4]))
        {
#ifdef VERBOSE
            cerr << " user: " << line + 5 << endl;
#endif
            // retrieve scalar number
            int userNo = 0;
            const char *lPtr = NULL;
            if (isspace(line[5]))
            {
                userNo = (line[4] - '0');
                lPtr = line + 5;
            }
            else if (isdigit(line[5]) && isspace(line[6]))
            {
                userNo = 10 * (line[4] - '0') + (line[5] - '0');
                lPtr = line + 6;
            }
            else if (isdigit(line[5]) && isdigit(line[6]) && isspace(line[7]))
            {
                userNo = 100 * (line[4] - '0') + 10 * (line[5] - '0') + (line[6] - '0');
                lPtr = line + 7;
            }

            if (userNo <= MAX_UDATA && userNo > 0)
            {
                // set parameter
                float min, max, val;
                sscanf(lPtr, "%f %f %f", &min, &max, &val);
                p_user[actRegion][userNo - 1]->setValue(min, max, val);
                p_user[actRegion][userNo - 1]->setActive(1);
                p_user[actRegion][userNo - 1]->enable();

                // make sure our base switch works, too
                //p_userSw[actRegion]->setActive(1);
                //p_userSw[actRegion]->enable();

                flags.icousr[actRegion][userNo - 1] = 1;
            }
            else
                sendError("UserData illegal Index: %s", line);
        }

        // get next line
        line = getLine(filebuffer);
    }

    ///  @@@@@@ We should now update the names for the scalar parameters used

    numRegions = actRegion + 1;

    // set the choice: count from 1 and add 1 for title
    p_region->setValue(actRegion + 2, choices, 0);

    // this is the number of regions we use
    flags.iconum = actRegion + 1;

    // show my region - includes all parameters
    p_region->show();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
/// Main setup routine

int StarCD::doSetup(const coDistributedObject *setupObj,
                    const coDistributedObject *commObj,
                    int useModified)
{

    // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    // @@@@@@@@@@@ Nach dem Review einkommentieren !!!!
    // We create a new set-up, so next time create NEW interactors
    // d_useOldConfig = 0;

    int i;

    // remove old stuff
    delete[] d_user;
    d_user = NULL; // user name for simulation run
    delete[] d_host;
    d_host = NULL; // host name for simulation run
    delete[] d_compDir;
    d_compDir = NULL; // directory for simulation run
    delete[] d_case;
    d_case = NULL; // case name for simulation run
    delete[] d_script;
    d_script = NULL; // script to create geometry
    delete[] d_meshDir;
    d_meshDir = NULL; // Filename for .mdl file in local space

    // empty string if no args come in -> always have content for %s argument
    d_commArg = strcpy(new char[1], "");

    for (i = 0; i < 10; i++) // user parameters 0..9
    {
        delete[] d_usr[i];
        d_usr[i] = NULL;
    }

    // ----------------- Read StarConfig file

    // the 'line' var is needed to move over one line from setupCase to setupRegions
    const char *line;

    // clear flags block
    memset(&flags, 0, sizeof(flags));

    // open the file
    int starConfig = 0;

    // if we read a .modified file, we ignore changes from
    // setup and command objects
    bool isModified = false;

    // try to open file *.modified after script call
    if (useModified)
    {
        char buffer[1024];
        sprintf(buffer, "%s.modified", p_setup->getValue());
        starConfig = open(buffer, O_RDONLY);
        if (starConfig > 0)
        {
            isModified = true;
            sendInfo("Now using %s", buffer);
        }
        else
        {
            sendInfo("Script did not modify config file");
        }
    }
    // if no .modified or not after script call: try direct read
    if (starConfig <= 0)
    {
        starConfig = open(p_setup->getValue(), O_RDONLY);
    }

    if (starConfig <= 0)
    {
        sendError("Could not open config file");
        return -1;
    }
    else
    {
        /// Get the complete file: get length and read into buffer
        struct stat statRec;
        if (fstat(starConfig, &statRec) < 0)
        {
            sendError("Could not retrieve lenght of file %s", p_setup->getValue());
            return -1;
        }

        // give an empty file to read everything from setupString
        if (statRec.st_size)
        {
            char *buffer = new char[statRec.st_size + 1];
            char *bufPtr = buffer;
            if (statRec.st_size != read(starConfig, buffer, statRec.st_size))
            {
                sendError("Short read from file %s", p_setup->getValue());
                return -1;
            }
            buffer[statRec.st_size] = '\0';

            /// retrieve everything before the first 'region' line : USER, HOST, DIR, CASE
            if (setupCase(buffer, line))
                return -1; // on error, quit

            // the rest must be configuration lines
            setupRegions(buffer, line);

            delete[] bufPtr;
        }
        close(starConfig);
    }

    // ----------------- Read Config object

    // overwrite from setup given in Object
    if (!isModified && setupObj && setupObj->isType("DOTEXT"))
    {
        coDoText *txtObj = (coDoText *)setupObj;
        int size = txtObj->getTextLength();
        if (size)
        {
            // we need a copy, we'll insert \0s
            char *buffer = new char[size + 1];
            char *bufPtr = buffer;
            char *objTextPtr;
            txtObj->getAddress(&objTextPtr);
            memcpy(buffer, objTextPtr, size);
            buffer[size] = '\0';

            if (setupCase(buffer, line))
                return -1; // on error, quit

            // the rest must be configuration lines
            setupRegions(buffer, line);

            delete[] bufPtr;
        }
        delete[] d_setupObjName;
        const char *objName = setupObj->getName();
        d_setupObjName = strcpy(new char[strlen(objName) + 1], objName);
    }

    // ----------------- Read Commands object

    // overwrite old value: always pre-set to at least empty string
    if (!isModified && commObj && commObj->isType("DOTEXT"))
    {
        coDoText *txtObj = (coDoText *)commObj;
        int size = txtObj->getTextLength();
        if (size)
        {
            char *objTextPtr;
            txtObj->getAddress(&objTextPtr);

            delete[] d_commArg;
            d_commArg = strcpy(new char[strlen(objTextPtr) + 1], objTextPtr);
        }
        else
            d_commArg = strcpy(new char[1], "");

        delete[] d_commObjName;
        const char *objName = commObj->getName();
        d_commObjName = strcpy(new char[strlen(objName) + 1], objName);
    }

    return 0;
}
