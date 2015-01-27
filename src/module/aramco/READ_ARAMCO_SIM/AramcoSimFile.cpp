/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AramcoSimFile.h"
#include <assert.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>

// own edfine in ARAMCO's header to remove global var
#define COVISE_INCLUDED
#include "ext2SV.h"
#undef COVISE_INCLUDED

/// switch verbose setting
#undef VERBOSE

#ifdef VERBOSE
void printHeader(const simHDR *hdr1);
void printDataHeader(const simDATA *hdr2);
#endif

//////////////////////////////////////////////////////////////
/////  Static Variable initializers
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
/////  Constructor
//////////////////////////////////////////////////////////////

AramcoSimFile::AramcoSimFile(const char *filename)
    : d_isValid(false)
    , // set to valid if everything worked
    d_label(NULL)
    , d_numDataSets(0)
    , d_numTimesteps(0)
    , d_file(NULL)
    , d_xyCoord(NULL)
    , d_zCoord(NULL)
    , d_activeMap(NULL)
    , d_startPos(NULL)
    , d_dataRep(NULL)
    , d_numLay(-1)
    , d_numRow(-1)
    , d_numCol(-1)
    , d_title(NULL)
{
    int i;

    // no error yet
    strcpy(d_error, "No Error specified");

    // try to open the file
    d_file = fopen(filename, "rb");

    if (!d_file)
    {
        sprintf(d_error, "Error opening %s : %s", filename, strerror(errno));
        return;
    }

    // Read File Header
    simHDR hdr1;
    fread(&hdr1, sizeof(simHDR), 1, d_file);
#ifdef VERBOSE
    printHeader(&hdr1);
#endif

    hdr1.title[119] = '\0';
    char *last = hdr1.title + strlen(hdr1.title) - 1;
    while (isspace(*last) && last != hdr1.title)
    {
        *last = '\0';
        last--;
    }
    d_title = strcpy(new char[strlen(hdr1.title) + 1], hdr1.title);

    // calculate total # of properties (add GRID SUBSEA & ACTIVE properties)
    int nATTR = hdr1.nIprop + hdr1.nTprop + 3;

    // allocate memory
    d_startPos = new long[nATTR];
    d_dataRep = new int[nATTR];
    d_label = new char *[nATTR];
    d_label[0] = strcpy(new char[8], "---");
    for (i = 1; i < nATTR; i++)
        d_label[i] = NULL;

    // we count our data fields in here
    d_numDataSets = 0;

    // read Data Headers
    int recNo;
    for (recNo = 0; recNo < nATTR; recNo++)
    {

        simDATA hdr2;
        long actPos = ftell(d_file);
        fread(&hdr2, sizeof(simDATA), 1, d_file);
#ifdef VERBOSE
        printDataHeader(&hdr2);
#endif

        // make sure that title is terminated
        hdr2.title[119] = '\0';

        // names are blank-filled -> don't keep it!
        char *last = hdr2.title + strlen(hdr2.title) - 1;
        while (isspace(*last) && last != hdr2.title)
        {
            *last = '\0';
            last--;
        }

        // calculate how many bytes to jump over
        long nbSKIP;
        switch (hdr2.type)
        {
        // XY coordinates
        case XYGRID:
        {
            //nbSKIP = sizeof(float) * hdr2.nR * hdr2.nC * 2;
            nbSKIP = 0; // we READ, so we don't skip here
            int numElem = hdr2.nR * hdr2.nC * 2;
            d_xyCoord = new float[numElem];
            fread(d_xyCoord, sizeof(float), numElem, d_file);
            for (i = 0; i < numElem; i++)
            {
                d_xyCoord[i] *= .001; // use km as unint for better values
            }
            break;
        }

        // Z coordinates
        case ELEVGRID:
        {
            //nbSKIP = sizeof(float) * hdr2.nR * hdr2.nC * hdr2.nL;
            nbSKIP = 0; // we READ, so we don't skip here
            int numElem = hdr2.nR * hdr2.nC * hdr2.nL;
            d_zCoord = new float[numElem];
            fread(d_zCoord, sizeof(float), numElem, d_file);
            d_numLay = hdr2.nL;
            d_numRow = hdr2.nR;
            d_numCol = hdr2.nC;
            for (i = 0; i < numElem; i++)
            {
                d_zCoord[i] *= .001; // use km as unint for better values
            }
            break;
        }

        // Activation map
        case ACTIVE:
        {
            nbSKIP = 0;

            // read activation into temporary
            float *active = new float[hdr2.nR * hdr2.nC * hdr2.nL];
            fread(active, sizeof(float), hdr2.nR * hdr2.nC * hdr2.nL, d_file);

            // build map global->active
            d_activeMap = new int[hdr2.nR * hdr2.nC * hdr2.nL];
            d_numActive = 0;
            int i; //overri
            for (i = 0; i < hdr2.nR * hdr2.nC * hdr2.nL; i++)
            {
                if (active[i])
                {
                    d_activeMap[i] = d_numActive;
                    d_numActive++;
                }
                else
                {
                    d_activeMap[i] = -1;
                }
            }

            delete[] active;
            break;
        }

        case ATTRGRID:
        {

            // notice our header's position for fast access
            d_startPos[d_numDataSets] = actPos;

            // and create the label ( [0] = "---", so add 1 )
            d_label[d_numDataSets + 1]
                = strcpy(new char[strlen(hdr2.title) + 8], hdr2.title);

            d_dataRep[d_numDataSets] = 0;

            // field attrib setting
            if (hdr2.dataref == CELL)
            {
                d_dataRep[d_numDataSets] |= 1;
            }

            if (hdr2.nTS > 1)
            {
                d_dataRep[d_numDataSets] |= 2;
                strcat(d_label[d_numDataSets + 1], " (t)");
            }

            // no timesteps is one timestep
            if (hdr2.nTS == 0)
            {
                hdr2.nTS = 1;
            }
            else
            {
                if (d_numTimesteps > 0)
                {
                    if (hdr2.nTS != d_numTimesteps)
                    {
                        strcpy(d_error, "Multiple time stepping not implemented");
                        return;
                    }
                }
                else
                {
                    d_numTimesteps = hdr2.nTS;
                }
            }

            nbSKIP = sizeof(float) * hdr2.nR * hdr2.nC * hdr2.nL
                     * hdr2.nTS;
            d_numDataSets++;
            break;
        }
        default:
        {
            sprintf(d_error, "Found unknown type of field '%d' at offset %d",
                    hdr2.type, i);
            return;
            break;
        }
        }

        if (feof(d_file))
        {
            printf(" reached end of file \n");
            break;
        }

        /*------- jump over data block to next attribute header */
        if (nbSKIP)
        {
            fseek(d_file, nbSKIP, SEEK_CUR);
        }
    }

    // if no errors found and all required data there - valid
    if (d_numLay > 0 && d_numRow > 0 && d_numCol > 0
        && d_xyCoord && d_zCoord && d_activeMap)
        d_isValid = true;
    else
    {
        strcpy(d_error, "Missing essential file parts:");
        if (d_numLay <= 0)
            strcat(d_error, " Layers<=0");
        if (d_numRow <= 0)
            strcat(d_error, " Rows<=0");
        if (d_numCol <= 0)
            strcat(d_error, " Columns<=0");
        if (!d_xyCoord)
            strcat(d_error, " XYGRID");
        if (!d_zCoord)
            strcat(d_error, " ELEVGRID");
        if (!d_activeMap)
            strcat(d_error, " ACTIVE");
    }
}

//////////////////////////////////////////////////////////////
/////  Destructors
//////////////////////////////////////////////////////////////

AramcoSimFile::~AramcoSimFile()
{
    if (d_file)
    {
        fclose(d_file);
    }

    // delete labels - we have a marker at the end ins
    int i;
    if (d_label)
    {
        for (i = 0; i <= d_numDataSets; i++)
        {
            delete[] d_label[i];
        }
    }
    delete[] d_label;
    delete[] d_startPos;
    delete[] d_dataRep;
    delete[] d_title;
    delete[] d_xyCoord;
    delete[] d_zCoord;
    delete[] d_activeMap;
}

//////////////////////////////////////////////////////////////
/////  Operations
//////////////////////////////////////////////////////////////

// get XY coordinate field
const float *AramcoSimFile::getXYcoord()
{
    return d_xyCoord;
}

// get Z coordinate field
const float *AramcoSimFile::getZcoord()
{
    return d_zCoord;
}

// get activation field
const int *AramcoSimFile::getActiveMap()
{
    return d_activeMap;
}

// read a data field, #0.., return number of read bytes
int AramcoSimFile::readData(float *buffer, int setNo, int stepNo)
{
    size_t floatsRead;
    simDATA hdr;

    if (setNo < 0 || setNo >= d_numDataSets)
        return -1;

    if (stepNo < 0 || stepNo > d_numTimesteps)
        return -1;

    // jump to file position
    if (positionFilePtr(d_startPos[setNo]) != 0)
        return -1;

    // read the header
    if (readHeader(hdr) != 0)
        return -1;

    if (stepNo)
    {
        long nbSKIP = sizeof(float) * hdr.nR * hdr.nC * hdr.nL * stepNo;
        fseek(d_file, nbSKIP, SEEK_CUR);
    }

    floatsRead = fread(buffer, sizeof(float), hdr.nR * hdr.nC * hdr.nL, d_file);

    return floatsRead * sizeof(float);
}

//////////////////////////////////////////////////////////////
/////  Attribute request/set functions
//////////////////////////////////////////////////////////////

// get the grid's sizes
void AramcoSimFile::getSize(int &numLay, int &numRow, int &numCol)
{
    numLay = d_numLay;
    numRow = d_numRow;
    numCol = d_numCol;
}

// get the number of data fields
int AramcoSimFile::numActive()
{
    return d_numActive;
}

// get the number of active Cells
int AramcoSimFile::numDataSets()
{
    return d_numDataSets;
}

// get number of timesteps of given data field, -1 on error
int AramcoSimFile::numTimeSteps()
{
    return d_numTimesteps;
}

const char *AramcoSimFile::getErrorMessage()
{
    return d_error;
}

bool AramcoSimFile::isBad()
{
    return !d_isValid;
}

const char *AramcoSimFile::getTitle()
{
    return d_title;
}

// get the labels for the choices
const char *const *AramcoSimFile::getLabels()
{
    return d_label;
}

// request whether data set is cell-based
bool AramcoSimFile::isCellBased(int fieldNo)
{
    return (d_dataRep[fieldNo] & 1) != 0;
}

// request whether data set is time-dependent
bool AramcoSimFile::isTransient(int fieldNo)
{
    return (d_dataRep[fieldNo] & 2) != 0;
}

//////////////////////////////////////////////////////////////
/////  Internally used functions
//////////////////////////////////////////////////////////////

// set a position and read the header
int AramcoSimFile::positionFilePtr(long position)
{
    // position file pointer
    if (fseek(d_file, position, SEEK_SET) != 0)
    {
        sprintf(d_error, "Error seeking file pos %ld : %s",
                position, strerror(errno));
        return -1;
    }

    return 0;
}

// read the header
int AramcoSimFile::readHeader(simDATA &hdr)
{
    int numRead = fread(&hdr, 1, sizeof(simDATA), d_file);
    if (numRead != sizeof(simDATA))
    {
        sprintf(d_error, "Error reading Header at pos %ld : %s",
                ftell(d_file), strerror(errno));
        return -1;
    }

    return 0;
}

//////////////////////////////////////////////////////////////
/// Prevent auto-generated functions by assert or implement
//////////////////////////////////////////////////////////////

/// Copy-Constructor: NOT IMPLEMENTED
AramcoSimFile::AramcoSimFile(const AramcoSimFile &)
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
AramcoSimFile &AramcoSimFile::operator=(const AramcoSimFile &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
AramcoSimFile::AramcoSimFile()
{
    assert(0);
}

#ifdef VERBOSE
static void printHeader(const simHDR *hdr1)
{
    // Print summary parameters to screen
    /*----------- Display the .SIM header information                     */

    printf("\n\n");
    printf("     SIM File Summary -\n\n");
    printf("        %s\n\n", hdr1->title);
    printf("        Simulator Name      :  %s\n", hdr1->sim_name);
    printf("        Version             :  %5d\n", hdr1->version);
    printf("        Number of Rows      :  %5d\n", hdr1->nR);
    printf("        Number of Cols      :  %5d\n", hdr1->nC);
    printf("        Number of Layers    :  %5d\n", hdr1->nL);
    printf("        Number of Time Steps:  %5d\n", hdr1->nTS);
    printf("        Number of IP        :  %5d\n", hdr1->nIprop);
    printf("        Number of TDP       :  %5d\n", hdr1->nTprop);
    printf("        X Offset            :  %f  \n", hdr1->xOFF);
    printf("        Y Offset            :  %f  \n", hdr1->yOFF);
    printf("        Angle of Rotation   :  %f  \n", hdr1->aROT);
    /*----------- print the calendar year for each time step */
    int indx;
    for (indx = 0; indx < hdr1->nTS; indx++)
        printf("%8sTime Step %-2d %-8.3f\n", " ", indx, hdr1->TS_dates[indx]);

    printf("\n\n");
}

static void printDataHeader(const simDATA *hdr2)
{
    printf(" ---------------------------------------\n");
    printf(" %s\n", hdr2->title);
    printf(" Type         :  %5d\n", hdr2->type);
    printf(" ROWS         :  %5d\n", hdr2->nR);
    printf(" COLUMNS      :  %5d\n", hdr2->nC);
    printf(" LAYERS       :  %5d\n", hdr2->nL);
    printf(" # Time Steps :  %5d\n", hdr2->nTS);
    printf(" Min Value    :  %.4f\n", hdr2->min);
    printf(" Max Value    :  %.4f\n", hdr2->max);
    printf(" dataref      :  %4c\n\n", (char)hdr2->dataref);
    fflush(stdout);
}
#endif
