/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ARAMCO_SIM_FILE_H_
#define __ARAMCO_SIM_FILE_H_

#include <stdio.h>

#include "ext2SV.h"

// Initial Creation by we_te 19.10.01

/**
 * Class
 *
 */
class AramcoSimFile
{
public:
    //////////////////////////////////////////////////////////////
    ///// Constructors / Destructor
    //////////////////////////////////////////////////////////////

    /// Constructor :
    AramcoSimFile(const char *filename);

    /// Destructor : virtual in case we derive objects
    virtual ~AramcoSimFile();

    //////////////////////////////////////////////////////////////
    ///// Operations
    //////////////////////////////////////////////////////////////

    // get XY coordinate field
    const float *getXYcoord();

    // get Z coordinate field
    const float *getZcoord();

    // get activation field
    const int *getActiveMap();

    // read a data field, #0.., return number of read bytes
    int readData(float *buffer, int setNo, int stepNo);

    //////////////////////////////////////////////////////////////
    ///// Attribute request/set functions
    //////////////////////////////////////////////////////////////

    // get the grid's sizes
    void getSize(int &numLay, int &numRow, int &numCol);

    // get the number of data sets
    int numDataSets();

    // get the number of active Cells
    int numActive();

    // get number of timesteps of given data field, -1 on error
    int numTimeSteps();

    // request whether data set is cell-based
    bool isCellBased(int fieldNo);

    // request whether data set is time-dependent
    bool isTransient(int fieldNo);

    // get the error message
    const char *getErrorMessage();

    // check validity
    bool isBad();

    // get the title of the simulation file
    const char *getTitle();

    // get the labels for the choices
    const char *const *getLabels();

protected:
    //////////////////////////////////////////////////////////////
    /////  Attributes
    //////////////////////////////////////////////////////////////

    // whether this is a valid Aramco Sim file
    bool d_isValid;

    // labels for the data fields
    char **d_label;

    // number of data fields
    int d_numDataSets;

    // number of time steps
    int d_numTimesteps;

    // index of the ACTIVE field within the data fields

    // the open data file
    FILE *d_file;

    // the XY coordinates of the file
    float *d_xyCoord;

    // the Z coordinates of the file
    float *d_zCoord;

    // the activation info # of active cells, Map global->active
    int *d_activeMap;
    int d_numActive;

    // Buffer for an error message to the caller class
    char d_error[1024];

    // starting offset for data fields
    long *d_startPos;

    // Data Representation: one int per data field
    //      Bit 0 (val=1):    CELL-based=1      VERTEX-based=0
    //      Bit 1 (val=2):    TIME-dependent=1  CONSTANT    =0
    int *d_dataRep;

    // grid sizes i,j,k = Layers / Rows / Columns = z,y,x ????
    int d_numLay, d_numRow, d_numCol;

    // Case Title
    char *d_title;

private:
    //////////////////////////////////////////////////////////////
    /////  Internally used functions
    //////////////////////////////////////////////////////////////

    // jump to a specific position in file
    // on error set error message and return !=0
    int positionFilePtr(long position);

    // read Data Header,
    // on error set error message and return !=0
    int readHeader(simDATA &hdr);

    //////////////////////////////////////////////////////////////
    /////  prevent auto-generated bit copy routines by default
    //////////////////////////////////////////////////////////////

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    AramcoSimFile(const AramcoSimFile &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    AramcoSimFile &operator=(const AramcoSimFile &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    AramcoSimFile();
};
#endif
