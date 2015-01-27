/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*-*-Mode: C;-*-
 * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * IMPLEMENTATION   bifbof_lib
 *
 * Description: Library for databus communication
 *              - open close BIF file
 *              - reading header and records
 *
 * Initial version: 05.2008
 *
 * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * (C) 2008 by Visenso
 * +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *
 * Changes:
 */

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hdmproto.h>

#include "BifBofInterface.h"

#ifdef _WIN32
#include <windows.h>
#endif

/* mapping of procedure names */

#define HDMOPI hdmopi
#define HDMCLS hdmcls
#define HDMSCC hdmscc
#define DSIO dsio
#define DSRPRE dsrpre
#define DSRHED dsrhed
#define DSRREC dsrrec
#define DSRLSE dsrlse
//#define DSRIRR	 dsrirr

#define TRIANGULAR 31
#define NODALPOINTS 1
using namespace std;

BifBof::BifBof(const char *catalogFile)
{
    fileInputUnit = 0;
/*
 * Initialize databus communication
 */
#ifndef _WIN32
    setenv("HDMcat", catalogFile, 0);
#else
    SetEnvironmentVariable("HDMcat", catalogFile);
#endif
    DSIO(&caeDataElemUnit, catalogFile, &p_ierr, strlen(catalogFile));
    recordsRead = 0;
    initErrors();
}

BifBof::~BifBof()
{
    if (fileInputUnit != 0)
    {
        closeBifFile();
    }
}

int BifBof::openBifFile(const char *inputfile)
{
    HDMOPI(inputfile, &fileInputUnit, strlen(inputfile));
    //printf("%s %s\n","HDMOPI Open file",inputfile);
    int p_ierr;
    /* get error code */
    HDMSCC(&p_ierr);
    return p_ierr;
}

int BifBof::closeBifFile()
{
    /* Release data bus file from dsio administration */
    DSRLSE(&fileInputUnit, &p_ierr);
    /* close the file */
    HDMCLS(&fileInputUnit);
    return p_ierr;
}

int BifBof::readElementHeader(Word *headBuffer, int seqElemNum, int &id,
                              int &subheaderFlag, int &numRecords)
{
    int headerLength, totalHeaderLength, numHeaderSubstucts, recordLength;
    int inBufferStart = 1;
    int inBufferEnd = 30;
    DSRHED(&fileInputUnit, &seqElemNum, headBuffer, &inBufferStart,
           &inBufferEnd, &id, &headerLength, &totalHeaderLength,
           &subheaderFlag, &numHeaderSubstucts, &numRecords,
           &recordLength, &p_ierr);
    recordsRead = 0;

    //printf("%s %d\n","DSRHED Read header data of a data element or record structure", seqElemNum);

    return p_ierr;
}

int BifBof::readRegularRecord(Word *elementBuffer, int &readingComplete)
{
    int inBufferLength = 0;
    int numRecords = 1; //Records to be read
    int inBufferStart = 1;
    int inBufferEnd = 100;

    DSRREC(&fileInputUnit, elementBuffer, &inBufferStart, &inBufferEnd,
           &numRecords, &inBufferLength, &readingComplete, &p_ierr);
    if (getLastError() == 0)
    {
        recordsRead++;
    }

    //printf("recordsRead%d ,fileInputUnit%d, inBufferStart%d, inBufferEnd%d,numRecords%d, inBufferLength%d, readingComplete %d, p_ierr%d\n", recordsRead ,fileInputUnit, inBufferStart, inBufferEnd,numRecords, inBufferLength, readingComplete, p_ierr );
    return p_ierr;
}

int BifBof::readFileHeader(std::string &programName, std::string &date,
                           std::string &time, std::string &description)
{
    char progName[9], fileDate[9], fileTime[9], fileDescription[81];
    DSRPRE(&fileInputUnit, progName, fileDate, fileTime, fileDescription, &p_ierr,
           sizeof(progName), sizeof(fileDate), sizeof(fileTime),
           sizeof(fileDescription));
    progName[8] = '\0';
    fileDate[8] = '\0';
    fileTime[8] = '\0';
    fileDescription[80] = '\0';
    programName = progName;
    date = fileDate;
    time = fileTime;
    description = fileDescription;
    return p_ierr;
}

void BifBof::initErrors()
{
    Error[0] = "no Error";
    Error[1] = "More data available than requested";
    Error[2] = "More logical records requested than available";
    Error[3] = "not specified error";
    Error[4] = "not specified error";
    Error[5] = "not specified error";
    Error[6] = "not specified error";
    Error[7] = "not specified error";
    Error[8] = "not specified error";
    Error[9] = "not specified error";
    Error[10] = "not specified error";
    Error[11] = "Block no. < 1";
    Error[12] = "Block no. < 1";
    Error[13] = "Block no. < 4 and mode > 1";
    Error[14] = "Block no. out of sequence";
    Error[15] = "Invalid input array length";
    Error[16] = "Invalid logical record count";
    Error[17] = "Processing logical record before header";
    Error[18] = "Current block still incomplete";
    Error[19] = "not specified error";
    Error[20] = "not specified error";
    Error[21] = "I/O unit JUNIT not opened";
    Error[22] = "Empty file";
    Error[23] = "End of file reached";
    Error[24] = "Writing to a file that was opened for reading";
    Error[25] = "Reading a file that was opened for writing";
    Error[26] = "Error while writing to disk";
    Error[27] = "Error while reading from disk";
    Error[28] = "File contains invalid internal record structure";
    Error[29] = "Error while opening file";
    Error[30] = "Error while closing file";
    Error[31] = "Invalid block length";
    Error[32] = "Invalid header length";
    Error[33] = "Inconsistent header data";
    Error[34] = "More data requested than available";
    Error[35] = "No more logical records";
    Error[36] = "Writing beyond end of block";
    Error[37] = "Insufficient storage for table of contents";
    Error[38] = "not specified error";
    Error[39] = "not specified error";
    Error[40] = "not specified error";
    Error[41] = "Data element catalog not specified";
    Error[42] = "Data element catalog open error";
    Error[43] = "Data element catalog read error";
    Error[44] = "Not a valid data element catalog";
    Error[45] = "Insufficient storage for data element catalog";
    Error[46] = "Data element not in catalog";
    Error[47] = "Data element and catalog entry differ";
    Error[48] = "Data element cannot get byte swapped";
}

string BifBof::returnErrorCode(int errorID)
{
    return Error[errorID];
}

//int BifBof::checkDSIOError();
//int BifBof::checkHDMSCCError();
