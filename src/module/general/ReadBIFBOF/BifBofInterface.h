/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
 *	        Libary to read files in COVISE format                     *
 *                                			                  *
 *                           (C); 2001 			                  *
 *                   VirCinity IT-Consulting GmbH                          *
 *                         Nobelstrasse 15				  *
 *                       D-70569 Stuttgart				  *
 *                            Germany					  *
 * Author: S. Kufer							  *
 * Date: 28. Juli 2001							  *
 **************************************************************************/

#ifndef _COVISE_BIFBOFLIB
#define _COVISE_BIFBOFLIB
#include <iostream> //cout
#include <string>
#include <map>
/** COVISE read in Bifbof data **/

#define BIFBOF_EOF 9
#define BIFBOF_UNKNOWN_DATATYPE 15
using namespace std;

class BifBof
{
public:
    typedef union
    {
        int i;
        float f;
        char c[4];
    } Word;

    BifBof(const char *catalogFile);
    ~BifBof();
    int openBifFile(const char *inputfile);
    int closeBifFile();

    int readFileHeader(std::string &programName, std::string &date,
                       std::string &time, std::string &description);

    int readElementHeader(Word *headBuffer, int seqElemNum, int &id,
                          int &subheaderFlag, int &numRecords);
    int readRegularRecord(Word *elementBuffer, int &readingComplete);
    void initErrors();
    std::string returnErrorCode(int errorID);

    //int checkDSIOError();
    //int checkHDMSCCError();
    int getLastError()
    {
        return p_ierr;
    };

private:
    int caeDataElemUnit;
    int fileInputUnit;
    int p_ierr;
    int recordsRead;
    map<int, string> Error;
};

#endif
