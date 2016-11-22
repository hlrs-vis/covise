/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_VOIS_H
#define _READ_VOIS_H
/**************************************************************************\ 
 **                                                   	   (C)2016 UKoeln  **
 **                                                                        **
 ** Description: Simple Reader for Volumes of Interest (VOIs)              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Wickeroth                                                   **
 **                                                                        **
 ** History:                                                               **
 ** June 2016        v1                                                    **
 **                                                                        **
 **                                                                        **
\**************************************************************************/

#include <vector>
#include <api/coModule.h>
#include "VoisGlobal.h"
using namespace covise;

static const int MAXVOIS = 20;

class ReadVois : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void quit();

    bool openFile();
    bool readFile();
    void printVois();
    bool triangulate();
    void sendDataToCovise();

    //  member data
    const char *m_filename; // obj file name
    FILE *m_file;

    std::vector<triangle_t> triangles[MAXVOIS];
    std::vector<voi_t> voiVector;

    coOutputPort *m_polygonPort;
    coFileBrowserParam *m_voiFileParam;
    coBooleanParam *m_voiActive[MAXVOIS];

public:
    ReadVois(int argc, char *argv[]);
    virtual ~ReadVois();
};
#endif
