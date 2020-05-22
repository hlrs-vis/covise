/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RWCOVISEGROUP_H
#define _RWCOVISEGROUP_H
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

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <util/ChoiceList.h>
#include <reader/CoviseIO.h>

#define MAXLINE 128

class GroupFile
{
private:
    FILE *fd;

public:
    GroupFile(const char *groupfile, int open_to_read);
    virtual ~GroupFile();
    ChoiceList *get_choice(ChoiceList **files);
    int put_choices(int num, char **desc, char **files);
    int isValid();
};

class RWCoviseGroup : public CoviseIO, public coModule
{

private:
    enum
    {
        NUMPORTS = 16
    };
    enum
    {
        READ = 1,
        WRITE = 0
    } mode;

    coFileBrowserParam *p_groupFile;
    coChoiceParam *p_file[NUMPORTS];
    coStringParam *p_desc[NUMPORTS];
    coInputPort *p_data_in[NUMPORTS];
    coOutputPort *p_data_out[NUMPORTS];

    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paramName, bool inMapLoading);

    GroupFile *groupfile;
    char GroupFileName[400];
    char GroupFilePath[400];
    char GroupFileString[400];

    int doupdate; // choice parameter update in compute
    int in_exec; // true during compute function
    bool mapLoading; // true during loading of map
    int num_in; // number of connected inports

    ChoiceList *choice=nullptr, *filenames=nullptr;

    // filenames received at map loading : do not immediately read it.
    char *mapLoad;

    int handleChangedGroupFile(const char *newpath,
                               int inMapLoading, int ignoreErr);

    void getMode();
    void checkMapLoading(bool inMapLoading);
    void updateChoices();
    void updateDescription(int oldmode, int old_num_in);
    const char *getSpecies(const coDistributedObject *obj);
    int getObjectNames();
    void parse_name(const char *filename);
    void gen_filename(char *desc, char *file);

public:
    virtual ~RWCoviseGroup();
    RWCoviseGroup(int argc, char *argv[]);
};
#endif
