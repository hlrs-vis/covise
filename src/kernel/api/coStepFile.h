/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _STEP_FILE_H_
#define _STEP_FILE_H_

#include <covise/covise.h>
#include <util/coTypes.h>

#define MAX_DELTA 500

namespace covise
{

class APIEXPORT coStepFile
{

private:
    // skip:  number of files to skip
    //        Example: skip = 1 -> image1.png, image3.png, image5.png ...
    int skip;
    int delta; // maximal number of files to process
    int len_nb; // length of the start path
    int finished; // detects if we are finished
    int base_number; // starting index
    int file_index; // current index (is incremented)
    char *prefix;
    char *suffix;
    char *base;
    bool singleFile; // true if there is only one file

public:
    // Member functions

    ///////////////////////////////////////////////////////
    // constructor
    // creates a new coStepFile
    // \param filepath the path to the first fileName of
    //        a sequence of files with similar names.
    // Example: coStepFile myStepFile("image1.png");
    coStepFile(const char *filepath);
    ~coStepFile();

    ///////////////////////////////////////////////////////
    // gets path to the next file in the sequence
    // \param resultpath function writes next filename in
    //        sequence to this parameter.
    void get_nextpath(char **resultpath);

    void set_delta(int);
    void getDelta(int *);
    void set_skip_value(int);
    void get_skip_value(int *);
};
}
#endif
