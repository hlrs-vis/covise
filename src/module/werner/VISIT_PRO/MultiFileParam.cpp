/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS MultiFileParam
//
// This class @@@
//
// Initial version: 2002-07-23 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "MultiFileParam.h"
#include <api/coModule.h>
#include <assert.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MultiFileParam::MultiFileParam(const char *name,
                               const char *desc,
                               const char *file,
                               const char *filter,
                               coModule *mod)
{
    char *buffer = new char[strlen(name) + 10];

    // Select multiple input files
    sprintf(buffer, "%s.browser", name);
    p_browser = mod->addFileBrowserParam(buffer, desc);
    p_browser->setValue(file, filter);

    sprintf(buffer, "%s.collect", name);
    p_collect = mod->addStringParam(buffer, desc);
    p_collect->setValue("");

    delete[] buffer;

    //
    d_numFiles = -1;
    d_allFiles = NULL;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

MultiFileParam::~MultiFileParam()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// to be forwarded from module
void MultiFileParam::param(const char *paramName)
{
    bool changed = false;

    if (0 == strcmp(paramName, p_browser->getName())
        && !Covise::in_map_loading())
    {
        const char *newFile = p_browser->getValue();
        const char *oldFiles = p_collect->getValue();
        char *buffer = new char[strlen(newFile) + strlen(oldFiles) + 2];
        strcpy(buffer, oldFiles);
        strcat(buffer, " ");
        strcat(buffer, newFile);
        p_collect->setValue(buffer);
        delete[] buffer;
        changed = true;
    }
    else if (0 == strcmp(paramName, p_collect->getName())
             && !Covise::in_map_loading())
    {
        changed = true;
    }

    // parse filenames and create array
    if (changed)
    {
        d_numFiles = -1;
        delete[] d_allFiles;
    }
}

// number of files we have
int MultiFileParam::numFiles()
{
    // not yet parsed - do it!
    if (d_numFiles < 0)
        parseNames();

    return d_numFiles;
}

// access i-th file
const char *MultiFileParam::fileName(int i)
{
    // not yet parsed - do it!
    if (d_numFiles < 0)
        parseNames();

    if (i < d_numFiles && i >= 0)
        return d_names[i];

    return NULL;
}

void MultiFileParam::show()
{
    p_browser->show();
    p_collect->show();
}

void MultiFileParam::hide()
{
    p_browser->hide();
    p_collect->hide();
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void MultiFileParam::parseNames()
{
    delete[] d_allFiles;

    // get string parameter content into own buffer for strtok
    const char *paraVal = p_collect->getValue();
    char *d_allFiles = strcpy(new char[1 + strlen(paraVal)], paraVal);
    char *filename = d_allFiles;
    while (*filename && isspace(*filename))
        filename++;

    filename = strtok(filename, " "); // start strtok sequence
    d_numFiles = 0;
    while (filename && d_numFiles < MAX_NAMES)
    {
        d_names[d_numFiles] = filename;
        d_numFiles++;
        filename = strtok(NULL, " "); // skip to next file name
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Copy-Constructor: NOT IMPLEMENTED
MultiFileParam::MultiFileParam(const MultiFileParam &)
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
MultiFileParam &MultiFileParam::operator=(const MultiFileParam &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
MultiFileParam::MultiFileParam()
{
    assert(0);
}
