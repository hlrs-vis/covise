/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __MULTI_FILE_PARAM_H_
#define __MULTI_FILE_PARAM_H_
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

class coModule;
class coFileBrowserParam;
class coStringParam;

/**
 * Managing a multi-File selection
 *
 */
class MultiFileParam
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    MultiFileParam(const char *name, const char *desc,
                   const char *file, const char *filter,
                   coModule *mod);

    /// Destructor : virtual in case we derive objects
    virtual ~MultiFileParam();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // to be forwarded from module
    void param(const char *paramName);

    // show and hide both parameters
    void show();
    void hide();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // number of files we have
    int numFiles();

    // access i-th filename
    const char *fileName(int i);

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // File browser for selection
    coFileBrowserParam *p_browser;

    // String for collecting
    coStringParam *p_collect;

    enum
    {
        MAX_NAMES = 256
    };

    // currently active names NULL if not yet parsed
    const char *d_names[MAX_NAMES];

    // number of currectly active names, <0 means not yet parsed
    int d_numFiles;

    // one strig to hold the currecnt information
    char *d_allFiles;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // parse content of strin param and set d_names and
    void parseNames();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    MultiFileParam(const MultiFileParam &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    MultiFileParam &operator=(const MultiFileParam &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    MultiFileParam();
};
#endif
