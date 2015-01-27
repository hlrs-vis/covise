/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ARGS_PARSER_H_
#define __ARGS_PARSER_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS ArgsParser
//
// Initial version: 2003-01-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

/**
 * Class to simplify Arguments parsing
 *
 */

namespace covise
{

class ArgsParser
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    ArgsParser(int argc, const char *const *argv);

    /// Destructor : virtual in case we derive objects
    virtual ~ArgsParser();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** get an option, with --longopt=<value> or -shortOpt <value>
       *  return defaulVal if not given
       */
    const char *getOpt(const char *shortOpt, const char *longOpt,
                       const char *defaultVal);

    /// return whether switch is set: true if set
    bool getSwitch(const char *shortOpt, const char *longOpt);

    /// Access non-option parameters
    // @@@ assume all short-opts have 1 parameter so far
    const char *operator[](int idx);

    /// number of non-optoion arguments
    int numArgs();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // argc from argument list
    int d_argc;

    // argv from argument list
    char **d_argv;

    // index of 1st non-Option string
    int d_firstArg;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    ArgsParser(const ArgsParser &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    ArgsParser &operator=(const ArgsParser &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    ArgsParser();
};
}
#endif
