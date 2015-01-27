/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
#ifndef _CO_FEEDBACK_PARSER_H_
#define _CO_FEEDBACK_PARSER_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coFeedbackParser
//
// This class parses INTERACTOR attributes
//
// Initial version: 2004-07-14 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include <covise/covise.h>

/**
 * Class This class parses INTERACTOR attributes
 * 
 */
class coFeedbackParser
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Types
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // Structure for each Parameter
    typedef struct
    {
        string name;
        string type;
        string value;
    } ParaDesc;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    coFeedbackParser(const string &interactorAttrib);

    /// Destructor : virtual in case we derive objects
    virtual ~coFeedbackParser();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    bool getBoolean(const string &name, bool &res);
    bool getIntScalar(const string &name, int &res);
    bool getFloatScalar(const string &name, float &res);
    bool getFloatVector(const string &name, float &x, float &y, float &z);
    bool getFloatVector(const string &name, int idx, float &val);
    bool getIntVector(const string &name, int &x, int &y, int &z);
    bool getIntVector(const string &name, int idx, int &val);
    bool getChoice(const string &name, int &val);
    bool getChoice(const string &name, int &val, list<string> &choices);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    void print(ostream &str);

    // get Feedback parameters
    const string &moduleName();
    int moduleInstance();
    string moduleNameInstance();
    const string &moduleHost();
    const string &pluginName();

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // Module name
    string module_;

    // Module Instance
    int instance_;

    // Module Host
    string host_;

    // Plugin Name
    string plugin_;

    // List of Parameters
    map<string, ParaDesc> paraMap_;

    // list of additional strings
    vector<string> stringList_;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    bool findPara(const string &name, const string &type, string &value);

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// Copy-Constructor: NOT IMPLEMENTED, checked by assert
    coFeedbackParser(const coFeedbackParser &);

    /// Assignment operator: NOT IMPLEMENTED, checked by assert
    coFeedbackParser &operator=(const coFeedbackParser &);

    /// Default constructor: NOT IMPLEMENTED, checked by assert
    coFeedbackParser();
};

#endif
