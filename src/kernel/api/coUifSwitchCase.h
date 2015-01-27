/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_UIF_SWITCH_CASE_H_
#define _CO_UIF_SWITCH_CASE_H_

// 15.09.99

namespace covise
{

class coUifElem; // use only prt in header, include in cpp
class coUifSwitch; // we don't know anything about it, we can only hold a pointer

/**
 * Class to handle the 'case' level of the switch hierarchie
 *
 */
class APIEXPORT coUifSwitchCase
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coUifSwitchCase(const coUifSwitchCase &);

    /// Assignment operator: NOT  IMPLEMENTED
    coUifSwitchCase &operator=(const coUifSwitchCase &);

    /// Default constructor: NOT  IMPLEMENTED
    coUifSwitchCase();

    /// case name == label on the Choice
    char *d_name;

    // pointer to my master ... whatever tyhis might be, we don't use it
    coUifSwitch *d_master;

    /// this is the REAL maximum number of ports
    enum
    {
        MAX_PORTS = 4096
    };

    // list of all Elements and number of Elements in list
    coUifElem *d_elemList[MAX_PORTS];
    int d_numElem;

public:
    /// Destructor : virtual in case we derive objects
    virtual ~coUifSwitchCase();

    /// Constructor: give this case a name
    coUifSwitchCase(const char *name, coUifSwitch *master);

    /// add one element to out group
    void add(coUifElem *elem);

    /// returns my name
    const char *getName() const;

    /// get my superior switch
    coUifSwitch *getMaster() const;

    /// Hide everything below
    virtual void hide();

    /// Show everything below
    virtual void show();
};
}
#endif
