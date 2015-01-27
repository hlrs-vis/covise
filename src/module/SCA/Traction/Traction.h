/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE Traction
//
//  Traction results
//
//  Initial version:   13.06.02 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _TRACTION_SCA_H_
#define _TRACTION_SCA_H_

#include <api/coModule.h>
using namespace covise;
#include <string>
#include <iostream>

class Traction : public coModule
{
public:
    Traction(int argc, char *argv[]);
    ~Traction();

protected:
    virtual int compute(const char *port);

private:
    coInputPort *p_ZugParam_; // zug parameter(s)
    bool DifferentZugParams();
    float Displacement_;
    int NumOutPoints_;
    bool readDisplacement_;
    bool readNumOutPoints_;
    bool simulation_; // the module has successfully performed a simulation
    coInputPort *p_mssg_; // message if the basic cell is available
    coOutputPort *p_file_; // coDoText with the ANSYS rst file

    std::string simDir_;

    int maxLen_;

    int LaunchANSYS();
    void outputDummies();

    int readIntSlider(istringstream &strText, int *addr);
    int readChoice(istringstream &strText, int *addr);
    int readFloatSlider(istringstream &strText, float *addr);

    void setBooleanFalse();
    int checkReadFlags();
    // bool gotDummies();
};
#endif
