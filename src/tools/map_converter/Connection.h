/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:  Class to store connection data in map-files           ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:  24.01.2001                                                   ++
// ++**********************************************************************/
#ifndef CONNECTION_H
#define CONNECTION_H

#include "ModuleSkel.h"

class Connection
{
public:
    enum Validity
    {
        Invalid = 0,
        ValidSourceModule = 1,
        ValidDestinationModule = 2,
        ValidSourcePort = 4,
        ValidDestinationPort = 8,
        Valid = ValidSourceModule | ValidDestinationModule | ValidSourcePort | ValidDestinationPort,
    };

    Connection();
    Connection(const ModuleSkeleton &frmMod, const ModuleSkeleton &toMod,
               const std::string &assertOutPort, const std::string &assertInPort);
    Connection(const Connection &rc);

    const Connection &operator=(const Connection &rc);

    // returns Valid if the connection is valid, something else otherwise
    int valid() const
    {
        return valid_;
    };

    // get, set host information
    std::string getFrmHost()
    {
        return frmHost_;
    };
    std::string getToHost()
    {
        return toHost_;
    };

    // return connection data
    std::string frmModName() const;
    std::string toModName() const;
    std::string frmModIdx() const;
    std::string toModIdx() const;
    std::string frmPort() const;
    std::string toPort() const;

    void setFrmHost(const std::string &host)
    {
        frmHost_ = host;
    };
    void setToHost(const std::string &host)
    {
        toHost_ = host;
    };

    // write connection information to s in a format suitible
    // for net-files
    friend std::ostream &operator<<(std::ostream &s, const Connection &con);

    // write a warning to stream s (may do nothing if the
    // connection is valid
    void warning(std::ostream &s) const;

    virtual ~Connection();

private:
    void build();

    int valid_;

    std::string frmModName_;
    std::string toModName_;
    std::string frmPrtName_;
    std::string toPrtName_;
    std::string frmPrtObjName_;
    std::string toHost_;
    std::string frmHost_;
    int frmNetIdx_;
    int toNetIdx_;

    ModuleSkeleton frmMod_;
    ModuleSkeleton toMod_;
};
#endif
