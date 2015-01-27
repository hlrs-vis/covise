/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                            ++
// ++             Implementation of class Connection                      ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 24.01.2001                                                    ++
// ++**********************************************************************/

#include <cstdio>
#include <cassert>
#include "Connection.h"

//
// Constructor
//
Connection::Connection()
    : valid_(Invalid)
    , frmModName_("")
    , toModName_("")
    , frmPrtName_("")
    , toPrtName_("")
    , frmPrtObjName_("")
    , frmNetIdx_(1)
    , toNetIdx_(1)

{
}

Connection::Connection(const ModuleSkeleton &frmMod, const ModuleSkeleton &toMod,
                       const std::string &assertInPort,
                       const std::string &assertOutPort)
    : valid_(Invalid)
    , frmModName_(frmMod.getName())
    , toModName_(toMod.getName())
    , frmPrtName_(assertInPort)
    , toPrtName_(assertOutPort)
    , frmPrtObjName_("")
    , toHost_(toMod.getHost())
    , frmHost_(frmMod.getHost())
    , frmNetIdx_(frmMod.getNetIndex())
    , toNetIdx_(toMod.getNetIndex())
    , frmMod_(frmMod)
    , toMod_(toMod)
{
    build();
    if (valid_ != Valid)
        warning(std::cerr);
}

Connection::Connection(const Connection &rc)
    : valid_(rc.valid_)
    , frmModName_(rc.frmModName_)
    , toModName_(rc.toModName_)
    , frmPrtName_(rc.frmPrtName_)
    , toPrtName_(rc.toPrtName_)
    , frmPrtObjName_(rc.frmPrtObjName_)
    , toHost_(rc.toHost_)
    , frmHost_(rc.frmHost_)
    , frmNetIdx_(rc.frmNetIdx_)
    , toNetIdx_(rc.toNetIdx_)
    , frmMod_(rc.frmMod_)
    , toMod_(rc.toMod_)
{
}

const Connection &
    Connection::
    operator=(const Connection &rc)
{

    valid_ = rc.valid_;
    frmModName_ = rc.frmModName_;
    toModName_ = rc.toModName_;
    frmPrtName_ = rc.frmPrtName_;
    toPrtName_ = rc.toPrtName_;
    frmPrtObjName_ = rc.frmPrtObjName_;
    frmMod_ = rc.frmMod_;
    toMod_ = rc.toMod_;
    frmNetIdx_ = rc.frmNetIdx_;
    toNetIdx_ = rc.toNetIdx_;
    toHost_ = rc.toHost_;
    frmHost_ = rc.frmHost_;

    return *this;
}

std::ostream &operator<<(std::ostream &s, const Connection &con)
{
    // write it to s
    if (con.valid_ == Connection::Valid)
    {
        s << con.frmModName_.c_str() << std::endl;
        s << con.frmNetIdx_ << std::endl;
        s << con.frmHost_.c_str() << std::endl;
        s << con.frmPrtName_.c_str() << std::endl;
        s << con.frmPrtObjName_.c_str() << std::endl;
        s << con.toModName_.c_str() << std::endl;
        s << con.toNetIdx_ << std::endl;
        s << con.toHost_.c_str() << std::endl;
        s << con.toPrtName_.c_str() << std::endl;
    }
    return s;
}

void
Connection::warning(std::ostream &s) const
{
    if (valid_ != Valid)
    {
        s << " WARNING: Connection from module "
          << (valid_ & ValidSourceModule ? "" : "*") << frmModName_.c_str()
          << ":" << (valid_ & ValidSourcePort ? "" : "*") << frmPrtName_.c_str()
          << " to module "
          << (valid_ & ValidDestinationModule ? "" : "*") << toModName_.c_str()
          << ":" << (valid_ & ValidDestinationPort ? "" : "*") << toPrtName_.c_str()
          << " is NOT VALID!" << std::endl;
        s << "          Connection will be REMOVED! " << std::endl;
    }
}

std::string
Connection::frmModName() const
{
    return frmModName_;
}

std::string
Connection::toModName() const
{
    return toModName_;
}

std::string
Connection::frmModIdx() const
{
    char nc[32];
    sprintf(nc, "%d", frmNetIdx_);
    std::string ret(nc);
    return ret;
}

std::string
Connection::toModIdx() const
{
    char nc[32];
    sprintf(nc, "%d", toNetIdx_);
    std::string ret(nc);
    return ret;
}

std::string
Connection::frmPort() const
{
    return frmPrtName_;
}

std::string
Connection::toPort() const
{
    return toPrtName_;
}

void
Connection::build()
{
    valid_ = Invalid;

    if (!frmMod_.empty())
        valid_ |= ValidSourceModule;

    if (!toMod_.empty())
        valid_ |= ValidDestinationModule;

    if (frmMod_.empty() || toMod_.empty())
    {
        // 	    cerr << " Connection::build():   empty modules found!"
        // 		 << " -invalidated-" << std::endl;
        return;
    }
    else
    {
        frmModName_ = frmMod_.getName();
        toModName_ = toMod_.getName();
        // 	    cerr << " Connection::build(): try to find port <"
        // 		 << frmPrtName_
        // 		 << "> in module <"
        // 		 << frmModName_ << ">" << std::endl;

        PortSkel prt(frmMod_.getPort(frmPrtName_));
        if (prt.empty())
        {
            // 		  cerr << " Connection::build():   empty port found!"
            // 		       << " -invalidated-"
            // 		       << prt.getName() << std::endl;
            return;
        }
        else
        {
            valid_ |= ValidSourcePort;
            frmPrtName_ = prt.getName();
            frmPrtObjName_ = prt.getCoObjName();
        }

        prt = toMod_.getPort(toPrtName_);
        if (prt.empty())
        {
            // 		  cerr << " Connection::build():   empty port found!"
            // 		       << " -invalidated-"
            // 		       << prt.getName() << std::endl;
            return;
        }
        else
        {
            valid_ |= ValidDestinationPort;
            toPrtName_ = prt.getName();
        }

        assert(valid_ == Valid);
    }
}

//
// Destructor
//
Connection::~Connection()
{
}
