/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    DataItem
//
// Description: data-class for description of data contained in ENSIGHT
//              case file
//
// Initial version: 23.05.2002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef DATAITEM_H
#define DATAITEM_H

#include <string>

class DataItem
{
public:
    enum
    {
        scalar,
        vector,
        tensor
    };

    // default CONSTRUCTOR
    DataItem();
    DataItem(const int &type, const std::string &file, const std::string &desc);

    // DESTRUCTOR
    virtual ~DataItem();

    void setType(const int &tp)
    {
        type_ = tp;
    };
    void setFileName(const std::string &fn)
    {
        fileName_ = fn;
    };
    void setDesc(const std::string &ds)
    {
        desc_ = ds;
    };
    void setDataType(const bool &t)
    {
        perVertex_ = t;
    };
    void setMeasured(const bool &t)
    {
        measured_ = t;
    };

    bool perVertex() // bad name
    {
        return perVertex_;
    };
    int getType() const
    {
        return type_;
    };
    std::string getFileName() const
    {
        return fileName_;
    };
    std::string getDesc() const
    {
        return desc_;
    };

private:
    int type_; // scalar vector tensor
    std::string fileName_;
    std::string desc_;
    bool perVertex_;
    bool measured_;
};
#endif
