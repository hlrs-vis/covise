#ifndef VISTLE_READENSIGHT_DATAITEM_H
#define VISTLE_READENSIGHT_DATAITEM_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    DataItem
//
// Description: data-class for description of data contained in EnSight
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

#include <string>

class DataItem {
public:
    enum Type { scalar, vector, tensor };
    enum Mapping { PerCase, PerNode, PerElement };

    // default CONSTRUCTOR
    DataItem();

    // DESTRUCTOR
    virtual ~DataItem();

    void setType(Type tp) { type_ = tp; }
    void setFileName(const std::string &fn) { fileName_ = fn; }
    void setDesc(const std::string &ds) { desc_ = ds; }
    void setMeasured(bool t) { measured_ = t; }
    void setMapping(Mapping m) { mapping_ = m; }
    void setTimeSet(int ts) { timeSet_ = ts; }

    bool perVertex() const // bad name
    {
        return mapping_ == PerNode;
    }
    Mapping mapping() const { return mapping_; }
    Type getType() const { return type_; }
    std::string getFileName() const { return fileName_; }
    std::string getDesc() const { return desc_; }
    int getTimeSet() const { return timeSet_; }

private:
    Type type_; // scalar vector tensor
    std::string fileName_;
    std::string desc_;
    bool measured_;
    Mapping mapping_;
    int timeSet_ = -1;
};
#endif
