/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS  EnFile
// CLASS  DataCont
//
// Description: Ensight file representation ( base class)
//
// Initial version: 01.06.2002 by RM
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef ENFILE_H
#define ENFILE_H

#include <util/coviseCompat.h>

#include "EnElement.h"
#include "EnPart.h"
#include "CaseFile.h"

namespace covise
{
class coModule;
}

// primitive data container
// be aware DataCont is only an assembly of pointers and numbers
// USE THIS CLASS WITH CARE !!! it is dangerous !!!
class DataCont
{
public:
    DataCont();

    virtual ~DataCont();

    void setNumCoord(const int &n)
    {
        nCoord_ = n;
    };
    void setNumElem(const int &n)
    {
        nElem_ = n;
    };
    void setNumConn(const int &n)
    {
        nConn_ = n;
    };

    int getNumCoord() const
    {
        return nCoord_;
    };
    int getNumElem() const
    {
        return nElem_;
    };
    int getNumConn() const
    {
        return nConn_;
    };

    float *x;
    float *y;
    float *z;

    int *el;
    int *cl;
    int *tl;

    void cleanAll();

private:
    int nCoord_;
    int nElem_;
    int nConn_;
};

class InvalidWordException
{
public:
    InvalidWordException(const string &type);
    string what();

private:
    string type_;
};

//
// base class for Ensight geometry files
// provide general methods for reading geometry files
//
class EnFile
{
public:
    enum
    {
        OFF,
        GIVEN,
        ASSIGN,
        EN_IGNORE
    };
    typedef enum
    {
        CBIN,
        FBIN,
        NOBIN,
        UNKNOWN
    } BinType;

    EnFile(const coModule *mod, const BinType &binType = UNKNOWN);

    EnFile(const coModule *mod, const string &name, const BinType &binType = UNKNOWN);

    EnFile(const coModule *mod, const string &name, const int &dim, const BinType &binType = UNKNOWN);

    bool isOpen();

    // check for binary file
    BinType binType();

    // read the file
    virtual void read(){};
    // read cell based data
    virtual void readCells(){};

    // Return data in data container.
    // this function is dangerous as only adresses are returned
    // copying the data would be to expensive
    DataCont getDataCont() const;

    // Set the master part list. This is the list of all part in the geometry file
    // or the geo. file for the first timestep. This means we must still check the
    // length of the connection table
    virtual void setMasterPL(PartList p);

    virtual void setPartList(PartList *p);

    virtual ~EnFile();

    virtual void parseForParts(){};

    // fill data container with data values ( only for the case of cell based data)
    virtual void buildParts(const bool &isPerVert = false);

    // use this function to create a geometry Ensight file representation
    // each Ensight geometry file has a own fctory to create associated
    // data files
    // you will have to change this method each time you enter a new type of
    // Ensight Geometry
    static EnFile *createGeometryFile(const coModule *mod, const CaseFile &c, const string &filename);

    void setActiveAlloc(const bool &b);

    void setDataByteSwap(const bool &v);
    void setIncludePolyeder(const bool &b);
    virtual coDistributedObject *getDataObject(std::string)
    {
        return NULL;
    };

    bool fileMayBeCorrupt_;

protected:
    // functions used for BINARY input
    virtual string getStr();

    // skip n ints
    void skipInt(const int &n);

    // skip n floats
    void skipFloat(const int &n);

    // get integer
    virtual int getInt();
    int getIntRaw();

    // get integer array
    virtual int *getIntArr(const int &n, int *iarr = NULL);

    // get float array
    virtual float *getFloatArr(const int &n, float *farr = NULL);

    // find a part by its part number
    virtual EnPart *findPart(const int &partNum) const;

    // find a part by its part number in the master part list
    virtual EnPart findMasterPart(const int &partNum) const;

    virtual void resetPart(const int &partNum, EnPart *p);

    // send a list of all parts to covise info
    void sendPartsToInfo();

    string className_;

    bool isOpen_;

    FILE *in_;

    int nodeId_;

    int elementId_;

    DataCont dc_; // container for read data

    BinType binType_;

    bool byteSwap_;

    PartList *partList_;

    PartList masterPL_;

    int dim_;

    bool activeAlloc_;

    bool dataByteSwap_;

    bool includePolyeder_;
    // pointer to module for sending ui messages
    const coModule *module_;

private:
    string name_;

    void getIntArrHelper(const int &n, int *iarr = NULL);
};
#endif
