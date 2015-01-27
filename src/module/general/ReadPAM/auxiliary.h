/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_PAM_AUXILIAR_H_
#define _READ_PAM_AUXILIAR_H_

#define COIDENT "$Header: /vobs/covise/src/application/general/READ_PAM/aux.h /main/vir_main/1 18-Dec-2001.11:15:29 we_te $"
#include <util/coIdent.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>

#include <string>
#include <api/coModule.h>
using namespace covise;
#include <util/coIntHash.h>
#include <limits.h>

#define _INCLUDE_SPH_

typedef int INDEX[9];

struct coStringObj
{
    enum Type // tensor symmetries are not considered ???
    {
        NONE = 0,
        SCALAR = 1,
        VECTOR = 3,
        TENSOR_2D = 4,
        TENSOR_3D = 9
    } ObjType_;
    // AQUI
    enum ElemType
    {
        NODAL = 0,
        SOLID = 1,
        SHELL = 2,
        TOOL = 10,
        BAR = 5,
        BEAM = 4,
        BAR1 = 101,
        SPRING = 103,
        SPH_JOINT = 104,
        FLX_TOR_JOINT = 105,
        SPOTWELD = 106,
        JET = 107,
        KINE_JOINT = 108,
        MESI_SPOTWELD = 109,
#ifdef _INCLUDE_SPH_
        SPH = 16, // 21 in dsyalt!!!!
#endif
        NODAL_ADDI = 12,
        GLOBAL = 31,
        MATERIAL = 32,
        TRANS_SECTION = 33,
        CONTACT_INTERFACE = 34,
        RIGID_WALL = 35,
        AIRBAG = 36,
        AIRBAG_CHAM = 38,
        AIRBAG_WALL = 39
    } ElemType_;
    std::string ObjName_; // variable title in the sense of the DSY docu
    char component_; // relevant for isolated components of vector or tensor
    // In this cases ObjType_ is marked as SCALAR
    INDEX index_;
    int position_; // used by special globals

    coStringObj()
    {
    }
    void fill_index(const INDEX ind);
    coStringObj(const coStringObj &input);
    coStringObj &operator=(const coStringObj &rhs);
    int operator==(const coStringObj &rhs)
    {
        if (ObjType_ != rhs.ObjType_ || ElemType_ != rhs.ElemType_ || ObjName_ != rhs.ObjName_ || component_ != rhs.component_ || position_ != rhs.position_)
            return 0;
        switch (ObjType_)
        {
        case VECTOR:
            if (index_[1] != rhs.index_[1] || index_[2] != rhs.index_[2])
                return 0;
        default:
            if (index_[0] != rhs.index_[0])
                return 0;
            break;
        }
        return 1;
    }
};

// AQUI
#ifdef _INCLUDE_SPH_
const int noTypes = 6 + 7 + 1;
#else
const int noTypes = 6 + 7;
#endif
const coStringObj::ElemType Types[] = {
    coStringObj::SOLID, coStringObj::SHELL,
    coStringObj::TOOL, coStringObj::BAR, coStringObj::BEAM,
    coStringObj::BAR1, coStringObj::SPRING,
    coStringObj::SPH_JOINT, coStringObj::FLX_TOR_JOINT,
    coStringObj::SPOTWELD, coStringObj::JET, coStringObj::KINE_JOINT,
    coStringObj::MESI_SPOTWELD
#ifdef _INCLUDE_SPH_
    ,
    coStringObj::SPH
#endif
};
const int connNumbers[] = {
    9, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
#ifdef _INCLUDE_SPH_
    ,
    2
#endif
};
// const int localRef[] = {0,1,0,0,1,0,1,1,1,0,0,1,0};
const int localRef[] = {
    0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0
#ifdef _INCLUDE_SPH_
    ,
    0
#endif
};
const int coviseType[] = {
    TYPE_HEXAEDER, TYPE_QUAD, TYPE_QUAD, TYPE_BAR, TYPE_BAR,
    TYPE_BAR, TYPE_BAR, TYPE_BAR, TYPE_BAR, TYPE_BAR,
    TYPE_BAR, TYPE_BAR, TYPE_BAR
#ifdef _INCLUDE_SPH_
    ,
    TYPE_POINT
#endif
};

class whichContents
{
    coStringObj *theContents_;
    int length_;
    int number_;
    void eliminate(int k); // eliminate element k
public:
    void clear()
    {
        delete[] theContents_;
        theContents_ = 0;
        length_ = 0;
        number_ = 0;
    }
    void reset();
    int operator==(whichContents &rhs);
    whichContents()
    {
        theContents_ = 0;
        length_ = 0;
        number_ = 0;
    }
    ~whichContents()
    {
        delete[] theContents_;
    }
    int no_options()
    {
        return number_;
    }
    coStringObj &operator[](int i)
    {
        return theContents_[i];
    }

    // before compression vector and tensor entries are only
    // present in the form of single components
    void compress(int base);

    void add(const char *new_option, coStringObj::Type new_type,
             coStringObj::ElemType new_elem_type, INDEX ind,
             char component = '0', int position = 0);

    whichContents(const whichContents &input);
    whichContents &operator=(const whichContents &rhs);
};

class Map1D
{
private:
    int *mapping_;
    int min_;
    int max_;
    int length_;
    enum method
    {
        TRIVIAL,
        HASHING
    } method_;
    // label numbers are mapped into natural numbers
    // using a fast mapping if the set of spanned labels
    // (from the minimum to the maximum, including those
    // that are not used)
    // encompasses less than "TRIVIAL_LIMIT" numbers
    // the problem is that if there are too many unused
    // labels in this set;
    // otherwise, hashing is used, which may be especially
    // convienient if the first method could take up
    // too much memory in an inefficient way. Hashing
    // is slower, but less problems with memory usage
    // may be expected.
    static const int TRIVIAL_LIMIT = 1000000;
    coIntHash<int> labels_;

public:
    // list enthaelt labels
    Map1D &operator=(const Map1D &rhs);
    void setMap(int l = 0, int *list = 0);
    Map1D(int l = 0, int *list = 0);
    //   ~Map1D(){ delete [] mapping_;}
    ~Map1D()
    {
    }
    const int &operator[](int i);
};

struct TensDescriptions
{
    int no_request; // number of tensors that have to be constructed
    int *req_label; // cell variable label: 9*no_request ints
    // filled labels follow the order given in
    // coDoTensor
    // the values have been obtained from getValue()-1
    std::string *requests; // names from port->getObjName
    coDoTensor::TensorType *ttype;
    char *markPort; // only used by ReadPAM not by readDSY

    TensDescriptions()
    {
        no_request = 0;
        req_label = 0;
        requests = 0;
        ttype = 0;
        markPort = 0;
    }
    ~TensDescriptions()
    {
        delete[] req_label;
        delete[] requests;
        delete[] ttype;
        delete[] markPort;
    }
};
#endif
