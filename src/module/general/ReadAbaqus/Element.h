/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Element
//
//  Description of an element and possibly attached data.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _ABAQUS_ELEMENT_H_
#define _ABAQUS_ELEMENT_H_

#include <util/coviseCompat.h>
#include "Node.h"
#include <do/coDoUnstructuredGrid.h>
class Data;

class Element
{
public:
    Element(int label, const char *type, const vector<int> &connectivity);
    ~Element();
    Element(const Element &rhs);
    Element &operator=(const Element &rhs);
    bool operator==(const Element &rhs) const;
    bool operator<(const Element &rhs) const;
    bool operator==(int rhs) const;
    bool operator<(int rhs) const;
    friend bool operator<(int lhs, const Element &elem);
    void AddData(Data *data);
    void Result(vector<Node> &nodes);
    void ElementDataConnectivity(vector<int> &elem_data,
                                 vector<int> &elem_no_data,
                                 vector<int> &conn_data,
                                 vector<int> &conn_no_data,
                                 vector<int> &type_data,
                                 vector<int> &type_no_data,
                                 vector<const Data *> &data_per_element) const;
    void NodalDataConnectivity(vector<int> &elem,
                               vector<int> &conn,
                               vector<int> &type) const;
    bool AllNodesHaveData(const vector<int> &dataNodeLabels) const;
    bool SomeNodesHaveData(const vector<int> &dataNodeLabels,
                           vector<int> &OtherNodes) const;

protected:
private:
    int _label;

    string _type;
    int _shape;

    vector<int> _connectivity;
    vector<int> _reduced_connectivity;

    vector<Data *> _data;
    Data *_result;

    vector<Element> _sub_elements;

    void CopyConnectivity(int num);

    static bool Continuum1D(const char *type, int &num);
    static bool Continuum2D(const char *type, int &num);
    static bool Continuum3D(const char *type, int &num);
    static bool ContinuumAX(const char *type, int &num);
    static bool ContinuumCyl(const char *type, int &num);
    static bool Membrane3D(const char *type, int &num);
    static bool MembraneAX(const char *type, int &num);
    static bool MembraneCyl(const char *type, int &num);
    static bool Truss(const char *type, int &num);
    static bool Beam(const char *type, int &num);
    static bool Frame(const char *type);
    static bool Shell3D(const char *type, int &num);
    static bool ShellAX(const char *type, int &num);
    static bool Rigid(const char *type, int &num);
};
#endif
