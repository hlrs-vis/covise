/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Node
//
//  This class describes a node with possibly attached data.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _ABAQUS_NODE_H_
#define _ABAQUS_NODE_H_

#include <util/coviseCompat.h>
typedef uint64_t uint64;
typedef int64_t int64;
#include "odb_FieldValue.h"

class Data;

class Node
{
public:
    Node(int label, const vector<float> &coordinates);
    virtual ~Node();
    Node(const Node &rhs);
    Node &operator=(const Node &rhs);
    bool operator==(const Node &rhs) const;
    bool operator==(int rhs) const;
    bool operator<(const Node &rhs) const;
    bool operator<(int rhs) const;
    friend bool operator<(int lhs, const Node &node);
    void SetDisplacement(const vector<int> &order,
                         const odb_SequenceFloat &data);
    void AddData(Data *data);
    const Data *GetData() const;
    void Result();
    void Result(Data *someData);
    int label() const;
    void Coordinates(float &x, float &y, float &z) const;
    void Displacements(float &x, float &y, float &z) const;
    void NodeData(vector<int> &dataNodeLabels) const;

    static float DISP_SCALE;

protected:
private:
    int _label;
    float _coordinates[3];
    float _displacement[3];
    vector<Data *> _data;
    Data *_result;
};
#endif
