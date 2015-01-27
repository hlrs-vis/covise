// ITE-Toolbox Interface to Comsol
//
// (C) Institute for Theory of Electrical Engineering
//
// author: A. Buchau
//
// basic element types

#pragma once

struct Tetra4
{
    unsigned int domainNo;
    unsigned long node1;
    unsigned long node2;
    unsigned long node3;
    unsigned long node4;
};
