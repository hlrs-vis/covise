/***************************************************************************
                          topo.h  -  description
                             -------------------
    begin                : Wed Jul 26 2006
	copyright			: (C) 2006-2014 IAG, University of Stuttgart
	email				: acco@iag.uni-stuttgart.de
 ***************************************************************************/

#ifndef TOPO_H
#define TOPO_H


#include <algorithm>
#include <iomanip>
#include "coord.h"
#include "assert.h"


typedef Coordinate Position;
typedef Coordinate Vector;



// ----------------------------------------
// ------------- Points -------------------
// ----------------------------------------
struct State {
	double mR;
	Vector mU; // fluid velicity
	double mP;
	Vector mV; // grid velocity
	State() : mR(), mU(), mP(), mV() {}
};

struct PointState : Position, State {
	PointState() : Position(), State(), mN() {}
	Vector mN;
};

// ----------------------------------------
// ------------- Topology -----------------
// ----------------------------------------
struct TopoState : State {
	double mSpace;
	Position mCenter;
};

template<int P>
struct Topo : TopoState {
	enum { NO_POINT=-1, POINTS=P };
	int mNodes[POINTS];
	Topo() { std::fill(mNodes, mNodes+POINTS, NO_POINT); }
	bool isCell() const;
	template<typename point_container_t>
	void computeCenter(point_container_t iPoints);
	template<typename point_container_t, typename member_t, typename dest_t>
	void computeAverage(point_container_t iPoints, member_t iSource, dest_t & oDest);
};

template<int P>
struct TopoLine : Topo<P> {
	Vector mForce;
	template<typename point_container_t>
	void computeGeometry(point_container_t iPoints);
	using Topo<P>::mNodes;
};

template<int P>
struct TopoSurface : Topo<P> {
	Vector mNormal;
	template<typename point_container_t>
	void computeGeometry(point_container_t iPoints);
	void invert();
	using Topo<P>::mNodes;
};

template<int P>
struct TopoVolume : Topo<P> {
	template<typename point_container_t>
	void computeGeometry(point_container_t iPoints);
	using Topo<P>::mNodes;
};

typedef TopoLine<2> LineTopo;
typedef TopoSurface<3> TriangleTopo;
typedef TopoSurface<4> QuadrangleTopo;
typedef TopoVolume<8> HexaederTopo;
typedef TopoVolume<4> TetraederTopo;

template<int P>
inline bool Topo<P>::isCell() const {
	// only cell centered, if no node is valid
	for (int p=0; p<P; ++p)
		if (mNodes[p]>=0) return false;
	return true;
}

template<int P>
template<typename point_container_t>
inline void Topo<P>::computeCenter(point_container_t iPoints) {
	mCenter=0;
	int unusedPoints=0;
	for (int p=0; p<P; ++p) {
		if (mNodes[p]>=0) {
			mCenter+=iPoints[mNodes[p]];
		}
		else ++unusedPoints;
	}
	mCenter*=1.0/(P-unusedPoints);
}

template<int P>
template<typename point_container_t, typename member_t, typename dest_t>
inline void Topo<P>::computeAverage(point_container_t iPoints, member_t iSource, dest_t & oDest) {
	oDest=0;
	int unusedPoints=0;
	for (int p=0; p<P; ++p) {
		if (mNodes[p]>=0) {
			oDest+=iPoints[mNodes[p]].*iSource;
		}
		else ++unusedPoints;
	}
	oDest*=1.0/(P-unusedPoints);
}

template<int P>
inline void TopoSurface<P>::invert() {
	assert(P==3 || P==4);
	std::swap(mNodes[1], mNodes[P-1]); // works for triangles and quadrangles
}

template<>
template<typename point_container_t>
inline void TopoLine<2>::computeGeometry(point_container_t iPoints) {
	assert(mNodes[0]>=0 && mNodes[1]>=0);
}

template<>
template<typename point_container_t>
inline void TopoSurface<3>::computeGeometry(point_container_t iPoints) {
	assert(mNodes[0]>=0 && mNodes[1]>=0 && mNodes[2]>=0);
	computeCenter(iPoints);
	mNormal=cross(iPoints[mNodes[1]]-iPoints[mNodes[0]], iPoints[mNodes[2]]-iPoints[mNodes[0]]);
	mSpace=0.5*mNormal.length();
	if (mSpace!=0) mNormal*=0.5/mSpace;
}

template<>
template<typename point_container_t>
inline void TopoSurface<4>::computeGeometry(point_container_t iPoints) {
	assert(mNodes[0]>=0 && mNodes[1]>=0 && mNodes[2]>=0 && mNodes[3]>=0);
	computeCenter(iPoints);
#if 1
	mNormal=cross(iPoints[mNodes[2]]-iPoints[mNodes[0]], iPoints[mNodes[3]]-iPoints[mNodes[1]]);
	mSpace=0.5*mNormal.length();
	if (mSpace!=0) mNormal*=0.5/mSpace;
#else
	Vector n1=cross(iPoints[mNodes[1]]-iPoints[mNodes[0]], iPoints[mNodes[3]]-iPoints[mNodes[0]]);
	Vector n2=cross(iPoints[mNodes[3]]-iPoints[mNodes[2]], iPoints[mNodes[1]]-iPoints[mNodes[2]]);
// 	if (n1.scalar(n2)<0.95*n1.length()*n2.length()) {
// 		std::cerr << "Distorted surface at points ";
//		SHOW_VAR(mNodes[0]. mNodes[1], mNodes[2], mNodes[3], n1, n2);
// 		SHOW_VAR5(setprecision(9), iPoints[mNodes[0]], iPoints[mNodes[1]], iPoints[mNodes[2]], iPoints[mNodes[3]]);
// 		SHOW_VAR4(iPoints[mNodes[0]].mN, iPoints[mNodes[1]].mN, iPoints[mNodes[2]].mN, iPoints[mNodes[3]].mN);
// 	}
	mNormal=n1+n2;
	mSpace=0.5*mNormal.length();
	if (mSpace!=0) mNormal*=0.5/mSpace;
#endif
}

template<>
template<typename point_container_t>
inline void TopoVolume<4>::computeGeometry(point_container_t iPoints) {
	assert(mNodes[0]>=0 && mNodes[1]>=0 && mNodes[2]>=0 && mNodes[3]>=0);
	computeCenter(iPoints);
	Vector d1=iPoints[mNodes[1]]-iPoints[mNodes[0]];
	Vector d2=iPoints[mNodes[2]]-iPoints[mNodes[0]];
	Vector d3=iPoints[mNodes[3]]-iPoints[mNodes[0]];
	mSpace=1.0/6*std::abs((d3*( d1^d2)));
}

template<>
template<typename point_container_t>
inline void TopoVolume<8>::computeGeometry(point_container_t iPoints) {
	assert(mNodes[0]>=0 && mNodes[1]>=0 && mNodes[2]>=0 && mNodes[3]>=0);
	assert(mNodes[4]>=0 && mNodes[5]>=0 && mNodes[6]>=0 && mNodes[7]>=0);
	computeCenter(iPoints);
	Vector d1=iPoints[mNodes[1]]-iPoints[mNodes[0]];
	Vector d2=iPoints[mNodes[3]]-iPoints[mNodes[0]];
	Vector d3=iPoints[mNodes[4]]-iPoints[mNodes[0]];
	double spat1=(d1^d2)*(d3);
	d1=iPoints[mNodes[2]]-iPoints[mNodes[6]];
	d2=iPoints[mNodes[5]]-iPoints[mNodes[6]];
	d3=iPoints[mNodes[7]]-iPoints[mNodes[6]];
	double spat2= (d1^d2)*(d3);
	mSpace=0.5*std::abs(spat1+spat2);
}

#endif
