/***************************************************************************
                          mesh.cpp  -  description
                             -------------------
    begin                : Wed Jul 26 2006
	copyright			: (C) 2006-2014 IAG, University of Stuttgart
	email				: acco@iag.uni-stuttgart.de
 ***************************************************************************/

#ifndef MESH_H
#define MESH_H


#include <vector>
#include <string>
#include "coord.h"

typedef Coordinate Position;
typedef Coordinate Vector;

struct PointState;
struct TopoState;
struct AcousticLineSources;
struct AcousticSurfaceSources;
struct AcousticVolumeSources;

enum NormalCheck { ALL_NORMALS=-1, NO_NORMALS, USE_READ, CHECK_READ };

class MeshBase {
public:
	MeshBase(int iNumCells, int iNumPoints);
	virtual ~MeshBase();
	
	int getNumCells() const { return mNumCells; }
	int getNumPoints() const { return mNumPoints; }
	void setName(std::string const & iName) { mName=iName; }
	std::string getName() const { return mName; }
	void setReferenceState(double iRho0, double iP0, double iC, Vector const & iFreestream, Vector const & iFlightspeed);
	virtual void cpToP(double iMachRef) = 0;
	virtual void dump(std::ostream & out);

	virtual TopoState & getState(int n) = 0;
	
	virtual AcousticLineSources getLineSource(int iLine, Position const & iObserver, double iTime);
	virtual AcousticSurfaceSources getSurfaceSource(int iCell, Position const & iObserver, double iTime);
	virtual AcousticVolumeSources getVolumeSource(int iCell, Position const & iObserver, double iTime);

	void addMechanics(Vector * oForce, Vector * oMoment, double * oPower);

	virtual void SetupLine();
	virtual void SetupVolume();
	virtual void SetupSurface(std::vector<Position> const & iInnerPoints, NormalCheck iCheck);

	virtual void Rotate(Position const & iCenter, double iTime, Position const & iAngularVelocity, Position const & iLinearVelocity) = 0;
	virtual void ReverseNormals();

protected:
	struct Transport {
		Position mR0;
		double mRAbs, mPs, mMr, mT, mDoppler;
	};
	template<typename SOURCE>
	void transport(SOURCE const & iSource, Position const & iObserver, double iTime, Transport & oTransport);
	virtual void getMechanics(int iCell, double * oSpace, Vector * oX, Vector * oF, Vector * oV) const = 0;
	int mNumCells;
	int mNumPoints;
	std::string mName;
	double mRho0;
	double mP0;
	double mC;
	Vector mFreestream, mFlightspeed;
};

template <class TOPO>
class Mesh : public MeshBase {
public:
	Mesh();
	Mesh(int iNumCells, int iNumPoints, TOPO * iCells, PointState * iPoints);
	~Mesh();

	virtual void cpToP(double iMachRef);
	virtual void Rotate(Position const & iCenter, double iTime, Position const & iAngularVelocity, Position const & iLinearVelocity);

protected:
	typedef TOPO topology_t;
	topology_t * mCells;
	PointState * mPoints;
	virtual TopoState & getState(int n);
};

template <class TOPO>
class LineMesh : public Mesh<TOPO> {
public:
	LineMesh(int iNumCells, int iNumPoints, TOPO * iCells, PointState * iPoints);
	virtual ~LineMesh();
	virtual void SetupLine();
	virtual AcousticLineSources getLineSource(int iCell, Position const & iObserver, double iTime);
	virtual void dump(std::ostream & out);
protected:
	virtual void getMechanics(int iCell, double * oSpace, Vector * oX, Vector * oF, Vector * oV) const;
	using Mesh<TOPO>::mNumCells;
	using Mesh<TOPO>::mCells;
	using Mesh<TOPO>::mPoints;
	using Mesh<TOPO>::mRho0;
	using Mesh<TOPO>::mP0;
	using Mesh<TOPO>::mC;
	using Mesh<TOPO>::mFreestream;
	using Mesh<TOPO>::mFlightspeed;
private:
	typedef TOPO topology_t;
	AcousticLineSources observe(topology_t const & iCell, Position const & iObserver, double iTime);
};

template <class TOPO>
class SurfaceMesh : public Mesh<TOPO> {
public:
	SurfaceMesh(int iNumCells, int iNumPoints, TOPO * iCells, PointState * iPoints);
	virtual ~SurfaceMesh();
	virtual void SetupSurface(std::vector<Position> const & iInnerPoint, NormalCheck iCheck);
	virtual void ReverseNormals();
	virtual AcousticSurfaceSources getSurfaceSource(int iCell, Position const & iObserver, double iTime);
	virtual void dump(std::ostream & out);
protected:
	virtual void getMechanics(int iCell, double * oSpace, Vector * oX, Vector * oF, Vector * oV) const;
	using Mesh<TOPO>::mNumCells;
	using Mesh<TOPO>::mCells;
	using Mesh<TOPO>::mPoints;
	using Mesh<TOPO>::mRho0;
	using Mesh<TOPO>::mP0;
	using Mesh<TOPO>::mC;
	using Mesh<TOPO>::mFreestream;
	using Mesh<TOPO>::mFlightspeed;
	using Mesh<TOPO>::getName;
private:
	typedef TOPO topology_t;
	AcousticSurfaceSources observe(topology_t const & iCell, Position const & iObserver, double iTime);
};

template <class TOPO>
class VolumeMesh : public Mesh<TOPO> {
public:
	VolumeMesh(int iNumCells, int iNumPoints, TOPO * iCells, PointState * iPoints);
	virtual ~VolumeMesh();
	virtual void SetupVolume();
	virtual AcousticVolumeSources getVolumeSource(int iCell, Position const & iObserver, double iTime);
protected:
	virtual void getMechanics(int iCell, double * oSpace, Vector * oX, Vector * oF, Vector * oV) const;
	using Mesh<TOPO>::mNumCells;
	using Mesh<TOPO>::mCells;
	using Mesh<TOPO>::mPoints;
	using Mesh<TOPO>::mRho0;
	using Mesh<TOPO>::mP0;
	using Mesh<TOPO>::mC;
	using Mesh<TOPO>::mFreestream;
	using Mesh<TOPO>::mFlightspeed;
	typedef TOPO topology_t;
	AcousticVolumeSources observe(topology_t const & iCell, Position const & iObserver, double iTime);
};

struct MeshPts {
	MeshPts(int iNumPoints, Position const * iPoints);
	~MeshPts();

	int mNumPoints;
	Position const * mPoints;
};

#endif
