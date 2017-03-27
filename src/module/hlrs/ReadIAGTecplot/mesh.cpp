/***************************************************************************
                          mesh.cpp  -  description
                             -------------------
    begin                : Wed Jul 26 2006
	copyright			: (C) 2006-2014 IAG, University of Stuttgart
	email				: acco@iag.uni-stuttgart.de
 ***************************************************************************/

#include "mesh.h"

#include <topo.h>
#include <sources.h>


#include <algorithm>
#include <fstream>
#include <cmath>

// predicate to select which point is closer to a reference point given in the constructor
struct NearPred : public std::binary_function<Position, Position, bool> {
	NearPred(Position const & iDest) : mDest(iDest) {}
	bool operator()(Position const & lhs, Position const & rhs) const {
		return (lhs-mDest).length()<(rhs-mDest).length();
	}
private:
	Position mDest;
};

// ----------------------------------------
// ------------- MeshBase ----------------
// ----------------------------------------

MeshBase::MeshBase(int iNumberOfCells, int iNumberOfPoints)
	: mNumCells(iNumberOfCells), mNumPoints(iNumberOfPoints), mRho0(1), mP0(1), mC(1) {}

MeshBase::~MeshBase() {}

void MeshBase::setReferenceState(double iRho0, double iP0, double iC, Vector const & iFreestream, Vector const & iFlightspeed) {
	mRho0=iRho0;
	mP0=iP0;
	mC=iC;
	mFreestream=iFreestream;
	mFlightspeed=iFlightspeed;
}

void MeshBase::ReverseNormals() {
	assert(0=="Pointless MeshBase::ReverseNormals");
	throw "Pointless MeshBase::ReverseNormals";
}

AcousticLineSources MeshBase::getLineSource(int /*iCell*/, Position const & /*iObserver*/, double /*iTime*/) {
	assert(0=="Pointless MeshBase::getLineSource");
	throw "Pointless MeshBase::getLineSource";
}

AcousticSurfaceSources MeshBase::getSurfaceSource(int /*iCell*/, Position const & /*iObserver*/, double /*iTime*/) {
	assert(0=="Pointless MeshBase::getSurfaceSources");
	throw "Pointless MeshBase::getSurfaceSources";
}

AcousticVolumeSources MeshBase::getVolumeSource(int /*iCell*/, Position const & /*iObserver*/, double /*iTime*/) {
	assert(0=="Pointless MeshBase::getVolumeSource");
	throw "Pointless MeshBase::getVolumeSource";
}

void MeshBase::dump(std::ostream & /*out*/) {
	assert(0=="Pointless MeshBase::dump");
	throw "Pointless MeshBase::dump";
}


void MeshBase::addMechanics(Vector * oForce, Vector * oMoment, double * oPower) {
	Vector position(0), force(0), moment(0), velocity(0);
	double space=0, power=0;
	for (int i=0; i<mNumCells; i++) {
		Vector x(0), f(0), v(0);
		double s(0);
		getMechanics(i, &s, &x, &f, &v);
		space+=s;
		position+=x*s;
		force+=f; // force on body
		moment+=x^f; // moment on body
		power-=f*v; // power FROM body
		velocity += v*s;
	}
	position/=space;
	velocity/=space;
//SHOW_VAR7(mName, position, space, force, moment, power, velocity);
	*oForce+=force;
	*oMoment+=moment;
	*oPower+=power;
}

void MeshBase::SetupLine() {
	assert(0=="Pointless MeshBase::SetupLine");
	throw "Pointless MeshBase::SetupLine";
}

void MeshBase::SetupVolume() {
	assert(0=="Pointless MeshBase::SetupVolume");
	throw "Pointless MeshBase::SetupVolume";
}

void MeshBase::SetupSurface(std::vector<Position> const & /*iInnerPoint*/, NormalCheck /*iCheck*/) {
	assert(0=="Pointless MeshBase::SetupSurface");
	throw "Pointless MeshBase::SetupSurface";
}

// ----------------------------------------
// ------------- Mesh --------------------
// ----------------------------------------

template <class TOPO>
Mesh<TOPO>::Mesh() : mCells(0), mPoints(0) {}

template <class TOPO>
Mesh<TOPO>::Mesh(int iNumberOfCells, int iNumberOfPoints, TOPO * iCells, PointState * iPoints)
	: MeshBase(iNumberOfCells, iNumberOfPoints), mCells(iCells), mPoints(iPoints) {}

template <class TOPO>
Mesh<TOPO>::~Mesh() {
	delete[] mCells;
	delete[] mPoints;
}

template <class TOPO>
TopoState & Mesh<TOPO>::getState(int n) { return mCells[n]; }

template <class TOPO>
void Mesh<TOPO>::cpToP(double iMachRef) {
	// p=(1+k/2 MaRef^2 cp) p_inf
	static double const KAPPA=1.4;
	double scale=KAPPA/2*iMachRef*iMachRef*mP0;
	for (int i=0; i<mNumPoints; ++i)
		mPoints[i].mP=mPoints[i].mP*scale+mP0;
}

template <class TOPO>
void Mesh<TOPO>::Rotate(Position const & iCenter, double iTime, 
		Position const & iAngularVelocity, Position const & iLinearVelocity) {
	assert(false);
/*	IAG::Matrix<double,3> rotation(IAG::rotation(iAngularVelocity, iTime*iAngularVelocity.length()));
	Position linear(iLinearVelocity);
	linear*=iTime;
	for (int i=0; i<mNumPoints; ++i) {
		mPoints[i].Position::operator=(rotation*(mPoints[i]-iCenter)+iCenter+linear);
		mPoints[i].mV=IAG::cross(iAngularVelocity, rotation*(mPoints[i]-iCenter))+iLinearVelocity;
		mPoints[i].mN=IAG::cross(iAngularVelocity, rotation*(mPoints[i].mN));
	}*/
}

template <class TOPO>
LineMesh<TOPO>::LineMesh(int iNumberOfCells, int iNumberOfPoints, TOPO * iCells, PointState * iPoints)
	: Mesh<TOPO>(iNumberOfCells, iNumberOfPoints, iCells, iPoints) {}

template <class TOPO>
LineMesh<TOPO>::~LineMesh() {}

template <class TOPO>
SurfaceMesh<TOPO>::SurfaceMesh(int iNumberOfCells, int iNumberOfPoints, TOPO * iCells, PointState * iPoints)
	: Mesh<TOPO>(iNumberOfCells, iNumberOfPoints, iCells, iPoints) {}

template <class TOPO>
SurfaceMesh<TOPO>::~SurfaceMesh() {}

template <class TOPO>
VolumeMesh<TOPO>::VolumeMesh(int iNumberOfCells, int iNumberOfPoints, TOPO * iCells, PointState * iPoints)
	: Mesh<TOPO>(iNumberOfCells, iNumberOfPoints, iCells, iPoints) {}

template <class TOPO>
VolumeMesh<TOPO>::~VolumeMesh() {}

template<class TOPO>
void LineMesh<TOPO>::dump(std::ostream & out) {
	for (int i=0; i<mNumCells; i++) {
		topology_t const & c=mCells[i];
		out << c.mCenter.X() << ' ' << c.mCenter.Y() << ' ' << c.mCenter.Z() << ' '
			<< c.mForce.X() << ' ' << c.mForce.Y() << ' ' << c.mForce.Z() << ' '
			<< mPoints[i].mU.X() << ' ' << mPoints[i].mU.Y() << ' ' << mPoints[i].mU.Z() << ' '
			<< mPoints[i].mV.X() << ' ' << mPoints[i].mV.Y() << ' ' << mPoints[i].mV.Z() << ' '
			<< c.mSpace << std::endl;
	}
}

template<class TOPO>
void SurfaceMesh<TOPO>::dump(std::ostream & out) {
	for (int i=0; i<mNumCells; i++) {
		topology_t const & c=mCells[i];
		out << c.mCenter.X() << ' ' << c.mCenter.Y() << ' ' << c.mCenter.Z() << ' '
			<< c.mNormal.X() << ' ' << c.mNormal.Y() << ' ' << c.mNormal.Z() << std::endl;
	}
}

template<class TOPO>
void SurfaceMesh<TOPO>::SetupSurface(std::vector<Position> const & iInnerPoint, NormalCheck iCheck) {
	assert(false);
	/*
	double minScalar=1, sumScalar=0, sumArea=0;
	bool normalRead=true;
	int numInverted=0;
	int badNormals=0;
	for (int i=0; i<mNumCells; i++) {
#ifdef MARENCO
		if (mCells[i].isCell()) {
			mCells[i].mCenter=mPoints[i];
			mCells[i].mU=mPoints[i].mV; // impermeable (physical) surface
			mCells[i].mV=mPoints[i].mV;
			mCells[i].mR=mRho0;
			mCells[i].mNormal=mPoints[i].mN;
			mCells[i].mP=mP0;
			mCells[i].mSpace=mPoints[i].mP; // P abused for area
		}
		else
#endif
		{
			mCells[i].computeGeometry(mPoints);
			Vector readNormal(0.);
			mCells[i].computeAverage(mPoints, &PointState::mN, readNormal);
			normalRead&=readNormal.norm2()!=0;
			readNormal.normalize();
			if (iCheck==USE_READ) {
				if (normalRead) mCells[i].mNormal=readNormal;
			}
			else if (iCheck==CHECK_READ && normalRead) {
				double scalar=mCells[i].mNormal.scalar(readNormal);
				assert(fabs (mCells[i].mNormal.norm2()-1)<1e-7);
				if (scalar<0) {
					++numInverted;
					mCells[i].mNormal*=-1;
				}
				else if (scalar<0.8) {
					++badNormals;
					IAG_WARNING("Read and computed normals for surface " << i << " on mesh " << getName() << " disagree significantly");
// 					SHOW_VAR4(i, mCells[i].mNormal, readNormal, mCells[i].mNormal.scalar(readNormal));
// 					int p0=mCells[i].mNodes[0], p1=mCells[i].mNodes[1], p2=mCells[i].mNodes[2], p3=mCells[i].mNodes[3];
// 					SHOW_VAR4(p0, p1, p2, p3);
// 					SHOW_VAR4(mPoints[p0], mPoints[p1], mPoints[p2], mPoints[p3]);
// 					SHOW_VAR4(mPoints[p0].mN, mPoints[p1].mN, mPoints[p2].mN, mPoints[p3].mN);
				}
				if (scalar<minScalar) minScalar=scalar;
				sumScalar+=scalar*mCells[i].mSpace;
				sumArea+=mCells[i].mSpace;
			}
			// find the  inner point with minimum distance
			if (iInnerPoint.size()) {
				Position innerPoint = *std::min_element(iInnerPoint.begin(), iInnerPoint.end(),
					NearPred(mCells[i].mCenter));
				Vector connectingVector(mCells[i].mCenter-innerPoint);
				// Check if normal points outwards, otherwise reverse it (without changing point order)
				double scalarProd=connectingVector.scalar(mCells[i].mNormal);
				if (scalarProd<0) {
					++numInverted;
					mCells[i].mNormal*=-1;
				}
			}
			mCells[i].computeAverage(mPoints, &PointState::mR, mCells[i].mR);
			mCells[i].computeAverage(mPoints, &PointState::mU, mCells[i].mU);
			mCells[i].computeAverage(mPoints, &PointState::mP, mCells[i].mP);
			mCells[i].computeAverage(mPoints, &PointState::mV, mCells[i].mV);
		}
	}
	if (iCheck==USE_READ) {
		if (!normalRead) IAG_WARNING("UseRead requested, but no normal read in mesh " << getName());
	}
	else if (iCheck==CHECK_READ) {
		if (!normalRead) IAG_WARNING("CheckRead requested, but no normal read in mesh " << getName());
		else {
			if (numInverted) {
				if (numInverted==mNumCells) IAG_WARNING("Inverted normals in mesh " << getName() << ", corrected");
				else IAG_WARNING("Some (" << numInverted << " out of " << mNumCells << ") inverted normals in mesh " << getName() << ", individually corrected");
			}
			else if (badNormals || sumScalar<0.99*sumArea)
				std::cerr << badNormals << " bad normals for surface " << getName() << ", maximum normal angle deviation is " 
					<< std::acos(minScalar)*180/M_PI  << ", area weighted average is " << std::acos(sumScalar/sumArea)*180/M_PI << std::endl;
		}
	}
	else {
		if (numInverted) {
			if (numInverted==mNumCells) IAG_WARNING("Inverted normals in mesh " << getName() << ", corrected");
			else IAG_WARNING("Some (" << numInverted << " out of " << mNumCells << ") inverted normals in mesh " << getName() << ", individually corrected");
		}
	}*/
}

template<class TOPO>
void SurfaceMesh<TOPO>::ReverseNormals() {
	for (int i=0; i<mNumCells; i++)
		mCells[i].invert();
}

template<class TOPO>
void LineMesh<TOPO>::SetupLine() {
	for (int i=0; i<mNumCells; i++) {
		assert(mCells[i].isCell());
		mCells[i].mCenter=mPoints[i];
		mCells[i].mU=0;
		mCells[i].mV=0;
		mCells[i].mR=mRho0;
		mCells[i].mForce=mPoints[i].mU+mPoints[i].mV; // T+D
		mCells[i].mP=mP0;
		mCells[i].mSpace=mPoints[i].mR*mPoints[i].mP; // dR*clen
	}
}

template<typename TOPO>
void VolumeMesh<TOPO>::SetupVolume() {
	for (int i=0; i<mNumCells; i++) {
		mCells[i].computeGeometry(mPoints);
		mCells[i].computeAverage(mPoints, &PointState::mR, mCells[i].mR);
		mCells[i].computeAverage(mPoints, &PointState::mU, mCells[i].mU);
		mCells[i].computeAverage(mPoints, &PointState::mP, mCells[i].mP);
		mCells[i].computeAverage(mPoints, &PointState::mV, mCells[i].mV);
	}
}

template<typename SOURCE>
void MeshBase::transport(SOURCE const & iSource, Position const & iObserver, double iTime, Transport & oTransport) {
	assert(false);
	/*
	Vector r_0=iObserver-iSource.mCenter-mFlightspeed*iTime;
	if (r_0.norm2()==0) {
		if (Parallel::isMaster) std::cerr << "Attention: observer concides with source point\n";
		throw "Attention: observer concides with source point";
	}
	oTransport.mPs=iSource.mP-mP0;
	// Time t, distance s=t*c, source position z0, observer position b0-t*freestream
	// |b0-t*freestream-z0|^2=t^2*c^2
	// (b0-t*freestream-z0).x^2+...=t^2*c^2
	// (c^2-|freestream|^2) t^2 + 2 freestream.(b0-z0) t - |b0-z0|^2 = 0
	// t=1/(c^2-|freestream^2|)[ - freestream.(b0-z0) \pm \sqrt(freestream.(b0-z)^2+|b0-z0|^2(c^2-|freestream|^2) ) ]
	double wr_0=-mFreestream.scalar(r_0);
	double a=sqr(mC)-mFreestream.norm2();
	double d=wr_0*wr_0+a*r_0.norm2();
	oTransport.mT = iTime + (wr_0+std::sqrt(d))/a;
	Vector r=r_0-(oTransport.mT-iTime)*mFreestream;
	oTransport.mRAbs=r.length();
if (std::abs(oTransport.mRAbs-(oTransport.mT-iTime)*mC)>1e-8) { // consistency check
	SHOW_VAR4(iObserver, iSource.mCenter, r_0, iTime);
	SHOW_VAR6(wr_0, a, d, oTransport.mT, r, r.length());
	assert(0);
}
	oTransport.mR0=r/oTransport.mRAbs;
	oTransport.mMr=-mFreestream.scalar(oTransport.mR0)/mC;
	oTransport.mDoppler=std::abs(1-(iSource.mV+mFlightspeed-mFreestream).scalar(oTransport.mR0)/mC);
//SHOW_VAR6(iTime, oTransport.mT, oTransport.mRAbs, oTransport.mR0, oTransport.mDoppler, oTransport.mMr);
// std::cout << iTime << " " << oTransport.mT << " " << oTransport.mRAbs << " " << oTransport.mMr << std::endl;*/
}
#define sqr(x) (x*x)

template <class TOPO>
AcousticLineSources LineMesh<TOPO>::observe(topology_t const & iCell,
		Position const & iObserver, double iTime) {
	MeshBase::Transport trans;
	this->transport(iCell, iObserver, iTime, trans);
	AcousticLineSources source;
	source.mArea = iCell.mSpace;
	double force=-(iCell.mForce*trans.mR0)/source.mArea;
	source.mF=force/(sqr(trans.mRAbs)*sqr(1-trans.mMr));
	source.mFd=force/(trans.mRAbs*sqr(1-trans.mMr));
	source.mTime=trans.mT;
	source.mDoppler=trans.mDoppler;
	source.mMr=trans.mMr;
// SHOW_VAR4(source, force, source.mFd, source.mF);
	return source;
}

template <class TOPO>
AcousticLineSources LineMesh<TOPO>::getLineSource(int iCell,
		Position const & iObserver, double iTime) {
	return observe(mCells[iCell], iObserver, iTime);
}

template <class TOPO>
void LineMesh<TOPO>::getMechanics(int iCell, double * oSpace, Vector * oX, Vector * oF, Vector * oV) const {
	topology_t const & cell(mCells[iCell]);
	*oSpace=cell.mSpace;
	*oX=cell.mCenter;
	*oF=cell.mForce;
	*oV=cell.mV;
// SHOW_VAR5(iCell, *oSpace, *oX, *oF, *oV);
}

template <class TOPO>
AcousticSurfaceSources SurfaceMesh<TOPO>::observe(topology_t const & iCell,
		Position const & iObserver, double iTime) {
	assert(false);
	AcousticSurfaceSources source;
	/*
	MeshBase::Transport trans;
	this->transport(iCell, iObserver, iTime, trans);
	double v_n = (iCell.mV+mFlightspeed-mFreestream)*(iCell.mNormal);	// surface normal velocity
	double u_n = (iCell.mU+mFlightspeed-mFreestream)*(iCell.mNormal);	// fluid normal velocity
	double u_r = (iCell.mU+mFlightspeed-mFreestream)*(trans.mR0);		// fluid velocity towards observer
	double n_r = iCell.mNormal*(trans.mR0);				// normal component towards observer
	double w_n=-mFreestream*(iCell.mNormal);
	double w_u=-mFreestream*((iCell.mU));
	double Ma2=mFreestream.norm2()/sqr(mC);
	source.mArea = iCell.mSpace;
	double monopole=mC*(mRho0*v_n + iCell.mR*(u_n-v_n));
	double dipole=(trans.mPs*n_r + iCell.mR*u_r*(u_n-v_n));
	source.mM=monopole/sqr(trans.mRAbs*(1-trans.mMr))*(trans.mMr-Ma2);
	source.mD=dipole/sqr(trans.mRAbs*(1-trans.mMr))*(1-Ma2)
		-(trans.mPs*w_n+iCell.mR*w_u*(u_n-v_n))/(mC*(1-trans.mMr)*sqr(trans.mRAbs));
	source.mMd=monopole/(trans.mRAbs*(1-trans.mMr));
	source.mDd=dipole/(trans.mRAbs*(1-trans.mMr));
	if (AcousticSourcesBase::mVelocity) {
		source.mVd=monopole/trans.mRAbs*trans.mR0;
		source.mV=monopole/sqr(trans.mRAbs)*trans.mR0;;
		Vector L(trans.mPs*iCell.mNormal+iCell.mR*iCell.mU*(u_n-v_n));
		source.mVd+=dipole/trans.mRAbs*trans.mR0;
		source.mV+=(3*dipole*trans.mR0-L)/sqr(trans.mRAbs);
		source.mVi+=(3*dipole*trans.mR0-L)/(sqr(trans.mRAbs)*trans.mRAbs);
	}
//SHOW_VAR5(v_n, u_n, n_r, monopole, dipole);
//SHOW_VAR4(source.mM, source.mMd, source.mD, source.mDd);
// std::cout << iTime << " " << trans.mDoppler << " " << trans.mMr << " " << trans.mT << " " << trans.mRAbs << " " << monopole << " " << dipole
// 		<< " " << source.mM << " " << source.mMd << " " << source.mD << " " << source.mDd << " " << u_r << " " << n_r << " " << std::endl;
	source.mTime = trans.mT;
	source.mDoppler = trans.mDoppler;
	source.mMr=trans.mMr;
//SHOW_VAR9(iTime, monopole, dipole, source.mMd, source.mDd, source.mTime, source.mDoppler, trans.mRAbs, trans.mMr);*/
	return source;
}

template <class TOPO>
AcousticSurfaceSources SurfaceMesh<TOPO>::getSurfaceSource(int iCell,
		Position const & iObserver, double iTime) {
	return observe(mCells[iCell], iObserver, iTime);
}

template <class TOPO>
void SurfaceMesh<TOPO>::getMechanics(int iCell, double * oSpace, Vector * oX, Vector * oF, Vector * oV) const {
	topology_t const & cell(mCells[iCell]);
	*oSpace=cell.mSpace;
	*oX=cell.mCenter;
	*oF= cell.mNormal * cell.mSpace*(mP0-cell.mP);
	*oV=cell.mV;
}

template <class TOPO>
AcousticVolumeSources VolumeMesh<TOPO>::observe(topology_t const & iCell,
		Position const & iObserver, double iTime) {
	assert(false);
	AcousticVolumeSources source;
	/*
	MeshBase::Transport trans;
	this->transport(iCell, iObserver, iTime, trans);
	IAG::Matrix<double,3> T(0); // Lighthill's tensor
	for (int i=0; i<3; i++) {
		for (int j=0; j<3; j++) {
			T(i,j) += iCell.mR * iCell.mU[i] * iCell.mU[j];
		}
		T(i,i) += trans.mPs - mC*mC*(iCell.mR-mRho0);
	}
	double contraction=trans.mR0.scalar(T*trans.mR0);
	source.mArea = iCell.mSpace;
	double quadrupole=(3*contraction-T.trace());
	source.mQdd=contraction/(sqr(1-trans.mMr)*trans.mRAbs);
	source.mQd=quadrupole/sqr(trans.mRAbs);
	source.mQ=source.mQd/trans.mRAbs;
	source.mQd/=1-trans.mMr;
	source.mTime = trans.mT;
	source.mDoppler = trans.mDoppler;*/
	return source;
}

template <class TOPO>
void VolumeMesh<TOPO>::getMechanics(int iCell, double * oSpace, Vector * oX, Vector * oF, Vector * oV) const {
	topology_t const & cell(mCells[iCell]);
	*oSpace=cell.mSpace;
	*oX=cell.mCenter;
	*oF=0;
	*oV=cell.mV;
}

template <class TOPO>
AcousticVolumeSources VolumeMesh<TOPO>::getVolumeSource(int iCell,
		Position const & iObserver, double iTime) {
	return observe(mCells[iCell], iObserver, iTime);
}

// explicit instantiations to keep the member definitions out of headers
template
class LineMesh<LineTopo>;
template
class SurfaceMesh<TriangleTopo>;
template
class SurfaceMesh<QuadrangleTopo>;
template
class VolumeMesh<TetraederTopo>;
template
class VolumeMesh<HexaederTopo>;


#undef sqr
// ----------------------------------------
// ------------- MeshPts------------------
// ----------------------------------------
MeshPts::MeshPts(int iNumberOfPoints, Position const * iPoints)
	: mNumPoints(iNumberOfPoints), mPoints(iPoints) {}

MeshPts::~MeshPts() {
	delete[] mPoints;
}
