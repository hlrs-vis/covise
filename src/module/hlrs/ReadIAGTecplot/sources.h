/***************************************************************************
                          sources.cpp  -  description
                             -------------------
    begin                : Wed Jul 26 2006
	copyright			: (C) 2006-2014 IAG, University of Stuttgart
	email				: acco@iag.uni-stuttgart.de
 ***************************************************************************/

#ifndef SOURCES_H
#define SOURCES_H

#include <coord.h>

#include <cassert>

typedef Coordinate Position;
typedef Coordinate Vector;

#define INTERPOLATE(mD) mD=iFrom.mD*(1-iAt)+iTo.mD*iAt
#define ADD(mD) mD+=f*rhs.mD

struct AcousticSourcesBase {
	double mTime;
	double mDoppler, mMr;
	double mArea;
	void interpolate(AcousticSourcesBase const & iFrom, AcousticSourcesBase const & iTo, double iAt);

	static void setReferenceState(double iRho0, double iP0, double iC, Vector const & iFreestream, Vector const & iFlightspeed);
	static double mRho0, mP0, mC;
	static Vector mFreestream, mFlightspeed;
	static bool mVelocity;
};

inline void AcousticSourcesBase::setReferenceState(double iRho0, double iP0, double iC, Vector const & iFreestream, Vector const & iFlightspeed) {
	mRho0=iRho0;
	mP0=iP0;
	mC=iC;
	mFreestream=iFreestream;
	mFlightspeed=iFlightspeed;
}

inline void AcousticSourcesBase::interpolate(AcousticSourcesBase const & iFrom, AcousticSourcesBase const & iTo, double iAt) {
	static double const EPS=1e-8;
	// BEWARE: mDoppler has to be interpolated backwards, to approximate reciprocal interpolation
	mDoppler=iTo.mDoppler*(1-iAt)+iFrom.mDoppler*iAt;
}

inline std::ostream & operator<<(std::ostream & out, AcousticSourcesBase const & iSrc) {
	return out << iSrc.mTime << '/' << iSrc.mDoppler << '/' << iSrc.mArea;
}

struct VelocitySources {
	Vector mVi, mV, mVd;
	VelocitySources() : mVi(0), mV(0), mVd(0) {}
	void interpolate(VelocitySources const & iFrom, VelocitySources const & iTo, double iAt);
	void add(VelocitySources const & rhs, double f);
};
/*
inline void VelocitySources::interpolate(VelocitySources const & iFrom, VelocitySources const & iTo, double iAt) {
	INTERPOLATE(mVi);
	INTERPOLATE(mV);
	INTERPOLATE(mVd);
}*/
/*
inline void VelocitySources::add(VelocitySources const & rhs, double f) {
	ADD(mVi);
	ADD(mV);
	ADD(mVd);
}*/

inline std::ostream & operator<<(std::ostream & out, VelocitySources const & iSrc) {
	return out << iSrc.mVi << '/' << iSrc.mV << '/' << iSrc.mVd;
}

struct AcousticVelocitySources : AcousticSourcesBase, VelocitySources {
	typedef VelocitySources sources_t;
	void interpolate(AcousticVelocitySources const & iFrom, AcousticVelocitySources const & iTo, double iAt);
};

inline void AcousticVelocitySources::interpolate(AcousticVelocitySources const & iFrom, AcousticVelocitySources const & iTo, double iAt) {
	AcousticSourcesBase::interpolate(iFrom, iTo, iAt);
	VelocitySources::interpolate(iFrom, iTo, iAt);
}

inline std::ostream & operator<<(std::ostream & out, AcousticVelocitySources const & iSrc) {
	return out << static_cast<AcousticSourcesBase const &>(iSrc) << ';' << static_cast<VelocitySources const &>(iSrc);
}

struct LineSources {
	double mFd, mF;
	LineSources() : mFd(0), mF(0) {}
	void interpolate(LineSources const & iFrom, LineSources const & iTo, double iAt);
	void add(LineSources const & rhs, double f);
};

inline void LineSources::interpolate(LineSources const & iFrom, LineSources const & iTo, double iAt) {
	INTERPOLATE(mFd);
	INTERPOLATE(mF);
}

inline void LineSources::add(LineSources const & rhs, double f) {
	ADD(mFd);
	ADD(mF);
}

inline std::ostream & operator<<(std::ostream & out, LineSources const & iSrc) {
	return out << iSrc.mFd << '/' << iSrc.mF;
}

struct AcousticLineSources : AcousticSourcesBase, LineSources {
	typedef LineSources sources_t;
	void interpolate(AcousticLineSources const & iFrom, AcousticLineSources const & iTo, double iAt);
};

inline void AcousticLineSources::interpolate(AcousticLineSources const & iFrom, AcousticLineSources const & iTo, double iAt) {
	AcousticSourcesBase::interpolate(iFrom, iTo, iAt);
	LineSources::interpolate(iFrom, iTo, iAt);
}

inline std::ostream & operator<<(std::ostream & out, AcousticLineSources const & iSrc) {
	return out << static_cast<AcousticSourcesBase const &>(iSrc) << ';' << static_cast<LineSources const &>(iSrc);
}

struct SurfaceSources : VelocitySources {
	double mMd, mM, mDd, mD;
	SurfaceSources() : mMd(0), mM(0), mDd(0), mD(0) {}
	void interpolate(SurfaceSources const & iFrom, SurfaceSources const & iTo, double iAt);
	void add(SurfaceSources const & rhs, double f);
};

inline void SurfaceSources::interpolate(SurfaceSources const & iFrom, SurfaceSources const & iTo, double iAt) {
	VelocitySources::interpolate(iFrom, iTo, iAt);
	INTERPOLATE(mMd);
	INTERPOLATE(mM);
	INTERPOLATE(mDd);
	INTERPOLATE(mD);
}

inline void SurfaceSources::add(SurfaceSources const & rhs, double f) {
	VelocitySources::add(rhs, f);
	ADD(mMd);
	ADD(mM);
	ADD(mDd);
	ADD(mD);
}

inline std::ostream & operator<<(std::ostream & out, SurfaceSources const & iSrc) {
	out << static_cast<VelocitySources const &>(iSrc) << '/';
	return out << iSrc.mMd << '/' << iSrc.mM << '/' << iSrc.mDd << '/' << iSrc.mD;
}

struct AcousticSurfaceSources : AcousticSourcesBase, SurfaceSources {
	typedef SurfaceSources sources_t;
	void interpolate(AcousticSurfaceSources const & iFrom, AcousticSurfaceSources const & iTo, double iAt);
};

inline void AcousticSurfaceSources::interpolate(AcousticSurfaceSources const & iFrom, AcousticSurfaceSources const & iTo, double iAt) {
	AcousticSourcesBase::interpolate(iFrom, iTo, iAt);
	SurfaceSources::interpolate(iFrom, iTo, iAt);
}

inline std::ostream & operator<<(std::ostream & out, AcousticSurfaceSources const & iSrc) {
	return out << static_cast<AcousticSourcesBase const &>(iSrc) << ';' << static_cast<SurfaceSources const &>(iSrc);
}

struct VolumeSources {
	double mQdd, mQd, mQ;
	VolumeSources() : mQdd(0), mQd(0), mQ(0) {}
	void interpolate(VolumeSources const & iFrom, VolumeSources const & iTo, double iAt);
	void add(VolumeSources const & rhs, double f);
};

inline void VolumeSources::interpolate(VolumeSources const & iFrom, VolumeSources const & iTo, double iAt) {
	INTERPOLATE(mQdd);
	INTERPOLATE(mQd);
	INTERPOLATE(mQ);
}

inline void VolumeSources::add(VolumeSources const & rhs, double f) {
	ADD(mQdd);
	ADD(mQd);
	ADD(mQ);
}

inline std::ostream & operator<<(std::ostream & out, VolumeSources const & iSrc) {
	return out << iSrc.mQdd << '/' << iSrc.mQd << '/' << iSrc.mQ;
}

struct  AcousticVolumeSources : AcousticSourcesBase, VolumeSources {
	typedef VolumeSources sources_t;
	void interpolate(AcousticVolumeSources const & iFrom, AcousticVolumeSources const & iTo, double iAt);
};

inline void AcousticVolumeSources::interpolate(AcousticVolumeSources const & iFrom, AcousticVolumeSources const & iTo, double iAt) {
	AcousticSourcesBase::interpolate(iFrom, iTo, iAt);
	VolumeSources::interpolate(iFrom, iTo, iAt);
}

inline std::ostream & operator<<(std::ostream & out, AcousticVolumeSources const & iSrc) {
	return out << static_cast<AcousticSourcesBase const &>(iSrc) << ';' << static_cast<VolumeSources const &>(iSrc);
}

struct ObserverSlot : LineSources, SurfaceSources, VolumeSources {
	ObserverSlot() {}
	double P() const { return mM+mD+mF+mQ; }
	Vector V() const { return mV; }
	void clear();
	using LineSources::add;
	using SurfaceSources::add;
	using VolumeSources::add;
};

inline bool isINF(ObserverSlot const & x) {
	return std::isinf(x.mVi.X()) || std::isinf(x.mV.X()) || std::isinf(x.mVd.X())
		|| std::isinf(x.mQdd) || std::isinf(x.mQd) || std::isinf(x.mQ)
		|| std::isinf(x.mDd) || std::isinf(x.mD) || std::isinf(x.mMd)
		|| std::isinf(x.mM);
}

inline void ObserverSlot::clear() {
	mQ=mD=mM=mF=0;
	mV=0;
}

struct ObserverLine : std::vector<ObserverSlot> {
	int mMinValidSlot, mMaxValidSlot;
	void resize(int iSize);
};

inline void ObserverLine::resize(int iSize) {
	std::vector<ObserverSlot>::resize(iSize);
	mMinValidSlot=0; mMaxValidSlot=iSize-1;
}

#endif
