/***************************************************************************
                          tecplotfile.cpp  -  description
                             -------------------
    begin                : Wed Jul 26 2006
	copyright			: (C) 2006-2014 IAG, University of Stuttgart
	email				: acco@iag.uni-stuttgart.de
 ***************************************************************************/

#include "tecplotfile.h"
//#include <Include/parallel.h>
#include <topo.h>

//#include <iag/debug.h>
//#include <iag/problem.h>
//#include <iag/stringutil.h>
//#include <iag/tool.h>

#include <fstream>
#include <iostream>

#include <cassert>
#include <cstring>
#include <cstdio>
#include <string>


#if defined(HAVE_STDINT_H)
#include <stdint.h>
typedef int32_t TEC_INT32;
#elif defined(_MSC_VER)
typedef __int32 TEC_INT32;
#else
typedef int TEC_INT32; // hope the best...
#endif
typedef float TEC_FLOAT32;
typedef double TEC_FLOAT64;


enum FileDataSet {
	UNKNOWN,
	FLOWER_SURFACE, FLOWER_PLANFILE, FLOWER_NOVS, FLOWER_VS, FLOWER_ACCO, LIFT_LINE, SURFACE, IAGCOUPLE_ACCOLOAD,
#ifdef AHD
	CAMRAD_LINE, CAMRAD_SURFACE, 
#endif
#ifdef MARENCO
	FLIGHTLAB_LINE, FLIGHTLAB_SURFACE,
#endif
};

struct TecplotFile::Zone {
public:
	struct RindSpec { int mMinI, mMaxI, mMinJ, mMaxJ, mMinK, mMaxK; };
	static void parseRindSpec(std::string const & iSpec, RindSpec & oRind);
	Zone();
	std::string getName() const { return mName; }
	void ReadHeader(TecplotFile::Impl * pimpl, int iNumVar);
	void ReadDataHeader(TecplotFile::Impl * pimpl, int iNumVar);
	MeshBase * ReadData(TecplotFile::Impl * pimpl, int iNumVar, RindSpec const & iRind);
	MeshPts * ReadInnerPtsData(TecplotFile::Impl * pimpl, int iNumVar);
	enum DataFormat { FLOAT=1, DOUBLE=2, LONG=3, SHORT=4, BYTE=5, BIT=6 };
	enum ZoneType { ORDERED=0, FELINESEG=1, FETRIANGLE=2, FEQUADRILATERAL=3,
		FETETRAHEDRON=4, FEBRICK=5, FEPOLYGON=6, FEPOLYHEDRON=7 };
	enum PackingType { BLOCK=0, POINT=1 };
private:
	std::string	mName;
	ZoneType mType;
	PackingType mDataPacking;
	int mVarLocation;
	int mNeighborConnections;
	int mIMax, mJMax, mKMax;
	int mNumPts, mNumElements, mICellDim, mJCellDim, mKCellDim;
	std::vector<DataFormat> mVarDataFormat;
};

struct TecplotFile::Impl {
	static double const ZONEMARKER;
	static double const GEOMETRYMARKER;
	static double const TEXTMARKER;
	static double const LABELMARKER;
	static double const USERMARKER;
	static double const DATAAUXMARKER;
	static double const VARAUXMARKER;
	static double const EOHMARKER;

	int mVersion;
	FileDataSet mDataSet;
	bool mOtherEndian;
	std::ifstream mStream;

	bool open(std::string const & iFileName);

	void skip32(int iNum=1);
	void skip64(int iNum=1);
	int fetchInt32();
	float fetchFloat();
	double fetchDouble();
	double fetchReal(TecplotFile::Zone::DataFormat iFormat);
	std::string fetchString();

	template<typename T>
	static T byteswap(T iVal);
};

double const TecplotFile::Impl::ZONEMARKER=299;
double const TecplotFile::Impl::GEOMETRYMARKER=399;
double const TecplotFile::Impl::TEXTMARKER=499;
double const TecplotFile::Impl::LABELMARKER=599;
double const TecplotFile::Impl::USERMARKER=699;
double const TecplotFile::Impl::DATAAUXMARKER=799;
double const TecplotFile::Impl::VARAUXMARKER=899;
double const TecplotFile::Impl::EOHMARKER=357;

template<typename T>
T TecplotFile::Impl::byteswap(T iVal) {
	union {
		char asChar[sizeof(T)];
		T asT;
	} in, out;
	in.asT=iVal;
	for (unsigned i=0;i<sizeof(T);i++) {
		out.asChar[i]=in.asChar[sizeof(T)-1-i];
	}
	return out.asT;
}

bool TecplotFile::Impl::open(std::string const & iFileName) {
	if (mStream.is_open()) mStream.close();
	mStream.open(iFileName.c_str(), std::ios::binary);
	if (!mStream.good()) {
		
			std::cerr << "Failed to open " << iFileName << " successfully.\n";
			std::cerr << "stream state: " << mStream.rdstate() << ", fail=" << mStream.fail()
				<< ", bad=" << mStream.bad() << ", eof=" << mStream.eof() << std::endl;
			std::perror("opening");
		
		return false;
	}
	char versionString[8];
	mStream.read(versionString, sizeof(versionString));
	if (std::strncmp(versionString, "#!TDV", 5)) {
		std::cerr << "No tecplot binary file.\n";
		return false;
	}
	mVersion=0;
	if (isdigit(versionString[5]))
		mVersion=versionString[5]-'0';
	if (isdigit(versionString[6]))
		mVersion=mVersion*10+versionString[6]-'0';
	if (isdigit(versionString[7]))
		mVersion=mVersion*10+versionString[7]-'0';
		std::cerr << " Version: "<< versionString[5] << versionString[6] << "." << versionString[7]
			<< " (" << mVersion << ")\n";
		if (mVersion>112) {
			std::cerr << "WARNING: Tecplot version newer than 360 2009 detected, hoping for compatibility...\n";
		}
	if (mVersion==112) {
// 		std::cout << "  Tecplot version 360 2009 detected\n";
	}
	else if (mVersion==111) {
// 		std::cout << "  Tecplot version 360 2008 detected\n";
	}
	else if (mVersion==107) {
// 		std::cout << "  Tecplot version 360 detected\n";
	}
	else if (mVersion==102) {
// 		std::cout << "  Tecplot version 10 detected\n";
	}
	else if (mVersion==75) {
// 		std::cout << "  Tecplot version 9.2 detected\n";
	}
	else {
		 std::cout << "WARNING: Unknown Tecplot version detected, hoping for compatibility...\n";
	}
// integer with value "1" to detect little or big endian -----------
	mOtherEndian = false; // ensure endianTest is fetched unchanged
	int endianTest=fetchInt32();
	if (endianTest!=1) {
		if (endianTest==16777216) {
			mOtherEndian = true;
		}
		else {
			std::cerr << "Illegal endianess test value.\n";
			return false;
		}
	}
	return true;
}

#define FETCH(VAR) do { \
	mStream.read(reinterpret_cast<char *>(&(VAR)), sizeof(VAR)); \
	if (mOtherEndian) (VAR) = byteswap(VAR); \
} while (false)

void TecplotFile::Impl::skip32(int iNum) {
	mStream.seekg(4*iNum, std::istream::cur);
}

void TecplotFile::Impl::skip64(int iNum) {
	mStream.seekg(8*iNum, std::istream::cur);
}

int TecplotFile::Impl::fetchInt32() {
	TEC_INT32 v;
	FETCH(v);
	return v; 
}

float TecplotFile::Impl::fetchFloat() {
	TEC_FLOAT32 v;
	FETCH(v);
	return v; 
}

double TecplotFile::Impl::fetchDouble() {
	TEC_FLOAT64 v;
	FETCH(v);
	return v; 
}

double TecplotFile::Impl::fetchReal(TecplotFile::Zone::DataFormat iFormat) {
	if (iFormat==TecplotFile::Zone::FLOAT) return fetchFloat();
	assert(iFormat==TecplotFile::Zone::DOUBLE);
	return fetchDouble();
}

std::string TecplotFile::Impl::fetchString() {
	std::string result;
	int nTemp=fetchInt32();
	while (nTemp!=0) {
		result += nTemp;
		nTemp=fetchInt32();
	}
	return result;
}

TecplotFile::Zone::Zone() : mType(ORDERED), mDataPacking(BLOCK), mVarLocation(0), mNeighborConnections(0),
	mIMax(0), mJMax(0), mKMax(0), mNumPts(0), mNumElements(0), mICellDim(0), mJCellDim(0), mKCellDim(0),
	mVarDataFormat(0) {}

void TecplotFile::Zone::ReadHeader(TecplotFile::Impl * pimpl, int iNumVar) {
	mName=pimpl->fetchString();
	if (pimpl->mVersion>=107) {
		//int parentZone=
		pimpl->fetchInt32();
		//int strandId=
		pimpl->fetchInt32();
		//double solutionTime=
		pimpl->fetchDouble();;
	}
	//int zoneColor=
	pimpl->fetchInt32();
	int type=pimpl->fetchInt32();
	if (type<ORDERED || type>FEPOLYHEDRON) {
		std::cerr << "Illegal zone type " << type <<"! \n";
		throw "Illegal zone type";
	}
	mType=static_cast<ZoneType>(type);
	// datapacking is always Block in 360.2009
	if (pimpl->mVersion>=112)
		mDataPacking=BLOCK;
	else mDataPacking=static_cast<PackingType>(pimpl->fetchInt32());
	// var location: 0 = Don't specify, all data is located at the nodes.  1 = Specify
	mVarLocation=pimpl->fetchInt32();
	if (mVarLocation!=0) {
		std::cerr << "Warning: Only unspecified ORDERED var location supported! \n"; 
    for (int i=0; i<iNumVar; i++) {
      pimpl->fetchInt32();
    }
//		throw "Only unspecified ORDERED var location supported!";
	}
	if (pimpl->mVersion>=111) {
		//int rawLocal1to1FaceNeighbours=
		pimpl->fetchInt32();
	}
	mNeighborConnections=pimpl->fetchInt32();
	if (mNeighborConnections!=0) {
		std::cerr << "User defined face neighbor mode (" 
			<< mNeighborConnections << ") not supported! \n"; 
		throw "User defined face neighbor mode not supported!";
	}
	if (mType==ORDERED) {
		mIMax=pimpl->fetchInt32();
		mJMax=pimpl->fetchInt32();
		mKMax=pimpl->fetchInt32();
	}
	else if (mType!=FEPOLYGON && mType !=FEPOLYHEDRON) {
		mNumPts=pimpl->fetchInt32();
		mNumElements=pimpl->fetchInt32();
		mICellDim=pimpl->fetchInt32();
		mJCellDim=pimpl->fetchInt32();
		mKCellDim=pimpl->fetchInt32();
	}
	else {
		std::cerr << "New zone types FEPOLYGON or FEPOLYHEDRON not supported! \n";
		throw "New zone types FEPOLYGON or FEPOLYHEDRON not supported!";
	}
	int auxiliaryNameValuePairs=pimpl->fetchInt32();
	if (auxiliaryNameValuePairs!=0) {
		std::cerr << "Auxiliary name/value pair: not supported! \n"; 
		throw "Auxiliary name/value pair: not supported!";
	}
}

void TecplotFile::Zone::ReadDataHeader(TecplotFile::Impl * pimpl, int iNumVar) {
	mVarDataFormat.resize(iNumVar);
	for (int i=0; i<iNumVar; i++) {
		int format=pimpl->fetchInt32();
		if (format<FLOAT || format>BIT) {
			std::cerr << "Illegal variable format " << format <<"! \n";
			throw "Illegal variable format";
		}
		mVarDataFormat[i]=static_cast<DataFormat>(format);
		if (mVarDataFormat[i]!=FLOAT && mVarDataFormat[i]!=DOUBLE) {
			std::cerr << "Unupported DataFormat " << format << "! \n";
			throw "Unupported variable format";
		}
	}
	if (pimpl->mVersion>=107) {
		int passiveVars=pimpl->fetchInt32();
		if (passiveVars) {
			std::cerr << "Passive Variables: not supported! \n";
			assert(0);
		}
	}
	int varSharing=pimpl->fetchInt32();
	if (varSharing!=0) {
		std::cerr << "Variable sharing: not supported! \n";
		assert(0);
	}
	int shareConnectivityList=pimpl->fetchInt32();
	if (shareConnectivityList!=-1) {
		std::cerr << "Share connectivity list: not supported! \n";
		assert(0);
	}
	if (pimpl->mVersion>=107) {
		for (int j=0; j<iNumVar; j++) {
			//double minVal=
			pimpl->fetchDouble();
			//double maxVal=
			pimpl->fetchDouble();
		}
	}
}

MeshPts * TecplotFile::Zone::ReadInnerPtsData(TecplotFile::Impl * pimpl, int iNumVar) {
	ReadDataHeader(pimpl, iNumVar);
	MeshPts * mesh=0;
	if (mType==ORDERED) {
		// zone data
		mNumPts=mIMax*mJMax*mKMax;
		Position * points(new PointState[mNumPts]);
		for (int p=0; p<mNumPts*iNumVar; ++p) {
			int i, j;
			if (mDataPacking==BLOCK) j=p/mNumPts, i=p%mNumPts;
			else if (mDataPacking==POINT) j=p%iNumVar, i=p/iNumVar;
			else throw "Unsupported DataPacking";
			if (j<3) points[i][j]=pimpl->fetchFloat();
			else pimpl->fetchFloat(); // dummy read, as only coordinates are supported
		}
		mesh = new MeshPts(mNumPts, points);
	}
	else {
		std::cerr << "Unsupported ZoneType " << mType << std::endl;
		throw "Unsupported ZoneType";
	}
	return mesh;
}

MeshBase * TecplotFile::Zone::ReadData(TecplotFile::Impl * pimpl, int iNumVar, RindSpec const & iRind) {
	ReadDataHeader(pimpl, iNumVar);
	MeshBase * mesh=0;
	RindSpec rind(iRind);
	if (rind.mMinI) --rind.mMinI;
	if (rind.mMinJ) --rind.mMinJ;
	if (rind.mMinK) --rind.mMinK;
	assert(rind.mMinI>=0 && rind.mMinJ>=0 && rind.mMinK>=0);
	{
		if (rind.mMaxI>mIMax) std::cerr << "Rind specification error: rindMaxI (" << rind.mMaxI
			<< ") > realMaxI (" << mIMax << "), ignored\n", rind.mMaxI=mIMax;
		if (rind.mMaxJ>mJMax) std::cerr << "Rind specification error: rindMaxJ (" << rind.mMaxJ
			<< ") > realMaxJ (" << mJMax << "), ignored\n", rind.mMaxJ=mJMax;
		if (rind.mMaxK>mKMax) std::cerr << "Rind specification error: rindMaxK (" << rind.mMaxK
			<< ") > realMaxK (" << mKMax << "), ignored\n", rind.mMaxK=mKMax;
	} 
	if (!rind.mMaxI) rind.mMaxI=mIMax;
	else if (rind.mMaxI<0) rind.mMaxI=mIMax+rind.mMaxI;
	if (!rind.mMaxJ) rind.mMaxJ=mJMax;
	else if (rind.mMaxJ<0) rind.mMaxJ=mJMax+rind.mMaxJ;
	if (!rind.mMaxK) rind.mMaxK=mKMax;
	else if (rind.mMaxK<0) rind.mMaxK=mKMax+rind.mMaxK;
	if (mType==ORDERED) {
		// now that data is read, adjust grid sizes to the cut ones
		int readI=mIMax, readJ=mJMax, readK=mKMax;
		int readNumPts=readI*readJ*readK;
		mIMax=rind.mMaxI-rind.mMinI;
		mJMax=rind.mMaxJ-rind.mMinJ;
		mKMax=rind.mMaxK-rind.mMinK;
		mNumPts=mIMax*mJMax*mKMax;
		bool is1D=mJMax==1 && mKMax==1;
#ifdef MARENCO
		PointState * points(new PointState[is1D && pimpl->mDataSet!=LIFT_LINE && pimpl->mDataSet!=FLIGHTLAB_LINE ? mNumPts*2 : mNumPts]);
#elif defined(AHD)
		PointState * points(new PointState[is1D && pimpl->mDataSet!=LIFT_LINE && pimpl->mDataSet!=CAMRAD_LINE ? mNumPts*2 : mNumPts]);
#else
		PointState * points(new PointState[is1D && pimpl->mDataSet!=LIFT_LINE ? mNumPts*2 : mNumPts]);
#endif
		for (int v=0, readV=0; readV<readNumPts*iNumVar; ++readV) {
			int readP, readM;
			if (mDataPacking==BLOCK) readM=readV/readNumPts, readP=readV%readNumPts;
			else if (mDataPacking==POINT) readM=readV%iNumVar, readP=readV/iNumVar;
			else throw "Unsupported DataPacking";
			int k=readP/(readI*readJ), j=readP%(readI*readJ);
			int i=j%readI;
			j/=readI;
			// read, but otherwise skip rind data
			if (i<rind.mMinI || i>=rind.mMaxI || j<rind.mMinJ || j>=rind.mMaxJ || k<rind.mMinK || k>=rind.mMaxK) {
				pimpl->fetchReal(mVarDataFormat[readM]);
				continue;
			}
			int p=0, m=0;
			if (mDataPacking==BLOCK) m=v/mNumPts, p=v%mNumPts;
			else if (mDataPacking==POINT) m=v%iNumVar, p=v/iNumVar;
			else throw "Unsupported DataPacking";
			assert(m==readM);
			// clear surface velocity, if not to be read
			if ((iNumVar<11 || pimpl->mDataSet==FLOWER_SURFACE || pimpl->mDataSet==FLOWER_PLANFILE || iNumVar==13 || iNumVar==14 ) && m==0) points[p].mV=0;
			double value=pimpl->fetchReal(mVarDataFormat[readM]);
			switch (pimpl->mDataSet) {
				case FLOWER_SURFACE:
				case FLOWER_PLANFILE:
					switch (m) {
						case 0: points[p][0]=value; break;
						case 1: points[p][1]=value; break;
						case 2: points[p][2]=value; break;
						case 3: points[p].mR=value; break;
						case 4: points[p].mU.X()=value; break;
						case 5: points[p].mU.Y()=value; break;
						case 6: points[p].mU.Z()=value; break;
						case 7: break;
						case 8: points[p].mP=value; break; // cp, in fact, has to be rescaled later on
						case 18: assert(pimpl->mDataSet==FLOWER_SURFACE); points[p].mN[0]=value; break;
						case 19: points[p].mN[1]=value; break;
						case 20: points[p].mN[2]=value; break;
						default: break;/*assert(0);*/ // just skip rest, nothing more interesting
					}
					break;
				case FLOWER_NOVS:
				case FLOWER_ACCO:
					switch (m) {
						case 0: points[p][0]=value; break;
						case 1: points[p][1]=value; break;
						case 2: points[p][2]=value; break;
						case 3: points[p].mR=value; break;
						case 4: points[p].mU.X()=value; break;
						case 5: points[p].mU.Y()=value; break;
						case 6: points[p].mU.Z()=value; break;
						case 7: points[p].mP=value; break; // cp, in fact, has to be rescaled later on
						default: break;/*assert(0);*/ // just skip rest, nothing more interesting
					}
					break;
				case FLOWER_VS:
					switch (m) {
						case 0: points[p][0]=value; break;
						case 1: points[p][1]=value; break;
						case 2: points[p][2]=value; break;
						case 3: points[p].mR=value; break;
						case 4: points[p].mU.X()=value; break;
						case 5: points[p].mU.Y()=value; break;
						case 6: points[p].mU.Z()=value; break;
						case 7: points[p].mP=value; break;
						case 8: points[p].mV.X()=value; break;
						case 9: points[p].mV.Y()=value; break;
						case 10: points[p].mV.Z()=value; break;
						default: assert(0);
					}
					break;
				case LIFT_LINE:
					switch (m) {
						case 0: break; // psi
						case 1: points[p][0]=value; break; // segment center, not corner
						case 2: points[p][1]=value; break;
						case 3: points[p][2]=value; break;
						case 4: points[p].mR=value; break; // dr_ACP, not density
						case 5: points[p].mP=value; break; // clen_ACP, not pressure
						case 6: points[p].mU.Z()=value; break; // BEWARE: dT/dr
						case 7: points[p].mV.Z()=value; break; // BEWARE: dD/dr
						case 8: points[p].mU.X()=points[p].mR*points[p].mU.Z()*value; break; // thrust direction, not fluid velocity
						case 9: points[p].mU.Y()=points[p].mR*points[p].mU.Z()*value; break; // using just read dr_ACP and dT/dr
						case 10: points[p].mU.Z()=points[p].mR*points[p].mU.Z()*value; break;
						case 11: points[p].mV.X()=points[p].mR*points[p].mV.Z()*value; break; // torque direction, not grid velocity
						case 12: points[p].mV.Y()=points[p].mR*points[p].mV.Z()*value; break; // using just read dr_ACP and dD/dr
						case 13: points[p].mV.Z()=points[p].mR*points[p].mV.Z()*value; break;
						default: break;
					}
					break;
				case SURFACE:
					switch (m) {
						case 0: points[p][0]=value; break;
						case 1: points[p][1]=value; break;
						case 2: points[p][2]=value; break;
						default: break;
					}
					break;

				case IAGCOUPLE_ACCOLOAD:
					switch (m) {
						case 0: break;						// Time
						case 1: break;						// Azimuth
						case 2: points[p][0]=value; break;  // Segment center, not corner
						case 3: points[p][1]=value; break;
						case 4: points[p][2]=value; break;
						case 5: {							// Force X-dir
							points[p].mU.X()=value;		
							points[p].mV.X()=0;
							break;
						}
						case 6: {							// Force Y-dir
							points[p].mU.Y()=value;
							points[p].mV.Y()=0;
							break;
						}
						case 7: {							// Force Z-dir
							points[p].mU.Z()=value;
							points[p].mV.Z()=0;
							break;
						}
						case 8: {							// Section area
							points[p].mP=1;
							points[p].mR=value;
							break;
						}
						case 9: 							// Density
						default: break; // nothing more
					}
					break;

#ifdef AHD
				case CAMRAD_SURFACE:
					switch (m) {
						case 0: break; // time
						case 1: break; // azimuth
						case 2: points[p][0]=value; break; // quad center, not corner
						case 3: points[p][1]=value; break;
						case 4: points[p][2]=value; break;
						case 5: points[p].mN.X()=value; break;
						case 6: points[p].mN.Y()=value; break;
						case 7: points[p].mN.Z()=value; break;
						case 8: points[p].mP=value; break; // quad area, not pressure
						default: break; // nothing more interesting: psi_blade, r_ACP, x_ACP
					}
					break;
				case CAMRAD_LINE:
					switch (m) {
						case 0: points[p][0]=value; break; // segment center, not corner
						case 1: points[p][1]=value; break;
						case 2: points[p][2]=value; break;
//						case 3: points[p].mP=(value+9.89523+(3-12.6*points[p].length()/5.5))*M_PI/180; break; // alpha
						case 3: points[p].mP=(value)*M_PI/180; break; // alpha
						case 4: points[p].mR=value; break; // dT
						case 5: { points[p].mU.X()=points[p].mU.Y()=points[p].mV.Z()=0; // dD
							double thrust=points[p].mR*cos(points[p].mP)-value*sin(points[p].mP);
							double torque=points[p].mR*sin(points[p].mP)+value*cos(points[p].mP);
							points[p].mU.Z()=thrust;
							points[p].mV.X()=points[p].Y(); points[p].mV.Y()=-points[p].X(); 
							double scale=torque/points[p].length(); 
							points[p].mV.X()*=scale; points[p].mV.Y()*=scale; } break;
						case 6: break; // dRadial
						case 7: break; // radius
						case 8: points[p].mP=value; break; // chord, not pressure
						case 9: points[p].mR=value; break; // dr
						default: break; // nothing more
					}
					break;
#endif // AHD
#ifdef MARENCO
				case FLIGHTLAB_SURFACE:
					switch (m) {
						case 0: break; // time
						case 1: break; // azimuth
						case 2: points[p][0]=value; break; // quad center, not corner
						case 3: points[p][1]=value; break;
						case 4: points[p][2]=value; break;
						case 5: points[p].mN.X()=value; break; // quad normal, not fluid velocity
						case 6: points[p].mN.Y()=value; break;
						case 7: points[p].mN.Z()=value; break;
						case 8: points[p].mP=value; break; // quad area, not pressure
						default: break; // nothing more interesting: psi_blade, r_ACP, x_ACP
					}
					break;
				case FLIGHTLAB_LINE:
					switch (m) {
						case 0: break; // time
						case 1: break; // azimuth
						case 2: break; // psi_blade
						case 3: break; // r_ACP
						case 4: break; // x_ACP
						case 5: points[p][0]=value; break; // segment center, not corner
						case 6: points[p][1]=value; break;
						case 7: points[p][2]=value; break;
						case 8: points[p].mR=value; break; // dr_ACP, not density
						case 9: points[p].mP=value; break; // clen_ACP, not pressure
						case 10: points[p].mU.Z()=value; break; // BEWARE: dT/dr
						case 11: points[p].mV.Z()=value; break; // BEWARE: dD/dr
						case 12: points[p].mU.X()=points[p].mR*points[p].mU.Z()*value; break; // thrust direction, not fluid velocity
						case 13: points[p].mU.Y()=points[p].mR*points[p].mU.Z()*value; break; // using just read dr_ACP and dT/dr
						case 14: points[p].mU.Z()=points[p].mR*points[p].mU.Z()*value; break;
						case 15: points[p].mV.X()=points[p].mR*points[p].mV.Z()*value; break; // torque direction, not grid velocity
						case 16: points[p].mV.Y()=points[p].mR*points[p].mV.Z()*value; break; // using just read dr_ACP and dD/dr
						case 17: points[p].mV.Z()=points[p].mR*points[p].mV.Z()*value; break;
						default: break; // nothing more interesting: psi_blade, r_ACP, x_ACP
					}
					break;
#endif
				default:
					throw "Unsupported data format";
			}
			++v;
		}
#ifdef AHD
		if (pimpl->mDataSet==CAMRAD_LINE) {
			// lift and drag are differential line loads, so make them absolute
			for (int p=0; p!=mNumPts; ++p) {
				points[p].mU*=points[p].mR;
				points[p].mV*=points[p].mR;
			}
		}
#endif
		if (is1D) {
			// generic line reading for 2D cases
			mNumElements=mIMax-1;
			std::cerr << mNumElements << " " << mNumPts << std::endl;;
#ifdef MARENCO
			if (pimpl->mDataSet==FLIGHTLAB_LINE) {
				LineTopo * cells(new LineTopo[++mNumElements]);
				mesh = new LineMesh<LineTopo>(mNumElements, mNumPts, cells, points);
			}
			else
#endif
#ifdef AHD
			if (pimpl->mDataSet==CAMRAD_LINE) {
				LineTopo * cells(new LineTopo[++mNumElements]);
				mesh = new LineMesh<LineTopo>(mNumElements, mNumPts, cells, points);
			}
			else
#endif
			if (pimpl->mDataSet==LIFT_LINE) {
				LineTopo * cells(new LineTopo[++mNumElements]);
				mesh = new LineMesh<LineTopo>(mNumElements, mNumPts, cells, points);
			}
			else
			if (pimpl->mDataSet==IAGCOUPLE_ACCOLOAD) {
				LineTopo * cells(new LineTopo[++mNumElements]);
				mesh = new LineMesh<LineTopo>(mNumElements, mNumPts, cells, points);
			}
			else
			{
				QuadrangleTopo * cells(new QuadrangleTopo[mNumElements]);
				// duplicate points on another z plane
				for (int i=0; i<mNumPts; ++i) {
					points[i+mNumPts]=points[i];
					points[i].Z()=points[i].Z()-0.5;
					points[i+mNumPts].Z()=points[i].Z()+1;
				}
				int counter=0;
				for (int i=0; i<mIMax-1; ++i) {
					cells[counter].mNodes[0]=i;
					cells[counter].mNodes[1]=i+1;
					cells[counter].mNodes[2]=i+1+mNumPts;
					cells[counter].mNodes[3]=i  +mNumPts;
					counter++;
				}
				mNumPts*=2;
				mesh = new SurfaceMesh<QuadrangleTopo>(mNumElements, mNumPts, cells, points);
			}
		}
		else if ((mIMax==1)||(mJMax==1)||(mKMax==1)) {
			// generic quad topology creation
			int counter=0;
			int di1=mIMax==1 ? 0 : 1, dj1=1-di1;
			// depending on ordering, the first running index may be i or j
			int dk2=mKMax==1 ? 0 : 1, dj2=1-dk2;
			// the second running index is either j or k
			assert((mJMax==1 && dj1+dj2==0) || (mJMax!=1 && dj1+dj2==1));
			assert(mKMax==1 && "");
			mNumElements=(mIMax-di1)*(mJMax-dj1-dj2)*(mKMax-dk2);
#ifdef MARENCO
			if (pimpl->mDataSet==FLIGHTLAB_SURFACE) mNumElements=mNumPts;
#endif
			QuadrangleTopo * cells(new QuadrangleTopo[mNumElements]);
#ifdef MARENCO			
			if (pimpl->mDataSet!=FLIGHTLAB_SURFACE)
#endif
				for (int k=0; k<mKMax-dk2; ++k) {
					for (int j=0; j<mJMax-dj1-dj2; ++j) {
						for (int i=0; i<mIMax-di1; ++i) {
							cells[counter].mNodes[0]=i     + mIMax*(j         + mJMax*k);
							cells[counter].mNodes[1]=i+di1 + mIMax*(j+dj1     + mJMax*k);
							cells[counter].mNodes[2]=i+di1 + mIMax*(j+dj1+dj2 + mJMax*(k+dk2));
							cells[counter].mNodes[3]=i     + mIMax*(j+dj2     + mJMax*(k+dk2));
							counter++;
						}
					}
				}
			mesh = new SurfaceMesh<QuadrangleTopo>(mNumElements, mNumPts, cells, points);
		}
		else { // bricks
			mNumElements=(mIMax-1)*(mJMax-1)*(mKMax-1);
			HexaederTopo * cells(new HexaederTopo[mNumElements]);
			int counter=0;
			for (int k=0; k<mKMax-1; k++) {
				for (int j=0; j<mJMax-1; j++) {
					for (int i=0; i<mIMax-1; i++) {
						cells[counter].mNodes[0]=i   + j    *mIMax  + k    *mIMax*mJMax;
						cells[counter].mNodes[1]=i+1 + j    *mIMax  + k    *mIMax*mJMax;
						cells[counter].mNodes[2]=i+1 + (j+1)*mIMax  + k    *mIMax*mJMax;
						cells[counter].mNodes[3]=i   + (j+1)*mIMax  + k    *mIMax*mJMax;
						cells[counter].mNodes[4]=i   + j    *mIMax  + (k+1)*mIMax*mJMax;
						cells[counter].mNodes[5]=i+1 + j    *mIMax  + (k+1)*mIMax*mJMax;
						cells[counter].mNodes[6]=i+1 + (j+1)*mIMax  + (k+1)*mIMax*mJMax;
						cells[counter].mNodes[7]=i   + (j+1)*mIMax  + (k+1)*mIMax*mJMax;
						counter++;
					}
				}
			}
			mesh = new VolumeMesh<HexaederTopo>(mNumElements, mNumPts, cells, points);
		}
		std::cout << "  Data = " << mIMax << "x"<< mJMax << "x"<< mKMax << "\n";
	}
	else { // FEM zones
		PointState * points(new PointState[mNumPts]);
		if (pimpl->mDataSet!=FLOWER_NOVS && pimpl->mDataSet!=FLOWER_VS && pimpl->mDataSet!=FLOWER_ACCO)
			throw "Unsupported data format for unstructured mesh";
		for (int p=0; p<mNumPts*iNumVar; ++p) {
			int i, j;
			if (mDataPacking==BLOCK) j=p/mNumPts, i=p%mNumPts;
			else if (mDataPacking==POINT) j=p%iNumVar, i=p/iNumVar;
			else throw "Unsupported DataPacking";
			// clear surface velocity, if it is not to be read
			if (iNumVar<11 && j==0) points[i].mV=0;
			double value=pimpl->fetchReal(mVarDataFormat[j]);
			if (pimpl->mDataSet==FLOWER_ACCO &&  j>7) continue;
			switch (j) {
				case 0:	points[i][0]=value; break;
				case 1:	points[i][1]=value; break;
				case 2:	points[i][2]=value; break;
				case 3:	points[i].mR=value; break;
				case 4:	points[i].mU.X()=value; break;
				case 5:	points[i].mU.Y()=value; break;
				case 6:	points[i].mU.Z()=value; break;
				case 7:	points[i].mP=value; break;
				case 8: points[i].mV.X()=value; break;
				case 9: points[i].mV.Y()=value; break;
				case 10: points[i].mV.Z()=value; break;
				default: break; // ignore the rest
			}
		}
		int connectivityBase=1;
		if (pimpl->mVersion>=107) connectivityBase=0;
		if (mType==FETRIANGLE) {	// Triangle
			TriangleTopo * cells(new TriangleTopo[mNumElements]);
			for (int i=0; i<mNumElements; i++) {
				cells[i].mNodes[0]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[1]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[2]=pimpl->fetchInt32()-connectivityBase;
			}
			mesh = new SurfaceMesh<TriangleTopo>(mNumElements, mNumPts, cells, points);
			
				std::cout << "  Data = " << mNumPts << " Points "<< mNumElements << " Triangle Elements \n";
		}
		else if (mType==FEQUADRILATERAL) {	// Quadrangle
			QuadrangleTopo * cells(new QuadrangleTopo[mNumElements]);
			for (int i=0; i<mNumElements; i++) {
				cells[i].mNodes[0]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[1]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[2]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[3]=pimpl->fetchInt32()-connectivityBase;
			}
			mesh = new SurfaceMesh<QuadrangleTopo>(mNumElements, mNumPts, cells, points);
		
			std::cout << "  Data = " << mNumPts << " Points "<< mNumElements << " Quadrangle Elements \n";
		}
		else if (mType==FETETRAHEDRON) {	// Tetrahedron
			TetraederTopo * cells(new TetraederTopo[mNumElements]);
			for (int i=0; i<mNumElements; i++) {
				cells[i].mNodes[0]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[1]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[2]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[3]=pimpl->fetchInt32()-connectivityBase;
			}
			mesh = new VolumeMesh<TetraederTopo>(mNumElements, mNumPts, cells, points);
			std::cout << "  Data = " << mNumPts << " Points "<< mNumElements << " Tetrahedron Elements \n";
		}
		else if (mType==FEBRICK) {	// Hexahedron
			assert(0=="Copied and changed, but not yet testes. Remove assert if working as expected");
			HexaederTopo * cells(new HexaederTopo[mNumElements]);
			for (int i=0; i<mNumElements; i++) {
				cells[i].mNodes[0]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[1]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[2]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[3]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[4]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[5]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[6]=pimpl->fetchInt32()-connectivityBase;
				cells[i].mNodes[7]=pimpl->fetchInt32()-connectivityBase;
			}
			mesh = new VolumeMesh<HexaederTopo>(mNumElements, mNumPts, cells, points);
			std::cout << "  Data = " << mNumPts << " Points "<< mNumElements << " Hexahedron Elements \n";
		}
		else {
			std::cerr << "Unknown ZoneType=" << mType << std::endl;
			assert(0=="Unknown ZoneType");
		}
	}
	assert(mesh);
	if (mesh) mesh->setName(mName);
	return mesh;
}

void splitMinMax(std::string const & iSpec, int & oMin, int & oMax) {
	oMin=oMax=0;
	if (iSpec.empty()) return;
	oMin= oMax = 0;
	int num = sscanf(iSpec.c_str(), "%d:%d", &oMin, &oMax);
	if (num <2) {
		
			std::cerr << "Warning: only one value " << oMin << " found for min/max specification (" << iSpec
				<< "), cutting off both sides.\n";
		oMax=-oMin++;
	}
}

void TecplotFile::Zone::parseRindSpec(std::string const & iSpec, TecplotFile::Zone::RindSpec & oRind) {
	if (iSpec.empty()) return;
	std::string spec(iSpec);
	size_t slash=spec.find('/');
	splitMinMax(std::string(spec, 0, slash), oRind.mMinI, oRind.mMaxI);
	if (slash==std::string::npos) return;
	spec.erase(0, slash+1);
	slash=spec.find('/');
	splitMinMax(std::string(spec, 0, slash), oRind.mMinJ, oRind.mMaxJ);
	if (slash==std::string::npos) return;
	spec.erase(0, slash+1);
	slash=spec.find('/');
	splitMinMax(std::string(spec, 0, slash), oRind.mMinK, oRind.mMaxK);
}

TecplotFile::TecplotFile(std::string const & iFile) : mNumVar(0), mVarNames(0), pimpl(new Impl) {
//	::Read(iFile, "");
	if (!pimpl->open(iFile)) throw "Failed to open";
	// ----------- 1.3) Title and variable names -----------
	// file type
	if (pimpl->mVersion>=111) {
		//int fileType=
		pimpl->fetchInt32();
	}
	// title
	mTitle=pimpl->fetchString();
	std::cout << "Title \"" << mTitle << "\"\n";
	// number of variables
    mNumVar=pimpl->fetchInt32();
	// variable names
	mVarNames.resize(mNumVar);
	for (int i=0; i<mNumVar; i++) {
		mVarNames[i]=pimpl->fetchString();
	}
	// marker
	double marker=pimpl->fetchFloat();
	while (marker != pimpl->EOHMARKER) {
		if (marker==pimpl->ZONEMARKER) {
			TecplotFile::Zone TecZone;
			TecZone.ReadHeader(&*pimpl,mNumVar);
			mZones.push_back(TecZone);
			marker=pimpl->fetchFloat();
		}
		else if (marker == pimpl->GEOMETRYMARKER) {
			assert(0=="untested territory of geometry skipping");
			assert(pimpl->mVersion>=75 && "unchecked data layout for tecplot version < 9.2");
			int position=pimpl->fetchInt32();
			int numCoord=2;
			if (position==4 && pimpl->mVersion>=101) numCoord=3; // Grid3D
			if (pimpl->mVersion<101) pimpl->skip32(1);
			else pimpl->skip32(2);
			pimpl->skip64(3);
			pimpl->skip32(4);
			int geomType=pimpl->fetchInt32();
			if (geomType==5 && pimpl->mVersion<101) numCoord=3; // 3DLine < V10.0
			pimpl->skip32(1);
			pimpl->skip64(2);
			pimpl->skip32(3);
			pimpl->skip64(2);
			pimpl->fetchString();
			int gtype=pimpl->fetchInt32(); // polyline type, 1=float, 2=double
			if (pimpl->mVersion>=101) pimpl->skip32(1);
			if (geomType==0 || (geomType==5 && pimpl->mVersion<101)) { // polyline || 3DLine
				assert(0=="untested territory of polyline geometry, check before removing assertion");
				int numLines=pimpl->fetchInt32();
				while (numLines--) {
					int numPoints=pimpl->fetchInt32();
					if (gtype==1) pimpl->skip32(numCoord*numPoints);
					else if (gtype==2) pimpl->skip64(numCoord*numPoints);
					else assert(0=="illegal gtype value");
				}
			}
			else if (geomType==1) { // rectangle
				assert(0=="untested territory of rectangle geometry, check before removing assertion");
				if (gtype==1) pimpl->skip32(2);
				else if (gtype==2) pimpl->skip64(2);
			}
			else if (geomType==2) { // square
				assert(0=="untested territory of square geometry, check before removing assertion");
				if (gtype==1) pimpl->skip32(1);
				else if (gtype==2) pimpl->skip64(1);
			}
			else if (geomType==3) { // circle
				assert(0=="untested territory of circle geometry, check before removing assertion");
				if (gtype==1) pimpl->skip32(1);
				else if (gtype==2) pimpl->skip64(1);
			}
			else if (geomType==4) { // ellipse
				assert(0=="untested territory of ellipse geometry, check before removing assertion");
				if (gtype==1) pimpl->skip32(2);
				else if (gtype==2) pimpl->skip64(2);
			}
			else assert(0=="illegal geometry type value");
			marker=pimpl->fetchFloat();
		}
		else if (marker == pimpl->TEXTMARKER) {
			assert(pimpl->mVersion>=75 && "unchecked data layout for tecplot version < 9.2, check before removing assertion");
			//int position=
			pimpl->fetchInt32();
			pimpl->skip32(1);
			if (pimpl->mVersion<101) pimpl->skip64(2);
			else pimpl->skip64(3);
			pimpl->skip32(2);
			pimpl->skip64(1);
			pimpl->skip32(1);
			pimpl->skip64(2);
			pimpl->skip32(2);
			pimpl->skip64(2);
			pimpl->skip32(3);
			pimpl->fetchString();
			if (pimpl->mVersion>=101) pimpl->skip32(1);
			pimpl->fetchString();
			marker=pimpl->fetchFloat();
		}
		else if (marker == pimpl->LABELMARKER) {
			assert(0=="untested territory of custom label skipping, check before removing assertion");
			assert(pimpl->mVersion>=75 && "unchecked data layout for tecplot version < 9.2, check before removing assertion");
			int numLabels=pimpl->fetchInt32();
			while (numLabels--)
				pimpl->fetchString();
			marker=pimpl->fetchFloat();
		}
		else if (marker == pimpl->USERMARKER) {
			assert(0=="untested territory of user record skipping");
			assert(pimpl->mVersion>=75 && "unchecked data layout for tecplot version < 9.2, check before removing assertion");
			pimpl->fetchString();
			marker=pimpl->fetchFloat();
		}
		else if (marker == pimpl->DATAAUXMARKER) {
			 {
				if (pimpl->mVersion<101) {
					std::cerr << "ERROR: Dataset auxiliary data allowed only from tecplot 10 onwards!\n";
					std::cerr << "\ttrying to keep going, but file inconsistency is to be expected\n";
				}
			}
			assert(0=="untested territory of dataset auxiliary data skipping, check before removing assertion");
			pimpl->fetchString();
			pimpl->skip32(1);
			pimpl->fetchString();
			marker=pimpl->fetchFloat();
		}
		else if (marker == pimpl->VARAUXMARKER) {
			 {
				if (pimpl->mVersion<107) {
					std::cerr << "ERROR: Variable auxiliary data allowed only from tecplot 360-2006 onwards!\n";
					std::cerr << "\ttrying to keep going, but file inconsistency is to be expected\n";
				}
			}
			assert(0=="untested territory of dataset auxiliary data skipping, check before removing assertion");
			assert(0=="implausible data layout?!? check before removing assertion");
			pimpl->fetchString();
			pimpl->fetchString();
			pimpl->skip32(1);
			pimpl->fetchString();
			marker=pimpl->fetchFloat();
		}
		else {
			
				std::cerr << "ERROR: Unknown marker found (" << marker << ") instead of end of header!\n";
			assert(0);
		}
	}
	if (mZones.size()>1) 
		std::cout << mZones.size() << " Zones found:\n";
}

// due to a bug in (at least) gcc 4.4.0 (apparently fixed in 4.4.1), an (otherwise
// unnecessary) explicit destructor is needed in order to close the Impl.mStream
TecplotFile::~TecplotFile() {}

MeshBaseVec TecplotFile::Read(std::string const & iZoneRindList) {
	pimpl->mDataSet=UNKNOWN;
	// check variables
	if (mNumVar==21) {
		if ((mVarNames[0]==std::string("CoordinateX")) &&
			(mVarNames[1]==std::string("CoordinateY")) &&
			(mVarNames[2]==std::string("CoordinateZ")) &&
			(mVarNames[3]==std::string("rho")) &&
			(mVarNames[4]==std::string("u")) &&
			(mVarNames[5]==std::string("v")) &&
			(mVarNames[6]==std::string("w")) &&
			(mVarNames[7]==std::string("Ma")) &&
			(mVarNames[8]==std::string("cp")) &&
			(mVarNames[9]==std::string("YPlus")) &&
			(mVarNames[10]==std::string("TotalPressLoss")) &&
			(mVarNames[11]==std::string("ViscosityEddy")) &&
			(mVarNames[12]==std::string("cfinf")) &&
			(mVarNames[13]==std::string("cfloc")) &&
			(mVarNames[14]==std::string("LamTransFlag")) &&
			(mVarNames[15]==std::string("TurbulentEnergyKinetic")) &&
			(mVarNames[16]==std::string("TurbulentDissipationRate")) &&
			(mVarNames[17]==std::string("iblank")) &&
			(mVarNames[18]==std::string("nx")) &&
			(mVarNames[19]==std::string("ny")) &&
			(mVarNames[20]==std::string("nz")) )
			pimpl->mDataSet=FLOWER_SURFACE;
	}
	else if (mNumVar==17) {
		if ((mVarNames[0]==std::string("CoordinateX")) &&
			(mVarNames[1]==std::string("CoordinateY")) &&
			(mVarNames[2]==std::string("CoordinateZ")) &&
			(mVarNames[3]==std::string("rho")) &&
			(mVarNames[4]==std::string("u")) &&
			(mVarNames[5]==std::string("v")) &&
			(mVarNames[6]==std::string("w")) &&
			(mVarNames[7]==std::string("Ma")) &&
			(mVarNames[8]==std::string("cp")) &&
			(mVarNames[9]==std::string("Entropy")) &&
			(mVarNames[10]==std::string("TotalPressLoss")) &&
			(mVarNames[11]==std::string("ViscosityEddy")) &&
			(mVarNames[12]==std::string("LamTransFlag")) &&
			(mVarNames[13]==std::string("TurbulentDistance")) &&
			(mVarNames[14]==std::string("TurbulentEnergyKinetic")) &&
			(mVarNames[15]==std::string("TurbulentDissipationRate")) &&
			(mVarNames[16]==std::string("iblank")) )
			pimpl->mDataSet=FLOWER_PLANFILE;
	}
	else if (mNumVar==11) {
		if ((mVarNames[0]==std::string("x")) &&
			(mVarNames[1]==std::string("y")) &&
			(mVarNames[2]==std::string("z")) &&
			(mVarNames[3]==std::string("rho")) &&
			(mVarNames[4]==std::string("u")) &&
			(mVarNames[5]==std::string("v")) &&
			(mVarNames[6]==std::string("w")) &&
			(mVarNames[7]==std::string("p")) &&
			(mVarNames[8]==std::string("us")) &&
			(mVarNames[9]==std::string("vs")) &&
			(mVarNames[10]==std::string("ws")) )
			pimpl->mDataSet=FLOWER_VS;
	}
	else if (mNumVar==8) {
		if (((mVarNames[0]==std::string("x")) || (mVarNames[0]==std::string("CoordinateX"))) &&
			((mVarNames[1]==std::string("y")) || (mVarNames[1]==std::string("CoordinateY"))) &&
			((mVarNames[2]==std::string("z")) || (mVarNames[2]==std::string("CoordinateZ"))) &&
			(mVarNames[3]==std::string("rho")) &&
			(mVarNames[4]==std::string("u")) &&
			(mVarNames[5]==std::string("v")) &&
			(mVarNames[6]==std::string("w")) &&
			(mVarNames[7]==std::string("p")) )
			pimpl->mDataSet=FLOWER_NOVS;
	}
	else if (mNumVar==13) {
		if ((mVarNames[0]==std::string("CoordinateX")) &&
			(mVarNames[1]==std::string("CoordinateY")) &&
			(mVarNames[2]==std::string("CoordinateZ")) &&
			(mVarNames[3]==std::string("rho")) &&
			(mVarNames[4]==std::string("u")) &&
			(mVarNames[5]==std::string("v")) &&
			(mVarNames[6]==std::string("w")) &&
			(mVarNames[7]==std::string("p")) &&
			(mVarNames[8]==std::string("iblank")) &&
			(mVarNames[9]==std::string("VortX")) &&
			(mVarNames[10]==std::string("VortY")) &&
			(mVarNames[11]==std::string("VortZ")) &&
			(mVarNames[12]==std::string("Lambda2")) )
			pimpl->mDataSet=FLOWER_ACCO;
	}
	else if (mNumVar==14) {
		if ((mVarNames[0]==std::string("psi")) &&
			(mVarNames[1]==std::string("QCx")) &&
			(mVarNames[2]==std::string("QCy")) &&
			(mVarNames[3]==std::string("QCz")) &&
			(mVarNames[4]==std::string("dr")) &&
			(mVarNames[5]==std::string("chord")) &&
			(mVarNames[6]==std::string("Fn")) &&
			(mVarNames[7]==std::string("Ft")) &&
			(mVarNames[8]==std::string("Nx")) &&
			(mVarNames[9]==std::string("Ny")) &&
			(mVarNames[10]==std::string("Nz")) &&
			(mVarNames[11]==std::string("Tx")) &&
			(mVarNames[12]==std::string("Ty")) &&
			(mVarNames[13]==std::string("Tz")) )
			pimpl->mDataSet=LIFT_LINE;
	}
	else if (mNumVar==3) {
		if ((mVarNames[0]==std::string("X")) &&
			(mVarNames[1]==std::string("Y")) &&
			(mVarNames[2]==std::string("Z")) )
			pimpl->mDataSet=SURFACE;
	}
	else if (mNumVar==12) {
		if (false) {}
#ifdef AHD
		else if ((mVarNames[0]==std::string("Time [s]")) &&
			(mVarNames[1]==std::string("Azimuth")) &&
			(mVarNames[2]==std::string("x [m]")) &&
			(mVarNames[3]==std::string("y [m]")) &&
			(mVarNames[4]==std::string("z [m]")) &&
			(mVarNames[5]==std::string("nx [-]")) &&
			(mVarNames[6]==std::string("ny [-]")) &&
			(mVarNames[7]==std::string("nz [-]")) &&
			(mVarNames[8]==std::string("Area [m^2]")) &&
			(mVarNames[9]==std::string("Psi [deg]")) &&
			(mVarNames[10]==std::string("r_ACP [m]")) &&
			(mVarNames[11]==std::string("x_ACP [-]")) )
			pimpl->mDataSet=CAMRAD_SURFACE;
#endif // AHD
#ifdef MARENCO
// "Time [s]" "Azimuth" "x [m]" "y [m]" "z [m]" "nx [-]" "ny [-]" "nz [-]" "Area [m^2]" "Psi [deg]" "r_ACP [m]""x_ACP [-]"
		else if ((mVarNames[0]==std::string("Time [s]")) &&
			(mVarNames[1]==std::string("Azimuth")) &&
			(mVarNames[2]==std::string("x [m]")) &&
			(mVarNames[3]==std::string("y [m]")) &&
			(mVarNames[4]==std::string("z [m]")) &&
			(mVarNames[5]==std::string("nx [-]")) &&
			(mVarNames[6]==std::string("ny [-]")) &&
			(mVarNames[7]==std::string("nz [-]")) &&
			(mVarNames[8]==std::string("Area [m^2]")) &&
			(mVarNames[9]==std::string("Psi [deg]")) &&
			(mVarNames[10]==std::string("r_ACP [m]")) &&
			(mVarNames[11]==std::string("x_ACP [-]")) )
			pimpl->mDataSet=FLIGHTLAB_SURFACE;
#endif // MARENCO
	}
	else if (mNumVar==10) {
//"x" "y" "z" "Alpha" "Lift" "Drag" "RadialForce" "Radius" "Chord" "PanelWidth"
		if ((mVarNames[0]==std::string("time [s]")) &&
			(mVarNames[1]==std::string("psi [deg]")) &&
			(mVarNames[2]==std::string("QCx_loc [m]")) &&
			(mVarNames[3]==std::string("QCy_loc [m]")) &&
			(mVarNames[4]==std::string("QCz_loc [m]")) &&
			(mVarNames[5]==std::string("Fx_loc [N]")) &&
			(mVarNames[6]==std::string("Fy_loc [N]")) &&
			(mVarNames[7]==std::string("Fz_loc [N]")) &&
			(mVarNames[8]==std::string("A [m^2]")) &&
			(mVarNames[9]==std::string("rho [kg/m^3]")))
			pimpl->mDataSet=IAGCOUPLE_ACCOLOAD;
#ifdef AHD
		else if ((mVarNames[0]==std::string("x")) &&
			(mVarNames[1]==std::string("y")) &&
			(mVarNames[2]==std::string("z")) &&
			(mVarNames[3]==std::string("Alpha")) &&
			(mVarNames[4]==std::string("Lift")) &&
			(mVarNames[5]==std::string("Drag")) &&
			(mVarNames[6]==std::string("RadialForce")) &&
			(mVarNames[7]==std::string("Radius")) &&
			(mVarNames[8]==std::string("Chord")) &&
			(mVarNames[9]==std::string("PanelWidth")))
			pimpl->mDataSet=CAMRAD_LINE;
#endif // AHD
	}
#ifdef MARENCO
	else if (mNumVar==18) {
//	"Time [s]" "Azimuth" "Psi [deg]" "r_ACP [m]""x_ACP [-]""x_ACP1_IN [m] ""y_ACP1_IN [m] ""z_ACP1_IN [m] ""dr_ACP [m]"
//	"clen_ACP [m]""dT/dr [N/m]" "dD/dr [N/m]" "Vvec_x_IN [-]" "Vvec_y_IN [-]" "Vvec_z_IN [-]" "Hvec_x_IN [-]" "Hvec_y_IN [-]" "Hvec_z_IN [-]"
		if ((mVarNames[0]==std::string("Time [s]")) &&
			(mVarNames[1]==std::string("Azimuth")) &&
			(mVarNames[2]==std::string("Psi [deg]")) &&
			(mVarNames[3]==std::string("r_ACP [m]")) &&
			(mVarNames[4]==std::string("x_ACP [-]")) &&
			(mVarNames[5]==std::string("x_ACP1_IN [m]")) &&
			(mVarNames[6]==std::string("y_ACP1_IN [m]")) &&
			(mVarNames[7]==std::string("z_ACP1_IN [m]")) &&
			(mVarNames[8]==std::string("dr_ACP [m]")) &&
			(mVarNames[9]==std::string("clen_ACP [m]")) &&
			(mVarNames[10]==std::string("dT/dr [N/m]")) &&
			(mVarNames[11]==std::string("dD/dr [N/m]")) &&
			(mVarNames[12]==std::string("Vvec_x_IN [-]")) &&
			(mVarNames[13]==std::string("Vvec_y_IN [-]")) &&
			(mVarNames[14]==std::string("Vvec_z_IN [-]")) &&
			(mVarNames[15]==std::string("Hvec_x_IN [-]")) &&
			(mVarNames[16]==std::string("Hvec_y_IN [-]")) &&
			(mVarNames[17]==std::string("Hvec_z_IN [-]")) )
			pimpl->mDataSet=FLIGHTLAB_LINE;
		else if ((mVarNames[0]==std::string("Time [s]")) &&
			(mVarNames[1]==std::string("Azimuth")) &&
			(mVarNames[2]==std::string("Psi [deg]")) &&
			(mVarNames[3]==std::string("r_ACP [m]")) &&
			(mVarNames[4]==std::string("x_ACP [-]")) &&
			(mVarNames[5]==std::string("x_ACP1_HC [m]")) &&
			(mVarNames[6]==std::string("y_ACP1_HC [m]")) &&
			(mVarNames[7]==std::string("z_ACP1_HC [m]")) &&
			(mVarNames[8]==std::string("dr_ACP [m]")) &&
			(mVarNames[9]==std::string("clen_ACP [m]")) &&
			(mVarNames[10]==std::string("dT/dr [N/m]")) &&
			(mVarNames[11]==std::string("dD/dr [N/m]")) &&
			(mVarNames[12]==std::string("Vvec_x_HC [-]")) &&
			(mVarNames[13]==std::string("Vvec_y_HC [-]")) &&
			(mVarNames[14]==std::string("Vvec_z_HC [-]")) &&
			(mVarNames[15]==std::string("Hvec_x_HC [-]")) &&
			(mVarNames[16]==std::string("Hvec_y_HC [-]")) &&
			(mVarNames[17]==std::string("Hvec_z_HC [-]")) )
			pimpl->mDataSet=FLIGHTLAB_LINE;
	}
#endif // MARENCO
	if (pimpl->mDataSet==UNKNOWN) {
		{
			std::cerr << "ERROR: Unsupported data format: number of variables = " << mNumVar << ":\n";
			if (mNumVar>0) {
				std::cerr << mVarNames[0];
				for (int i=1; i<mNumVar; ++i)
					std::cerr << " " << mVarNames[i];
				std::cerr << std::endl;
			}
		}
		throw "Unsupported variables";
	}

	// data section
	MeshBaseVec mesh;
	int j=1;
	for (std::vector<TecplotFile::Zone>::iterator i=mZones.begin(); i!=mZones.end(); ++i, ++j) {
		TecplotFile::Zone::RindSpec rind = { 0, 0, 0, 0, 0, 0, };
		size_t zoneNamePos=iZoneRindList.find(','+i->getName()+'{');
		if (zoneNamePos!=std::string::npos) {
			// rind specification found
			size_t rindPos=zoneNamePos+1+i->getName().length()+1;
			size_t endRind=iZoneRindList.find('}', rindPos);
			if (endRind==std::string::npos) {
				
					std::cerr << "Missing closing brace in rind specification for zone " << i->getName() << std::endl;
				throw "Missing closing brace in rind specification for zone";
			}
			size_t comma=iZoneRindList.find(',', rindPos);
			if (comma!=std::string::npos && comma<endRind) {
				
					std::cerr << "Missing closing brace (early comma) in rind specification for zone " << i->getName() << std::endl;
				throw "Missing closing brace (early comma) in rind specification for zone";
			}
			i->parseRindSpec(std::string(iZoneRindList, rindPos, endRind-rindPos), rind);
		}
		double marker=pimpl->fetchFloat();
		assert(marker==pimpl->ZONEMARKER);
		std::cout << "Zone (" << i->getName() << ") "<< j <<":";
		mesh.push_back(i->ReadData(&*pimpl, mNumVar, rind));
	}
	return mesh;
}

std::vector<MeshPts *> TecplotFile::ReadInnerPts() {
	std::cout << "var names ..." << mVarNames[0] << mVarNames[1] << mVarNames[2]  << std::endl;
	// check variables
	if (mNumVar==3) {
		if ((mVarNames[0]!=std::string("x")) ||
				(mVarNames[1]!=std::string("y")) ||
				(mVarNames[2]!=std::string("z")) ) {
			std::cerr << "ERROR: Inner Points, Wrong variables (Support only for: x,y,z)! \n";
			std::cerr << mVarNames[0] << " " << mVarNames[1] << " " << mVarNames[2] << std::endl;
			throw "Inner Points, Unsupported variables";
		}
	}
	else {
		std::cerr << "ERROR: Inner Points, Unsupported number of variables (" << mNumVar << ")! \n";
		throw "Inner Points, Unsupported number of variables";
	}
	// data section
	MeshPtsVec mesh;
	int j=0;
	for (std::vector<TecplotFile::Zone>::iterator i=mZones.begin(); i!=mZones.end(); ++i, ++j) {
		double marker=pimpl->fetchFloat();
		assert(marker==pimpl->ZONEMARKER);
		std::cout << "Zone " << j <<":";
		mesh.push_back(i->ReadInnerPtsData(&*pimpl, mNumVar));
	}
	return mesh;
}

#ifdef HAVE_TECIO
#include <MASTER.h>
#include <GLOBAL.h>
#include <TECIO.h>
#include <DATAUTIL.h>
#include <ARRLIST.h>
#include <STRLIST.h>
#endif

/*ReadTec(Boolean_t   GetHeaderInfoOnly,
	char           *FName,
	short          *IVersion,
	char          **DataSetTitle,
	EntIndex_t     *NumZones,
	EntIndex_t     *NumVars,
	StringList_pa  *VarNames,
	StringList_pa  *ZoneNames,
	LgIndex_t     **NumPtsI,
	LgIndex_t     **NumPtsJ,
	LgIndex_t     **NumPtsK,
	ZoneType_e    **ZoneType,
	StringList_pa  *UserRec,
	AuxData_pa     *DatasetAuxData,
	Boolean_t       RawDataspaceAllocated,
	NodeMap_t    ***NodeMap,
	double       ***VDataBase);
*/
MeshBaseVec Read(std::string const & iFile, std::string const & /*iZoneRindList*/) {
	MeshBaseVec zones;
#ifdef HAVE_TECIO
	short fileVersion;
	char * title;
	EntIndex_t numZones, numVars;
	StringList_pa varNames, zoneNames;
	LgIndex_t * numPtsI, * numPtsJ, * numPtsK;
	ZoneType_e * zoneType;
	NodeMap_t ** nodeMap=0;
	double ** vDataBase=0;
	bool ok=ReadTec(FALSE /*GetHeaderInfoOnly*/,
		const_cast<char *>(iFile.c_str()), &fileVersion,
		&title, &numZones, &numVars, &varNames, &zoneNames, 
		&numPtsI, &numPtsJ, &numPtsK, &zoneType, 
		0 /*UserRec*/, 
		0 /*DatasetAuxData*/, 
		FALSE /*RawDataspaceAllocated*/,
		&nodeMap,
		&vDataBase);
SHOW_VAR4(ok, title, numZones, numVars);
if (StringListCount(varNames)>0) SHOW_VAR1(StringListGetStringRef(varNames, 0));
if (StringListCount(varNames)>1) SHOW_VAR1(StringListGetStringRef(varNames, 1));
if (StringListCount(zoneNames)>0) SHOW_VAR1(StringListGetStringRef(zoneNames, 0));
if (StringListCount(zoneNames)>1) SHOW_VAR1(StringListGetStringRef(zoneNames, 1));
if (numZones>0) SHOW_VAR3(numPtsI[0], numPtsJ[0], numPtsK[0]);
if (numZones>1) SHOW_VAR3(numPtsI[1], numPtsJ[1], numPtsK[1]);
SHOW_VAR2(zoneType[0], zoneType[1]);
SHOW_VAR2(nodeMap, vDataBase);
#endif
	return zones;
}
