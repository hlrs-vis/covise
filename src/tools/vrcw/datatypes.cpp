#include "datatypes.h"


/*****
 * class configVal
 *****/

/*****
 * constructor - destructor
 *****/

configVal::configVal()
{
   coConfDirWritable = false;
}

configVal::~configVal()
{

}




/*****
 * class wallVal
 *****/

/*****
 * constructor - destructor
 *****/

wallVal::wallVal()
{
   wall = Front;
   res = QVector<int>(2,0);
   screenSize = QVector<int>(2,0);
   rowCol = QVector<int>(2,0);
   typeProj = Monitor;
   wallSize = QVector<int>(2,0);
   overlap = QVector<int>(2,0);
   frame = QVector<int>(2,0);
}

wallVal::~wallVal()
{

}



/*****
 * class handSensVal
 *****/

/*****
 * constructor - destructor
 *****/

handSensVal::handSensVal()
{
   bSys = ART_Fly;
   bDrv = DTrack;
   bIndex = 0;
   bAddr = 0;
}

handSensVal::~handSensVal()
{

}



/*****
 * class headSensVal
 *****/

/*****
 * constructor - destructor
 *****/

headSensVal::headSensVal()
{
   bIndex = 0;
}

headSensVal::~headSensVal()
{

}



/*****
 * class personVal
 *****/

/*****
 * constructor - destructor
 *****/
personVal::personVal()
{

}

personVal::~personVal()
{

}




/*****
 * class sensTrackSysDim
 *****/

/*****
 * constructor - destructor
 *****/
sensTrackSysDim::sensTrackSysDim()
{
   x = 0;
   y = 0;
   z = 0;
   h = 0;
   p = 0;
   r = 0;
};

sensTrackSysDim::~sensTrackSysDim()
{

}



/*****
 * class trackHwValDim
 *****/

/*****
 * constructor - destructor
 *****/

trackHwValDim::trackHwValDim()
{
   tHandle = cov;
   vrcPort = 0;
   tSys = ART;
   checkArtHost = false;
   artHostSPort = 0;
   artRPort = 0;
   numHandSens = 0;
   handSVal = QVector<handSensVal*>(0);
   numHeadSens = 0;
   headSVal = QVector<headSensVal*>(0);
   numPersons = 0;
   persVal = QVector<personVal*>(0);
   tSysDim = new sensTrackSysDim();
   handSDim = QVector<sensTrackSysDim*>(0);
   headSDim = QVector<sensTrackSysDim*>(0);
}

trackHwValDim::~trackHwValDim()
{

}



/*****
 * class hostIPLookupVal
 *****/

/*****
 * constructor - destructor
 *****/
hIPLkpVal::hIPLkpVal()
{

}

hIPLkpVal::~hIPLkpVal()
{

}
