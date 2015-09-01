#ifndef VRCWUTILS_H_
#define VRCWUTILS_H_

#include <QString>

class hIPLkpVal;


/*****
 * Constants
 *****/
const int DEF_WARN = 99;
const int DEF_ERROR = 99;
const int UNUSED_ADDR = 99;
const QString OTHER = "--other--";
const QString HIPLKP_EC_1 = "hostInvalid";


/*****
 * variables, lists
 *****/
enum cWall {Front = 0, Left, Right, Bottom, Top, Back};
enum proKind {Powerwall = 0, CAVE, _3D_TV};
enum stType {passive = 0, active, topBottom, sideBySide, checkerboard,
   vInterlaced, cInterleave, hInterlaced, rInterleave};
enum trackHandle {cov = 0, vrc};
enum trackSys {ART = 0, Vicon, InterSense, Polhemus, FOB, Motionstar, Wii};
enum btnSys {ART_Fly = 0, o_Optical, Wand, Stylus, FOB_Mouse, Hornet, Mike,
   Wiimote};
enum btnDrv {DTrack = 0, Mouse_Buttons, PolhemusDrv, FOB_Drv, MotionstarDrv,
   HornetDrv, MikeDrv, WiimoteDrv};
enum opSys {Linux = 0, Windows};
enum execMode {ssh = 0, covremote, CovDaemon, rsh};
enum typePro {Monitor = 0, Projector};
enum aspRat {_43 = 0, _169, _1610, _oth, _all};
enum loRoOrient {leftOf = 0, rightOf};
enum zPoint {cDim = 0, fWall, boWall};


/*****
 * functions
 *****/
//Umwandlung von QString in enum
cWall strToCWall(const QString& cw);
proKind strToProKind(const QString& pk);
stType strToStType(const QString& st);
trackSys strToTrackSys(const QString& ts);
btnSys strToBtnSys(const QString& bs);
btnDrv strToBtnDrv(const QString& bd);
opSys strToOpSys(const QString& os);
execMode strToExecMode(const QString& es);
typePro strToTypePro(const QString& tp);
zPoint strToZeroPoint(const QString& zp);

//Umwandlung von enum nach QString
QString cWallToStr(const cWall& cw);
QString proKindToStr(const proKind& pk);
QString stTypeToStr(const stType& st);
QString trackSysToStr(const trackSys& ts);
QString btnSysToStr(const btnSys& bs);
QString btnDrvToStr(const btnDrv& bd);
QString opSysToStr(const opSys& os);
QString execModeToStr(const execMode& em);
QString typeProToStr(const typePro& tp);
QString zeroPintToStr(const zPoint& zp);

//for validation of given hostname or IP-Address
hIPLkpVal* hostIPLookup(const QString& host);

#endif /* VRCWUTILS_H_ */
