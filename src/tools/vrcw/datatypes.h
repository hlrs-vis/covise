#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <QString>
#include <QVector>
#include <QHostInfo>

#include "vrcwutils.h"



/*****
 * class configVal
 *****/
class configVal
{
public:
   /*****
    * constructor - destructor
    *****/
   configVal();
   ~configVal();

   /*****
    * variables
    *****/
   QString coConfDir;
   bool coConfDirWritable;
   QString coConfName;
   //Name give in LineEdit
   QString projName;
   QString coConfTmpDir;
};



/*****
 * class wallVal
 *****/
class wallVal
{
public:
   /*****
    * constructor - destructor
    *****/
   wallVal();
   ~wallVal();

   /*****
    * variables
    *****/
   cWall wall;
   QVector<int> res;
   QVector<int> screenSize;
   QVector<int> rowCol;
   typePro typeProj;
   QVector<int> wallSize;
   QVector<int> overlap;
   QVector<int> frame;
};



/*****
 * class handSensVal
 *****/
class handSensVal
{
public:
   /*****
    * constructor - destructor
    *****/
   handSensVal();
   ~handSensVal();

   /*****
    * variables
    *****/
   QString sensLabel; //hand sensor 1-3
   btnSys bSys;
   QString bDev;
   btnDrv bDrv;
   int bIndex;
   int bAddr;
};



/*****
 * class headSensVal
 *****/
class headSensVal
{
public:
   /*****
    * constructor - destructor
    *****/
   headSensVal();
   ~headSensVal();

   /*****
    * variables
    *****/
   QString sensLabel; //head sensor 1-3
   int bIndex;
};


/*****
 * class personVal
 *****/
class personVal
{
public:
   /*****
    * constructor - destructor
    *****/
   personVal();
   ~personVal();

   /*****
    * variables
    *****/
   QString personLabel; //Person 1-9
   QString handSens;
   QString headSens;
};



/*****
 * class sensTrackSysDim
 *****/
class sensTrackSysDim
{
public:
   /*****
    * constructor - destructor
    *****/
   sensTrackSysDim();
   ~sensTrackSysDim();

   /*****
    * variables
    *****/
   QString sensLabel; //hand sensor 1-3 or head sensor 1-3
   int x;
   int y;
   int z;
   int h;
   int p;
   int r;
};



/*****
 * class trackHwValDim
 *****/
class trackHwValDim
{
public:
   /*****
    * constructor - destructor
    *****/
   trackHwValDim();
   ~trackHwValDim();

   /*****
    * variables
    *****/
   trackHandle tHandle;
   int vrcPort;
   trackSys tSys;
   bool checkArtHost;
   int artHostSPort;
   int artRPort;
   QString polFobSPort;
   // ART, Vicon, Motionstar host/IP Address stored in hostIPAddr
   QString hostIPAddr;
   int numHandSens;
   QVector<handSensVal*> handSVal;
   int numHeadSens;
   QVector<headSensVal*> headSVal;
   int numPersons;
   QVector<personVal*> persVal;
   sensTrackSysDim* tSysDim;
   QVector<sensTrackSysDim*> handSDim;
   QVector<sensTrackSysDim*> headSDim;
};



/*****
 * class hostIPLookupVal
 *****/
class hIPLkpVal
{
public:
   /*****
    * constructor - destructor
    *****/
   hIPLkpVal();
   ~hIPLkpVal();


   /*****
    * variables
    *****/
   QString toLookup;
   QString ipAddr_1;
   QString fqdn_2;
   QString hostname_2;
   QString errCode;

   //for testing output
   QHostInfo hostInfo_1;
   QHostInfo hostInfo_2;
};

#endif /* DATATYPES_H_ */
