#include <config/CoviseConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <util/unixcompat.h>

#ifdef WIN32
#include <direct.h>
#endif

#include "DomainDecomposition.h"
#include <General/include/geo.h>
#include <General/include/cov.h>
#include <General/include/log.h>
#include <zerno/callback.h>
//#include <DraftTube/include/tube.h>
//#include <DraftTube/include/tgrid.h>
#ifdef WIN32
# ifndef __MINGW32__
# define mainn_ MAINN
# endif
#define sendgeodata_ SENDGEODATA
#define sendrbedata_ SENDRBEDATA
#endif

extern "C" void mainn_(int*, int*, int*,
					   int*, int*, int*, float*, float*, float*,
					   int*, int*,
					   int*, int*,
					   int*, int*,
					   int*, float*,
					   int*, int *, int*,
					   int*, int*, int*,
					   int*, int*, int*, //fl float*,
					   int*, int*);
extern "C" void sendgeodata_module_(int *igeb, int *npoin_ges, int *nelem_ges, int *knmax_num, int *elmax_num,
							 int *nkn, int *nel, int *ncd, int *nkd, float *cov_coord,
							 int *cov_lnods, int *cov_lnods_num, int *cov_lnods_proz,
							 int *cov_coord_num, int *cov_coord_joi, int *cov_lnods_joi, int *cov_coord_mod,
							 int *cov_lnods_mod, int *cov_coord_proz);

extern "C" void sendrbedata_module_(int *igeb,int *nrbpo_geb,int *nwand_geb,int *npres_geb,
							 int *nsyme_geb, int *nconv_geb,int *nrbknie,
							 int *cov_displ_kn,int *cov_displ_typ,
							 int *cov_wand_el,int *cov_wand_kn,int *cov_wand_num,
							 int *cov_pres_el,int *cov_pres_kn,int *cov_pres_num,
							 int *cov_conv_el,int *cov_conv_kn,int *cov_conv_num,
							 float *cov_displ_wert, int *reicheck);

static coDistributedObject **pboco;
static coDistributedObject **pgrid;
#ifndef YAC
static const char* pbocobase;
static const char* pgridbase;
#endif
static coOutputPort *p_distGrid;
static coOutputPort *p_distBoco;

void DumpIntArr(const char *s, int *a, int num, int col);

void DomainDecomposition::postInst()
{
	p_numPart->show();
}


DomainDecomposition::DomainDecomposition(int argc, char *argv[])
: coSimpleModule(argc, argv, "Domain Decomposition" )
{
        set_sendgeodata_func(sendgeodata_module_);
        set_sendrbedata_func(sendrbedata_module_);

	// input parameters
	// rei: NUR temporear !!!
	p_numPart = addInt32Param("numPart","number of Partitions");
	p_numPart->setValue(2);

	// Starting directory for zer
	p_dir   = addStringParam("directory","Starting directory");
   std::string dataPath; 
#ifdef WIN32
   const char *defaultDir = getenv("USERPROFILE");
#else
   const char *defaultDir = getenv("HOME");
#endif
   if(defaultDir)
      dataPath=coCoviseConfig::getEntry("value","Module.IHS.DataPath",defaultDir);
   else
      dataPath=coCoviseConfig::getEntry("value","Module.IHS.DataPath","/data/IHS");
	p_dir->setValue((dataPath+std::string("/domdec")).c_str());
    
	// write output files?
	p_writeFiles = addBooleanParam("writeFiles","write output Files?");
	p_writeFiles->setValue(0);
	
	p_zerbuflen = addInt32Param("memory","Memory for zerno in MB");
	p_zerbuflen->setValue(20);
   
	// the intput ports
	p_grid  = addInputPort("grid","UnstructuredGrid","Input Grid");
	p_boco  = addInputPort("boco","USR_FenflossBoco","Boundary Conditions");

	// the output ports
	p_distGrid = addOutputPort("distGrid","UnstructuredGrid","Distributed Grid");
	p_distBoco = addOutputPort("distBoco","USR_DistFenflossBoco","Distributed Boundary Cond");
	//p_ogrid    = addOutputPort("mGrid","coDoUnstructuredGrid","Computation grid");
}


int DomainDecomposition::compute(const char *)
{
	//time
#ifndef WIN32
	timeval time1, time2;
	gettimeofday (&time1, NULL);
#endif

	int numParts,i;
	const coDistributedObject *obj;
	const coDoUnstructuredGrid *grid;
	const coDoSet *boco;
	static int *zerbuf = NULL;
	int zerbuflen = 1024*1024*p_zerbuflen->getValue();
        int col_node, col_elem; //, col_wall, col_diriclet, col_balance, col_press;
	int covise_run = 1;
	//int numnodeInfo;
	int *nodeInfo;
	//int numelemInfo;
	int *elemInfo;
	int numDiriclet;
	int *diricletIndex;
	float *diricletVal;
	int numWall;
	int *wall;
	int numBalance;
	int *balance;
	int *press=0;
// fl	float *pressVal=NULL;
	int numPress=0;

	///////////////////////////// get GRID
	obj = p_grid->getCurrentObject();
    grid = dynamic_cast<const coDoUnstructuredGrid *>(obj);
	if (!grid)
	{
		sendWarning("Object for port %s not correctly received",p_grid->getName());
		fprintf(stderr, "Set of old grid ...\n");
		return STOP_PIPELINE;
	}

	int *elemList,*covConnList;
	int nume, numc, nump;
	float *x,*y,*z;
	grid->getAddresses(&elemList,&covConnList,&x,&y,&z);
	grid->getGridSize(&nume, &numc, &nump);

	// fortran numbering
	int *connList = new int[numc];

	for (i=0;i<numc;i++)
		connList[i] = covConnList[i]+1;

	obj = p_boco->getCurrentObject();
    boco = dynamic_cast<const coDoSet *>(obj);
	if (!boco)
	{
		sendWarning("Object for port %s not correctly received",p_boco->getName());
		return FAIL;
	}

	int numObj;
	const coDistributedObject *const *setObj = boco->getAllElements(&numObj);

	if ( ! ((numObj == 7) || (numObj == 8))  )
	{
                sendWarning("Wrong number of elements in coDistributedObject: Call programmer ;-)");
		return FAIL;
	}
	// We extract from the distributed object the elements in the same
	// way, like we insert them in GeometryGenerator

	// 0. the number of cols ...
    const coDoIntArr *colInfo = dynamic_cast<const coDoIntArr *>(setObj[0]);
	if (!colInfo)
	{
                sendWarning("illegal part type (0) in boco");
		return FAIL;
	}
	else
	{
		//int numcolInfo = colInfo->getDimension(1);
		int *tmp = colInfo->getAddress();
		col_node     = tmp[0];
		col_elem     = tmp[1];
//		col_diriclet = tmp[2];
//		col_wall     = tmp[3];
//		col_balance  = tmp[4];
//		if (numObj == 8)
//			col_press    = tmp[5];
	}

	// 1. node-info
    const coDoIntArr *nodeInfoObj = dynamic_cast<const coDoIntArr *>(setObj[1]);
	if (!nodeInfoObj)
	{
                sendWarning("illegal part type (1) in boco");
		return FAIL;
	}
	else
	{
		//numnodeInfo = nodeInfoObj->getDimension(1);
		nodeInfo    = nodeInfoObj->getAddress();
	}

	// 2. elem-info
    const coDoIntArr *elemInfoObj = dynamic_cast<const coDoIntArr *>(setObj[2]);
	if (!elemInfoObj)
	{
                sendWarning("illegal part type (2) in boco");
		return FAIL;
	}
	else
	{
		//numelemInfo = elemInfoObj->getDimension(0);
		elemInfo    = elemInfoObj->getAddress();
	}

	int colDiriclet;

	// 3.diriclet
    const coDoIntArr *diricletIndexObj = dynamic_cast<const coDoIntArr *>(setObj[3]);
	if (!diricletIndexObj)
	{
		sendWarning("illegal part type (3) in boco");
		return FAIL;
	}
	else
	{
		colDiriclet   = diricletIndexObj->getDimension(0);
		numDiriclet   = diricletIndexObj->getDimension(1);
		diricletIndex = diricletIndexObj->getAddress();
		printf("numDiriclet=%d, colDiriclet=%d\n", numDiriclet, colDiriclet);
		DumpIntArr("diricletIndex", diricletIndex, numDiriclet, 2);
	}

	///////// Diriclet values ...
    const coDoFloat *diricletValObj = dynamic_cast<const coDoFloat *>(setObj[4]);
	if (!diricletValObj)
	{
                sendWarning("illegal part type (4) in boco");
		return FAIL;
	}
	else
	{
		diricletValObj->getAddress(&diricletVal);
	}

	int colWall;
	///////// Wall indices
    const coDoIntArr *wallObj = dynamic_cast<const coDoIntArr *>(setObj[5]);
	if (!wallObj)
	{
                sendWarning("illegal part type (5) in boco");
		return FAIL;
	}
	else
	{
		colWall = wallObj->getDimension(0);
		numWall = wallObj->getDimension(1);
		wall    = wallObj->getAddress();
		DumpIntArr("wall", wall, numWall, colWall);
	}

	int colBalance;
	///////// balance indices
    const coDoIntArr *balanceObj = dynamic_cast<const coDoIntArr *>(setObj[6]);
	if (!balanceObj)
	{
                sendWarning("illegal part type (6) in boco");
		return FAIL;
	}
	else
	{
		coDoIntArr *balanceObj = (coDoIntArr *) setObj[6];
		colBalance = balanceObj->getDimension(0);
		numBalance = balanceObj->getDimension(1);
		balance    = balanceObj->getAddress();
		DumpIntArr("balance", balance, numBalance, colBalance);
	}

	int colPress;
	if (numObj == 8)            // we have a pressure boundary condition!
	{
        const coDoIntArr *pressObj = dynamic_cast<const coDoIntArr *>(setObj[7]);
		///////// pressure indices
		if (!pressObj)
		{
			sendWarning("illegal part type (7) in boco");
			return FAIL;
		}
		else
		{
			colPress = pressObj->getDimension(0);
			numPress = pressObj->getDimension(1);
			press = pressObj->getAddress();
			DumpIntArr("press", press, numPress, colPress);
		}

		///////// pressure values
		/*if (strcmp(setObj[8]->getType(),"USTSDT"))
		{
			Covise::sendWarning("illegal part type (8) in boco");
			return FAIL;
		}
		else
		{
			coDoFloat *pressValObj = (coDoFloat *) setObj[8];
			pressValObj->getAddress(&pressVal);
			}*/

	}

	///////////////////////////// split it ...
	numParts = p_numPart->getValue();

	if (!zerbuf)
	{
		zerbuf = (int *)calloc(zerbuflen, sizeof(int));
	}

	// WARNING: Declaration is in the top of this file (static global)
	pboco = new coDistributedObject *[numParts+1];
	pgrid = new coDistributedObject *[numParts+1];
	for (i=0;i<=numParts;i++)
	{
		pboco[i] = NULL;
		pgrid[i] = NULL;
	}

	// ZER: does the real work (domain decomposition ...)
	// stores data in static global variables ... (with sendpartdata_())
#ifdef WIN32
	int ret = _chdir(p_dir->getValue());
#else
	int ret = chdir(p_dir->getValue());
#endif
	if(ret == -1)
	{
		sendWarning("failed to chdir to %s: %s", p_dir->getValue(), strerror(errno));
	}
	
	int writeFiles = p_writeFiles->getValue();
	
	mainn_(zerbuf, &zerbuflen, &numParts,
		   &nume ,connList, &nump,x,y,z,               // grid
		   &col_node, nodeInfo,                        // add. info for nodes
		   &col_elem, elemInfo,                        // add. info for elems
		   &colDiriclet, &numDiriclet,                 // bc: dirichlet
		   diricletIndex, diricletVal,
		   &colWall, &numWall, wall,                   // wall
		   &colBalance, &numBalance, balance,          // balance
		   &colPress, &numPress, press,                // pressure
		   &covise_run, &writeFiles);                               // covise is running

	p_distGrid->setCurrentObject( new coDoSet(p_distGrid->getObjName(),pgrid) );
	p_distBoco->setCurrentObject( new coDoSet(p_distBoco->getObjName(),pboco) );

	//time
#ifndef WIN32
	gettimeofday (&time2, NULL);
	double sec = double (time2.tv_sec - time1.tv_sec);
	double usec = 0.000001 * (time2.tv_usec - time1.tv_usec);
	printf("DomainDecomposition Laufzeit: %5.2lf Sekunden\n", sec+usec);
#endif

	return SUCCESS;
}

#ifndef YAC
coDistributedObject *floatDO(const char *name, const float *data, int numElem)
#else
coDistributedObject *floatDO(coObjInfo name, const float *data, int numElem)
#endif
{
	coDoFloat *res = new coDoFloat(name,numElem);
	float *shmData;
	res->getAddress(&shmData);
	memcpy(shmData,data,numElem*sizeof(float));
	return res;
}

#ifndef YAC
coDistributedObject *intDO(const char *name, const int *data, int numElem)
#else
coDistributedObject *intDO(coObjInfo name, const int *data, int numElem)
#endif
{
	coDoIntArr *res = new coDoIntArr(name,1,&numElem);
	int *shmData;
	res->getAddress(&shmData);
	memcpy(shmData,data,numElem*sizeof(int));
	return res;
}


#define  GNUM  12
void sendgeodata_module_(int *igeb, int *npoin_ges, int *nelem_ges, int *knmax_num, int *elmax_num,
				  int *nkn, int *nel, int *ncd, int *nkd, float *cov_coord,
				  int *cov_lnods, int *cov_lnods_num, int *cov_lnods_proz,
				  int *cov_coord_num, int *cov_coord_joi, int *cov_lnods_joi,
				  int *cov_coord_mod, int *cov_lnods_mod, int *cov_coord_proz)
{
#ifndef YAC
	char buf[256];
#endif
	fprintf(stderr, "Gebiet = %d\n", *igeb);
	fprintf(stderr, "npoin_ges = %d, nelem_ges = %d, knmax_num = %d\n",
			*npoin_ges, *nelem_ges, *knmax_num);
	fprintf(stderr, "elmax_num = %d, nkn = %d, nel = %d\n",
			*elmax_num, *nkn, *nel);

	coDistributedObject *geo[GNUM];

	//global data per node
	int i=9;
#ifndef YAC
	sprintf(buf,"%s_%d_global"     ,pgridbase,*igeb);
	coDoIntArr *globalData = new coDoIntArr(buf,1,&i);
#else
	coDoIntArr *globalData = new coDoIntArr(p_distGrid->getNewObjectInfo(),1,&i);
#endif
	int *data=globalData->getAddress();
	data[0]=*igeb;
	data[1]=*npoin_ges;
	data[2]=*nelem_ges;
	data[3]=*knmax_num;
	data[4]=*elmax_num;
	data[5]=*nkn;
	data[6]=*nel;
	data[7]=*ncd;
	data[8]=*nkd;
	geo[0] = globalData;
#ifndef YAC
	sprintf(buf,"%s_%d_cov_coord"     ,pgridbase,*igeb); geo[1]=floatDO(buf,cov_coord,*nkn * *ncd);
	sprintf(buf,"%s_%d_cov_lnods"     ,pgridbase,*igeb); geo[2]=  intDO(buf,cov_lnods,*nel * *nkd);
	sprintf(buf,"%s_%d_cov_lnods_num" ,pgridbase,*igeb); geo[3]=  intDO(buf,cov_lnods_num,*nel);
	sprintf(buf,"%s_%d_cov_lnods_proz",pgridbase,*igeb); geo[4]=  intDO(buf,cov_lnods_proz,*nel);
	sprintf(buf,"%s_%d_cov_coord_num" ,pgridbase,*igeb); geo[5]=  intDO(buf,cov_coord_num,*nkn);
	sprintf(buf,"%s_%d_cov_coord_joi" ,pgridbase,*igeb); geo[6]=  intDO(buf,cov_coord_joi,*nkn);
	sprintf(buf,"%s_%d_cov_lnods_joi" ,pgridbase,*igeb); geo[7]=  intDO(buf,cov_lnods_joi,*nel);
	sprintf(buf,"%s_%d_cov_coord_mod" ,pgridbase,*igeb); geo[8]=  intDO(buf,cov_coord_mod,*nkn);
	sprintf(buf,"%s_%d_cov_lnods_mod" ,pgridbase,*igeb); geo[9]=  intDO(buf,cov_lnods_mod,*nel);
	sprintf(buf,"%s_%d_cov_coord_proz",pgridbase,*igeb); geo[10]= intDO(buf,cov_coord_proz,*nkn);
#else
        geo[1] = floatDO(p_distGrid->getNewObjectInfo(),cov_coord,*nkn * *ncd);
        geo[2] = intDO(p_distGrid->getNewObjectInfo(),cov_lnods,*nel * *nkd);
        geo[3] = intDO(p_distGrid->getNewObjectInfo(),cov_lnods_num,*nel);
        geo[4] = intDO(p_distGrid->getNewObjectInfo(),cov_lnods_proz,*nel);
        geo[5] = intDO(p_distGrid->getNewObjectInfo(),cov_coord_num,*nkn);
        geo[6] = intDO(p_distGrid->getNewObjectInfo(),cov_coord_joi,*nkn);
        geo[7] = intDO(p_distGrid->getNewObjectInfo(),cov_lnods_joi,*nel);
        geo[8] = intDO(p_distGrid->getNewObjectInfo(),cov_coord_mod,*nkn);
        geo[9] = intDO(p_distGrid->getNewObjectInfo(),cov_lnods_mod,*nel);
        geo[10] = intDO(p_distGrid->getNewObjectInfo(),cov_coord_proz,*nkn);
#endif
	geo[11]=NULL;
#ifndef YAC
	for (i=0;i<GNUM-1;i++)
	{
		if(!geo[i]->checkObject())
		{
			fprintf(stderr," Fehler!!!!!!!!!!!!!!!!!!!!!!!!%d\n",i);
		}
	}
	sprintf(buf,"%s_%d"     ,pgridbase,*igeb);
	pgrid[*igeb-1] = new coDoSet(buf,geo);
#else
        pgrid[*igeb-1] = new coDoSet(p_distGrid->getNewObjectInfo(), geo);
#endif
	// delete
	for (i=0;i<GNUM;i++)
		delete geo[i];
}


#define  RNUM  14
void sendrbedata_module_(int *igeb,int *nrbpo_geb,int *nwand_geb,int *npres_geb,int *nsyme_geb,
				  int *nconv_geb,int *nrbknie,
				  int *cov_displ_kn,int *cov_displ_typ,
				  int *cov_wand_el,int *cov_wand_kn,int *cov_wand_num,
				  int *cov_pres_el,int *cov_pres_kn,int *cov_pres_num,
				  int *cov_conv_el,int *cov_conv_kn,int *cov_conv_num,
				  float *cov_displ_wert, int *reicheck)
{
#ifndef YAC
	char buf[256];
#endif
	coDistributedObject *boc[RNUM];

	// all data for every CPU: igeb = 1 ... numParts
	// Parameter: nrbpo_geb,nwand_geb,npres_geb,nsyme_geb,nconv_geb,nrbknie
	if (*reicheck != 999)
		fprintf(stderr, "Schrott: reicheck != 999\n");

	//global data per node
	int i=6;
#ifndef YAC
	sprintf(buf,"%s_%d_global"     ,pbocobase,*igeb);
	coDoIntArr *globalData = new coDoIntArr(buf,1,&i);
#else
	coDoIntArr *globalData = new coDoIntArr(p_distGrid->getNewObjectInfo(),1,&i);
#endif
	int *data=globalData->getAddress();
	data[0]=*nrbpo_geb;
	data[1]=*nwand_geb;
	data[2]=*npres_geb;
	data[3]=*nsyme_geb;
	//*nconv_geb=0;
	data[4]=*nconv_geb;
	data[5]=*nrbknie;
	boc[0] = globalData;
#ifndef YAC
	sprintf(buf,"%s_%d_cov_displ_kn"  ,pbocobase,*igeb); boc[1] =intDO  (buf,cov_displ_kn,*nrbpo_geb);
	sprintf(buf,"%s_%d_cov_displ_ty"  ,pbocobase,*igeb); boc[2] =intDO  (buf,cov_displ_typ,*nrbpo_geb);
	sprintf(buf,"%s_%d_cov_wand_el"   ,pbocobase,*igeb); boc[3] =intDO  (buf,cov_wand_el,*nwand_geb);
	sprintf(buf,"%s_%d_cov_wand_kn"   ,pbocobase,*igeb); boc[4] =intDO  (buf,cov_wand_kn,*nwand_geb * *nrbknie);
	sprintf(buf,"%s_%d_cov_wand_num"  ,pbocobase,*igeb); boc[5] =intDO  (buf,cov_wand_num,*nwand_geb);
	sprintf(buf,"%s_%d_cov_pres_el"   ,pbocobase,*igeb); boc[6] =intDO  (buf,cov_pres_el,*npres_geb);
	sprintf(buf,"%s_%d_cov_pres_kn"   ,pbocobase,*igeb); boc[7] =intDO  (buf,cov_pres_kn,*npres_geb * *nrbknie);
	sprintf(buf,"%s_%d_cov_pres_num"  ,pbocobase,*igeb); boc[8] =intDO  (buf,cov_pres_num,*npres_geb);
	sprintf(buf,"%s_%d_cov_conv_el"   ,pbocobase,*igeb); boc[9] =intDO  (buf,cov_conv_el,*nconv_geb);
	sprintf(buf,"%s_%d_cov_conv_kn"   ,pbocobase,*igeb); boc[10]=intDO  (buf,cov_conv_kn,*nconv_geb * *nrbknie);
	sprintf(buf,"%s_%d_cov_conv_num"  ,pbocobase,*igeb); boc[11]=intDO  (buf,cov_conv_num,*nconv_geb);
	sprintf(buf,"%s_%d_cov_displ_wert",pbocobase,*igeb); boc[12]=floatDO(buf,cov_displ_wert,*nrbpo_geb);
	boc[13]=NULL;
	for (i=0;i<RNUM-1;i++)
	{
   		if(!boc[i]->checkObject())
		{
			fprintf(stderr," Fehler2!!!!!!!!!!!!!!!!!!!!!!!! %d\n",i);
   		}
	}

	sprintf(buf,"%s_%d"     ,pbocobase,*igeb);
	pboco[*igeb-1] = new coDoSet(buf,boc);
#else
        boc[1] = intDO(p_distGrid->getNewObjectInfo(),cov_displ_kn,*nrbpo_geb);
        boc[2] = intDO(p_distGrid->getNewObjectInfo(),cov_displ_typ,*nrbpo_geb);
        boc[3] = intDO(p_distGrid->getNewObjectInfo(),cov_wand_el,*nwand_geb);
        boc[4] = intDO(p_distGrid->getNewObjectInfo(),cov_wand_kn,*nwand_geb * *nrbknie);
        boc[5] = intDO(p_distGrid->getNewObjectInfo(),cov_wand_num,*nwand_geb);
        boc[6] = intDO(p_distGrid->getNewObjectInfo(),cov_pres_el,*npres_geb);
        boc[7] = intDO(p_distGrid->getNewObjectInfo(),cov_pres_kn,*npres_geb * *nrbknie);
        boc[8] = intDO(p_distGrid->getNewObjectInfo(),cov_pres_num,*npres_geb);
        boc[9] = intDO(p_distGrid->getNewObjectInfo(),cov_conv_el,*nconv_geb);
        boc[10] = intDO(p_distGrid->getNewObjectInfo(),cov_conv_kn,*nconv_geb * *nrbknie);
        boc[11] = intDO(p_distGrid->getNewObjectInfo(),cov_conv_num,*nconv_geb);
        boc[12] = floatDO(p_distGrid->getNewObjectInfo(),cov_displ_wert,*nrbpo_geb);
        boc[13] = NULL;
        pboco[*igeb-1] = new coDoSet(p_distGrid->getNewObjectInfo(),boc);
#endif
	// delete
	for (i=0;i<RNUM-1;i++)
		delete boc[i];

}


void DumpIntArr(const char *s, int *a, int num, int col)
{
	int i, j;

	return;
	printf("%s: num=%d, col=%d\n", s, num, col);
	for (i=0; i< num*col; i+=col)
	{
		printf("\t");
		for (j=0; j<col; j++)
			printf("%5d  ", a[i+j]);
		printf("\n");
	}
}

#ifdef YAC
void DomainDecomposition::paramChanged(coParam *param) {

   (void) param;
}
#endif


MODULE_MAIN(VISiT, DomainDecomposition)
