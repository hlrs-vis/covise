// create computational grid and boundary conditions for covise module
// and set attributes to be passed as params to FENFLOSS

#include "RadialRunner.h"
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>
#include <General/include/log.h>

#define ROTLABEL  11
#define FIXLABEL  10
#define PRESLABEL 20

#define INLET		100
#define OUTLET		200
#define ALLWALL		150
#define PSSURFACE	202
#define SSSURFACE	202

#define RRINLET		101
#define RROUTLET	102

#define PSLEPERIOD	110
#define SSLEPERIOD	120
#define PSTEPERIOD	110
#define SSTEPERIOD	120

void RadialRunner::CreateGrid(void)
{
	int i, nume, numn, noffset=0, eoffset=0;
	char buf[500];
	const char *err_msg;

	coDoPolygons *poly;

	dprintf(2," creating grid ...\n");
	rrg = CreateRR_Mesh(geo->rr);
	if(!gridInitialized) {
		gridInitialized = 1;
		RadialRunner::Grid2CtrlPanel();
	}
	//RadialRunner::Grid2CtrlPanel();
	// this is ugly, but somehow the booleans are set to FALSE if we update
	// the whole grid params with Grid2CtrlPanel().
	// since cledis might be changed in CreateRR_GridRegions
	// we need to update here
	p_CircumfDisLe->setValue((float)rrg->cledis, rrg->clebias,
							 (float)rrg->clebias_type);
	dprintf(2," ... grid created ...\n");
	if((err_msg = GetLastGridErr())) sendError("%s",err_msg);
	else {
		sendInfo(" Computational grid created: %d nodes, %d elements.",
                                rrg->n->num, rrg->e->nume);
		dprintf(2," ************************************\n%s\n",buf);

// **************************************************
		// nodes and elements
		// **************************************************
		// show complete grid or only some layers?
		if(p_ShowComplete->getValue()) {
			nume = rrg->e->nume;
			numn = rrg->n->num;
		}
		else {
			eoffset = rrg->e->nume/(rrg->ge_num-1);
			noffset = rrg->n->num /(rrg->ge_num);
			numn = (p_GridLayers->getValue(1) -
					p_GridLayers->getValue(0)+2) * noffset;
			nume = (p_GridLayers->getValue(1) -
					p_GridLayers->getValue(0)+1) * eoffset;
		}
		dprintf(2,"nume = %d, numn = %d, noffset = %d, eoffset = %d\n",
				nume, numn, noffset, eoffset);

		// create grid outport object
		coDoUnstructuredGrid *unsGrd =
			new coDoUnstructuredGrid(grid->getObjName(),
									nume,8*nume,numn,1);
		int *elem,*conn,*type;
		float *xc,*yc,*zc;

		unsGrd->getAddresses(&elem,&conn,&xc,&yc,&zc);
		unsGrd->getTypeList(&type);

		// connectivity list
		int **RRgridConn = rrg->e->e;
		for (i=0;i<nume;i++) {
			*elem = 8*i;			   elem++;

			*conn = (*RRgridConn)[0];	conn++;
			*conn = (*RRgridConn)[1];	conn++;
			*conn = (*RRgridConn)[2];	conn++;
			*conn = (*RRgridConn)[3];	conn++;
			*conn = (*RRgridConn)[4];	conn++;
			*conn = (*RRgridConn)[5];	conn++;
			*conn = (*RRgridConn)[6];	conn++;
			*conn = (*RRgridConn)[7];	conn++;

			*type = TYPE_HEXAGON;	   type++;

			RRgridConn++;
		}

		// nodal coordinates
		if(p_ShowComplete->getValue())
			CreateRR_Grid4Covise(rrg->n,xc,yc,zc,0,rrg->n->num);
		else
			CreateRR_Grid4Covise(rrg->n,xc,yc,zc,
								 (p_GridLayers->getValue(0)-1)*noffset,
								 (p_GridLayers->getValue(1)+1)*noffset);

		// set out port
		grid->setCurrentObject(unsGrd);

		// set attributes
		char paramtext[80];
		char numberofblades[80];
		int  rotdir=1;
		if(p_Pump->getValue()) rotdir=-1;
#ifndef YAC
		snprintf(paramtext, 80, "%ld", p_NumberOfBlades->getValue());
#else
		snprintf(paramtext, 80, "%d", p_NumberOfBlades->getValue());
#endif
		unsGrd->addAttribute(M_NUMBER_OF_BLADES, paramtext);

		snprintf(paramtext, 80, "%16.6f", rotdir*p_DRevolut->getValue());
		unsGrd->addAttribute(M_DESIGN_N, paramtext);

		snprintf(paramtext, 80, "%d",1);
		unsGrd->addAttribute("periodic", paramtext);

		snprintf(paramtext, 80, "%d",1);
		unsGrd->addAttribute("rotating", paramtext);

		sprintf(paramtext,"115,wand_omega,%d,0.0,0.0,%.4f",11,
				rotdir*p_DRevolut->getValue()*M_PI/30.);
		unsGrd->addAttribute("walltext",paramtext);

#ifndef YAC
		sprintf(paramtext,"111,nomatch,110,120,perio_rota,%ld,3", p_NumberOfBlades->getValue());
		sprintf(numberofblades,"%ld", p_NumberOfBlades->getValue());
#else
		sprintf(paramtext,"111,nomatch,110,120,perio_rota,%d,3", p_NumberOfBlades->getValue());
		sprintf(numberofblades,"%d", p_NumberOfBlades->getValue());
#endif

		unsGrd->addAttribute("periotext",paramtext);
		unsGrd->addAttribute("numblades",numberofblades);
		


    // **************************************************
		// Boundary conditions
		// **************************************************
		// boundary condition lists
		int   num_node, num_corn, num_poly;
		int   *corners, *poly_list;
		float *xb, *yb, *zb;
		int   nset;
		void **elset;
		// 1. Cells at entry
		num_node = rrg->inlet->num;
		num_corn = rrg->einlet->nume*NPE_BC;
		num_poly = rrg->einlet->nume;
		poly = new coDoPolygons(bcin->getObjName(),
							   num_node,num_corn,num_poly);
		poly->getAddresses(&xb, &yb, &zb, &corners, &poly_list);
		elset  = Create_ElemSet(&nset,rrg->einlet,NULL);
		if(CreateRR_BClistbyElemset(rrg->n,(struct Element **)elset, nset,
									xb,yb,zb, corners, poly_list,
									num_corn, num_poly, num_node, buf)) {
			sendError("%s",buf);
		}

		FreeElemSetPtr();
		poly->addAttribute("vertexOrder","1");
		bcin->setCurrentObject(poly);

		// 2. Cells at outlet
		num_node = rrg->outlet->num;
		num_corn = rrg->eoutlet->nume*NPE_BC;
		num_poly = rrg->eoutlet->nume;
		poly = new coDoPolygons(bcout->getObjName(),
							   num_node,num_corn,num_poly);
		poly->getAddresses(&xb, &yb, &zb, &corners, &poly_list);
		elset  = Create_ElemSet(&nset,rrg->eoutlet,NULL);
		if(CreateRR_BClistbyElemset(rrg->n,(struct Element **)elset, nset,
									xb,yb,zb, corners, poly_list,
									num_corn, num_poly, num_node, buf)) {
			sendError("%s",buf);
		}

		FreeElemSetPtr();
		poly->addAttribute("vertexOrder","1");
		bcout->setCurrentObject(poly);

		// 3. Cells on wall, all walls!
		num_node = rrg->n->num;
		num_poly = rrg->frictless->nume + rrg->shroud->nume +
			rrg->shroudext->nume + rrg->wall->nume;
		num_corn = num_poly*NPE_BC;
		elset  = Create_ElemSet(&nset,rrg->frictless, rrg->shroud,
								rrg->shroudext,
								rrg->wall,NULL);
		poly = new coDoPolygons(bcwall->getObjName(),
							   num_node,num_corn,num_poly);		
		poly->getAddresses(&xb, &yb, &zb, &corners, &poly_list);
		if(CreateRR_BClistbyElemset(rrg->n,(struct Element **)elset, nset,
									xb,yb,zb, corners, poly_list,
									num_corn, num_poly, num_node, buf)) {
			sendError("%s",buf);
		}

		FreeElemSetPtr();
		poly->addAttribute("vertexOrder","1");
		bcwall->setCurrentObject(poly);

		// 4. Cells on blade surface
		/*num_node = rrg->psnod->num + rrg->ssnod->num;
		num_poly = rrg->psblade->nume + rrg->ssblade->nume;
		num_corn = num_poly*NPE_BC;
		elset  = Create_ElemSet(&nset,rrg->psblade,rrg->ssblade,NULL);
		poly = new coDoPolygons(bcblade->getObjName(),
							   num_node,num_corn,num_poly);
		poly->getAddresses(&xb, &yb, &zb, &corners, &poly_list);

		if(CreateRR_BClistbyElemset(rrg->n,(struct Element **)elset, nset,
									xb,yb,zb, corners, poly_list,
									num_corn, num_poly, num_node, buf)) {
			sendError(buf);
		}

		FreeElemSetPtr();
		poly->addAttribute("vertexOrder","1");
		bcblade->setCurrentObject(poly);*/

		// 5. Cells at periodic borders
		num_node = rrg->psle->num + rrg->ssle->num +
			       rrg->pste->num + rrg->sste->num;
		num_poly = rrg->psleperiodic->nume + rrg->ssleperiodic->nume +
			       rrg->psteperiodic->nume + rrg->ssteperiodic->nume;
		num_corn = num_poly*NPE_BC;
		poly = new coDoPolygons(bcperiodic->getObjName(),
							   num_node,num_corn,num_poly);
		poly->getAddresses(&xb, &yb, &zb, &corners, &poly_list);
		elset  = Create_ElemSet(&nset,rrg->psleperiodic,
								rrg->ssleperiodic,rrg->psteperiodic,
								rrg->ssteperiodic,NULL);
		if(CreateRR_BClistbyElemset(rrg->n,(struct Element **)elset, nset,
									xb,yb,zb, corners, poly_list,
									num_corn, num_poly, num_node, buf)) {
			sendError("%s",buf);
		}

		FreeElemSetPtr();

		poly->addAttribute("vertexOrder","1");
		bcperiodic->setCurrentObject(poly);

		// this is the clumsy version ...
		// it works but yields multiple duplications
		// of the WHOLE node coordinate list!
		// this is only done temporarily for the blade surfaces
		// since otherwise we cannot show pressure results on the
		// blades since the order is changed!
		// we have to pass a vector including the ids of the blade
		// nodes and pick the respective nodes in the FENFLOSS-Module
		// and send the values via an extra port!
		int i_corn, i_poly;
		num_node = rrg->n->num;
		num_poly = rrg->psblade->nume + rrg->ssblade->nume;
		num_corn = num_poly*NPE_BC;
		poly = new coDoPolygons(bcblade->getObjName(),
							   num_node,num_corn,num_poly);		
		poly->getAddresses(&xb, &yb, &zb, &corners, &poly_list);
		CreateRR_Grid4Covise(rrg->n,xb,yb,zb,0,rrg->n->num);
		i_corn = i_poly = 0;
		if(CreateRR_BClist4Covise(rrg->psblade, corners, poly_list,
								  &i_corn, &i_poly,
								  num_corn, num_poly, buf)) {
			sendError("%s",buf);
		}
		if(CreateRR_BClist4Covise(rrg->ssblade, corners, poly_list,
								  &i_corn, &i_poly,
								  num_corn, num_poly, buf)) {
			sendError("%s",buf);
		}
		poly->addAttribute("vertexOrder","1");
		bcblade->setCurrentObject(poly);

		// we had several additional info, we should send to the
		// Domaindecomposition:
		//   0. number of columns per info
		//   1. type of node
		//   2. type of element
		//   3. list of nodes with bc (a node may appear more than one time)
		//   4. corresponding type to 3.
		//   5. wall
		//   6. balance
		//   7. pressure
		//   8. NULL

		coDistributedObject *partObj[10];

		int size[2];
		int i, j, num;
		int *data;
		float *bPtr;

		//   0. number of columns per info
#ifndef YAC
		char name[256];
		const char *basename = boco->getObjName();
		sprintf(name,"%s_colinfo",basename);
#else
                coObjInfo basename = boco->getNewObjectInfo();
                coObjInfo name = boco->getNewObjectInfo();
#endif
		size[0] = 6;
		size[1] = 0;
		coDoIntArr *colInfo = new coDoIntArr(name,1,size);
		data = colInfo->getAddress();
		data[0] = RG_COL_NODE;                      // (=2)
		data[1] = RG_COL_ELEM;                      // (=2)
		data[2] = RG_COL_DIRICLET;                  // (=2)
		data[3] = RG_COL_WALL;                      // (=7)
		data[4] = RG_COL_BALANCE;                   // (=7)
		data[5] = RG_COL_PRESS;                     // (=6)
		partObj[0]=colInfo;

		//   1. type of node
		//   this is the model number of the part!
		//   it has to be introduced as a parametre to be set != 0
#ifndef YAC
		sprintf(name,"%s_nodeinfo",basename);
#else
                name = boco->getNewObjectInfo();
#endif
		size[0] = RG_COL_NODE;
		size[1] = rrg->n->num;
		coDoIntArr *nodeInfo = new coDoIntArr(name,2,size);
		data = nodeInfo->getAddress();
		for (i = 0; i < rrg->n->num; i++) {
			*data++ = rrg->n->n[i]->id;
			*data++ = 0;
		}
		partObj[1]=nodeInfo;

		//   2. type of element
		//   elements are supposed to be numbered according to their
		//   generation index (id = index+1)!
#ifndef YAC
		sprintf(name,"%s_eleminfo",basename);
#else
                name = boco->getNewObjectInfo();
#endif
		size[0] = RG_COL_ELEM;
		size[1] = rrg->e->nume;
		coDoIntArr *elemInfo = new coDoIntArr(name, 2, size);
		data = elemInfo->getAddress();
		for (i = 0; i < rrg->e->nume; i++) {
			*data++ = i+1;
			*data++ = 0;                             // same comment ;-)
		}
		partObj[2]=elemInfo;

		// maybe there are no inlet bcs
#ifndef YAC
		sprintf(name,"%s_diricletNodes",basename);
#else
                name = boco->getNewObjectInfo();
#endif
		if(rrg->bcval) {
			//   3. list of nodes with bc
			//      and its types
			size [0] = RG_COL_DIRICLET;
			size [1] = 5*rrg->inlet->num;
			coDoIntArr *diricletNodes = new coDoIntArr(name, 2, size);
			data = diricletNodes->getAddress();

			//   4. corresponding value to 3.
#ifndef YAC
			sprintf(name,"%s_diricletValue",basename);
#else
                        name = boco->getNewObjectInfo();
#endif
			coDoFloat *diricletValues
				= new coDoFloat(name, 5*rrg->inlet->num);
			diricletValues->getAddress(&bPtr);
			for(i = 0; i < rrg->inlet->num; i++)
			{
				// 5 diriclet bc values in 3D!
				for(j = 0; j < 5; j++) {
					*data++ = rrg->inlet->list[i]+1;
					*data++ = j+1;
					*bPtr++ = rrg->bcval[i][j];
				}
			}
			partObj[3] = diricletNodes;
			partObj[4] = diricletValues;
		}
		else {
			size [0] = RG_COL_DIRICLET;
			size [1] = 0;
			coDoIntArr *diricletNodes = new coDoIntArr(name, 2, size);
			data = diricletNodes->getAddress();
#ifndef YAC
			sprintf(name,"%s_diricletValue",basename);
#else
                        name = boco->getNewObjectInfo();
#endif
			coDoFloat *diricletValues
				= new coDoFloat(name, 1);
			diricletValues->getAddress(&bPtr);
			// no data values (no diriclet nodes)
			// dummy
			*bPtr = -99.0;

			sendInfo(" No diriclet boundary conditions set!\n");
			partObj[3] = diricletNodes;
			partObj[4] = diricletValues;
		}

		//   5. wall
		num =  rrg->shroudext->nume + rrg->wall->nume +
			   rrg->frictless->nume + rrg->shroud->nume;
	
#ifndef YAC
		sprintf(name,"%s_wall",basename);
#else
                name = boco->getNewObjectInfo();
#endif
		size[0] = RG_COL_WALL;
		size[1] = num;
		coDoIntArr *faces = new coDoIntArr(name, 2, size);
		data = faces->getAddress();

		// shroud inlet extension may be fixed!
		if(!rrg->rot_ext) SetElement(&data, rrg->shroudext, FIXLABEL);
		else SetElement(&data, rrg->shroudext, ROTLABEL);
		SetElement(&data, rrg->wall, ROTLABEL);
		SetElement(&data, rrg->frictless, ROTLABEL);
		SetElement(&data, rrg->shroud, ROTLABEL);

		// fl debug
		data = faces->getAddress();
		for(int i= 0; i < num*RG_COL_WALL; i++) {
			dprintf(4," CreateGrid(): wall: i = %7d, %8d\n",i,data[i]);
		}

		partObj[5]=faces;

		//   6. balance
#ifndef YAC
		sprintf(name,"%s_balance",basename);
#else
                name = boco->getNewObjectInfo();
#endif
		num = rrg->einlet->nume + rrg->eoutlet->nume +
			  rrg->psleperiodic->nume + rrg->ssleperiodic->nume +
			  rrg->psteperiodic->nume + rrg->ssteperiodic->nume +
			  rrg->psblade->nume + rrg->ssblade->nume +
			  rrg->rrinlet->nume + rrg->rroutlet->nume + 
			  rrg->shroudext->nume + rrg->wall->nume +
			  rrg->frictless->nume + rrg->shroud->nume;

		size[0] = RG_COL_BALANCE;
		size[1] = num;
		coDoIntArr *balance = new coDoIntArr(name, 2, size);
		data = balance->getAddress();

		SetElement(&data, rrg->einlet, INLET);
		SetElement(&data, rrg->eoutlet, OUTLET);
		SetElement(&data, rrg->psleperiodic, PSLEPERIOD);
		SetElement(&data, rrg->psteperiodic, PSTEPERIOD);
		SetElement(&data, rrg->ssleperiodic, SSLEPERIOD);
		SetElement(&data, rrg->ssteperiodic, SSTEPERIOD);
		SetElement(&data, rrg->psblade, PSSURFACE);
		SetElement(&data, rrg->ssblade, SSSURFACE);
		SetElement(&data, rrg->rrinlet, RRINLET);
		SetElement(&data, rrg->rroutlet, RROUTLET);
		SetElement(&data, rrg->shroudext, ALLWALL);
		SetElement(&data, rrg->wall, ALLWALL);
		SetElement(&data, rrg->frictless, ALLWALL);
		SetElement(&data, rrg->shroud, ALLWALL);

		partObj[6] = balance;

      //  7. pressure bc: outlet elements
#ifndef YAC
		sprintf(name,"%s_pressElems",basename);
#else
                name = boco->getNewObjectInfo();
#endif
		size[0] = RG_COL_PRESS;
		size[1] = rrg->eoutlet->nume;
		coDoIntArr *pressElems = new coDoIntArr(name, 2, size );
		data=pressElems->getAddress();

		SetElement(&data, rrg->eoutlet, PRESLABEL);

		partObj[7] = pressElems;

		// end
		partObj[8] = NULL;

		coDoSet *bcset = new coDoSet(basename,
                                             (coDistributedObject **)partObj);
		boco->setCurrentObject(bcset);

      /****************************************************************************/
  int ii;
  int jj;
  int counter;
  int mySize[2];
  int nNodesFace;
  int nodesRef[] = {1,2,4,3}; // nodes are ordered x-over!!
  
  nNodesFace = 4; // number of nodes per face
  coDistributedObject* boundaryElementFacesParts[17];
  coDoSet* boundaryElementFacesSet;
  
  //inlet-----------------------------------------------------------------------
  // create size vector for coDoIntArr
  mySize[0] = 1;
  mySize[1] = rrg->einlet->nume*nNodesFace;
  coDoIntArr* inlet_element_faces = new coDoIntArr("inlet element faces",2,mySize);
  data = inlet_element_faces->getAddress();
  cout << "inlet" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = 1;
  for (ii = 0; ii < rrg->einlet->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->einlet->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  inletElementFaces->setCurrentObject(inlet_element_faces);
  
  //outlet----------------------------------------------------------------------
  // create size vector for coDoIntArr
  mySize[0] = 1;
  mySize[1] = rrg->eoutlet->nume*nNodesFace;
  //create intArr
  coDoIntArr* outlet_element_faces = new coDoIntArr("outlet element faces",2,mySize);
  data = outlet_element_faces->getAddress();
  cout << "outlet" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->eoutlet->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->eoutlet->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }  
//  outletElementFaces->setCurrentObject(outlet_element_faces);

  //shroud----------------------------------------------------------------------
  //create size vector for coDoIntArr
  mySize[0] = 1;
  mySize[1] = rrg->shroud->nume*nNodesFace;
  //create intArr
  coDoIntArr* shroud_element_faces = new coDoIntArr("shroud element faces",2,mySize);
  data = shroud_element_faces->getAddress();  
  cout << "shroud" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->shroud->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->shroud->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  shroudElementFaces->setCurrentObject(shroud_element_faces);
  
  //shroudext-------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->shroudext->nume*nNodesFace;
  coDoIntArr* shroudExt_element_faces = new coDoIntArr("shroudExt element faces",2,mySize);
  data = shroudExt_element_faces->getAddress();   
  cout << "shroudext" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->shroudext->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->shroudext->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  shroudExtElementFaces->setCurrentObject(shroudExt_element_faces);

  //frictless-------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->frictless->nume*nNodesFace;
  coDoIntArr* frictless_element_faces = new coDoIntArr("frictless element faces",2,mySize);
  data = frictless_element_faces->getAddress();   
  cout << "frictless" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->frictless->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->frictless->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }  
//  frictlessElementFaces->setCurrentObject(frictless_element_faces);

  //psblade---------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->psblade->nume*nNodesFace;
  coDoIntArr* psblade_element_faces = new coDoIntArr("psblade element faces",2,mySize);
  data = psblade_element_faces->getAddress();   
  cout << "psblade" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->psblade->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->psblade->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  psbladeElementFaces->setCurrentObject(psblade_element_faces);


  //ssblade----------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->ssblade->nume*nNodesFace;
  coDoIntArr* ssblade_element_faces = new coDoIntArr("ssblade element faces",2,mySize);
  data = ssblade_element_faces->getAddress();     
  cout << "ssblade" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->ssblade->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->ssblade->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  ssbladeElementFaces->setCurrentObject(ssblade_element_faces);

  //wall------------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->wall->nume*nNodesFace;  
  coDoIntArr* wall_element_faces = new coDoIntArr("wall element faces",2,mySize);
  data = wall_element_faces->getAddress();    
  cout << "wall" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->wall->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->wall->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }  
//  wallElementFaces->setCurrentObject(wall_element_faces);
  
  //ssleperiodic----------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->ssleperiodic->nume*nNodesFace;  
  coDoIntArr* ssleperiodic_element_faces = new coDoIntArr("ssleperiodic element faces",2,mySize);
  data = ssleperiodic_element_faces->getAddress();    
  cout << "ssleperiodic" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->ssleperiodic->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->ssleperiodic->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  ssleperiodicElementFaces->setCurrentObject(ssleperiodic_element_faces);

  //psleperiodic----------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->psleperiodic->nume*nNodesFace;  
  coDoIntArr* psleperiodic_element_faces = new coDoIntArr("psleperiodic element faces",2,mySize);
  data = psleperiodic_element_faces->getAddress();    
  cout << "psleperiodic" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->psleperiodic->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->psleperiodic->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  psleperiodicElementFaces->setCurrentObject(psleperiodic_element_faces);

  //ssteperiodic----------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->ssteperiodic->nume*nNodesFace;  
  coDoIntArr* ssteperiodic_element_faces = new coDoIntArr("ssteperiodic element faces",2,mySize);
  data = ssteperiodic_element_faces->getAddress();    
  cout << "ssteperiodic" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->ssteperiodic->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->ssteperiodic->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  ssteperiodicElementFaces->setCurrentObject(ssteperiodic_element_faces);

  //psteperiodic----------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->psteperiodic->nume*nNodesFace;  
  coDoIntArr* psteperiodic_element_faces = new coDoIntArr("psteperiodic element faces",2,mySize);
  data = psteperiodic_element_faces->getAddress();    
  cout << "psteperiodic" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->psteperiodic->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->psteperiodic->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }
//  psteperiodicElementFaces->setCurrentObject(psteperiodic_element_faces);

  //rrinlet---------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->rrinlet->nume*nNodesFace;  
  coDoIntArr* rrinlet_element_faces = new coDoIntArr("rrinlet element faces",2,mySize);
  data = rrinlet_element_faces->getAddress();    
  cout << "rrinlet" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->rrinlet->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->rrinlet->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }

  //rroutlet---------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->rroutlet->nume*nNodesFace;  
  coDoIntArr* rroutlet_element_faces = new coDoIntArr("rroutlet element faces",2,mySize);
  data = rroutlet_element_faces->getAddress();    
  cout << "rroutlet" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->rroutlet->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->rroutlet->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }  

  //hubAll---------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->hubAll->nume*nNodesFace;  
  coDoIntArr* hubAll_element_faces = new coDoIntArr("hubAll element faces",2,mySize);
  data = hubAll_element_faces->getAddress();    
  cout << "hubAll" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->hubAll->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->hubAll->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  }  

  //shroudAll---------------------------------------------------------------------
  mySize[0] = 1;
  mySize[1] = rrg->shroudAll->nume*nNodesFace;  
  coDoIntArr* shroudAll_element_faces = new coDoIntArr("shroudAll element faces",2,mySize);
  data = shroudAll_element_faces->getAddress();    
  cout << "shroudAll" << endl;
  counter = 0;
  *(data+counter) = nNodesFace;
  counter = counter + 1;
  for (ii = 0; ii < rrg->shroudAll->nume; ii++){
    for(jj = 0; jj < nNodesFace; jj++){
      *(data+counter) = rrg->shroudAll->e[ii][nodesRef[jj]];
      counter = counter + 1;
    }
  } 
  
  boundaryElementFacesParts[0] = inlet_element_faces;
  boundaryElementFacesParts[1] = outlet_element_faces;
  boundaryElementFacesParts[2] = shroud_element_faces;
  boundaryElementFacesParts[3] = shroudExt_element_faces;
  boundaryElementFacesParts[4] = frictless_element_faces;
  boundaryElementFacesParts[5] = psblade_element_faces;
  boundaryElementFacesParts[6] = ssblade_element_faces;
  boundaryElementFacesParts[7] = wall_element_faces;
  boundaryElementFacesParts[8] = ssleperiodic_element_faces;
  boundaryElementFacesParts[9] = psleperiodic_element_faces;
  boundaryElementFacesParts[10] = ssteperiodic_element_faces;
  boundaryElementFacesParts[11] = psteperiodic_element_faces;
  boundaryElementFacesParts[12] = rrinlet_element_faces;
  boundaryElementFacesParts[13] = rroutlet_element_faces;
  boundaryElementFacesParts[14] = hubAll_element_faces;
  boundaryElementFacesParts[15] = shroudAll_element_faces;
  boundaryElementFacesParts[16] = NULL;
  boundaryElementFacesSet = new coDoSet((char*) boundaryElementFaces->getObjName(), (coDistributedObject **) boundaryElementFacesParts);
  boundaryElementFaces->setCurrentObject(boundaryElementFacesSet);
  
// **************************************************
		// we don't need the old grid data anymore ... so, dump it!
		// **************************************************
		FreeRRGridMesh(rrg);
	}											// no error
}
