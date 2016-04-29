#include "writeToCgns.h"

#ifndef CGNS_ENUMV
//#define CGNS_ENUMV(v) CG_##v
#define CGNS_ENUMV(v) v
typedef int cgsize_t;
#endif

//predefines
const int SUCCESS  = 1;
const int FAILURE = 0;

/* constructor */
writeToCgns::writeToCgns(int argc, char *argv[])
:coModule(argc, argv, "writeToCgns") {
   //input ports
   p_inputPort_grid = addInputPort("in_grid", "UnstructuredGrid", "computational grid");  
   p_inputPort_boundaryElementFaces = addInputPort("boundary_element_faces","coDoSet","boundary element faces");
   //decide if ports are required and initialize
   p_inputPort_grid->setRequired(1);
   p_inputPort_boundaryElementFaces->setRequired(1);

   //specify filename of cgns file with browser
   cgns_filebrowser = addFileBrowserParam("path_to_file", "filename of cgns file");
   cgns_filebrowser->setValue(".", "*.cgns/*");
}

/* override compute function */
int writeToCgns::compute(const char *) {
  //declarations
  //----------------------------------------------------------------------------
  char port[256];
  int nElem;
  int nConn;
  int nCoords;
  int index_file;
  int ier;
  //input ports and casts
  const coDistributedObject* obj;
  const coDoUnstructuredGrid* inGrid;
  const coDoIntArr* inInletElementNodes;
  const coDoIntArr* inOutletElementNodes;
  const coDoIntArr* inShroudElementNodes;
  const coDoIntArr* inShroudExtElementNodes;
  const coDoIntArr* inFrictlessElementNodes;
  const coDoIntArr* inPsbladeElementNodes;
  const coDoIntArr* inSsbladeElementNodes;
  const coDoIntArr* inWallElementNodes;
  const coDoIntArr* inSsleperiodicElementNodes;
  const coDoIntArr* inPsleperiodicElementNodes;
  const coDoIntArr* inSsteperiodicElementNodes;
  const coDoIntArr* inPsteperiodicElementNodes;
  const coDoIntArr* inRrinletElementNodes;
  const coDoIntArr* inRroutletElementNodes;
  const coDoIntArr* inHubAllElementNodes;
  const coDoIntArr* inShroudAllElementNodes;
  const coDoSet* inBoundaryElementFaces;
  //grid attributes
  int* tList;
  int* elem;
  int* conn;
  cgsize_t* intHelper;
  float* xCoord;
  float* yCoord;
  float* zCoord;
  int nPoly;
  int nCorn;
  int* corn;
  int* poly;
  int ii;
  int base_i;
  int baseCFX_i;
  int zone_i;
  int zoneCFX_i;
  int coord_i;
  int section_i;
  int tmpCounter;
  int gElemCounter;
  int boco_i;
  cgsize_t* ptrToCgsize;
  cgsize_t cgsize[3][3];
  int nNodesFace;
  int nElemInlet;
  int nElemOutlet;
  int nElemWall;
  int nElemFrictless;
  int nElemShroud;
  int nElemShroudExt;
  int nElemPsblade;
  int nElemSsblade;
  int nElemSsleperiodic;
  int nElemPsleperiodic;
  int nElemSsteperiodic;
  int nElemPsteperiodic;
  int nElemRrinlet;
  int nElemRroutlet;
  int nElemHubAll;
  int nElemShroudAll;
  int nNodesInlet;
  int nNodesOutlet;
  int nNodesWall;
  int nNodesFrictless;
  int nNodesShroud;
  int nNodesShroudExt;
  int nNodesPsblade;
  int nNodesSsblade;
  int nNodesSsleperiodic;
  int nNodesPsleperiodic;
  int nNodesSsteperiodic;
  int nNodesPsteperiodic;
  int nNodesRrinlet;
  int nNodesRroutlet;
  int nNodesHubAll;
  int nNodesShroudAll;
  int nNodesFaceInlet;
  int nNodesFaceOutlet;
  int nNodesFaceWall;
  int nNodesFaceFrictless;
  int nNodesFaceShroud;
  int nNodesFaceShroudExt;
  int nNodesFacePsblade;
  int nNodesFaceSsblade;
  int nNodesFaceSsleperiodic;
  int nNodesFacePsleperiodic;
  int nNodesFaceSsteperiodic;
  int nNodesFacePsteperiodic;
  int nNodesFaceRrinlet;
  int nNodesFaceRroutlet;
  int nNodesFaceHubAll;
  int nNodesFaceShroudAll;
  int nObjects;
  const coDistributedObject *const *objSet;
  //----------------------------------------------------------------------------
  
  // initialize
  //----------------------------------------------------------------------------
  gElemCounter = 0;
  //----------------------------------------------------------------------------
  
  //get input port and do some type checks
  //----------------------------------------------------------------------------
  //check ports and cast to expected types
  //grid
  obj = p_inputPort_grid->getCurrentObject();
  if (!obj) {
    sendError("did not receive object at port %s", p_inputPort_grid->getName());
    return FAILURE;
  }
  else {
    inGrid = dynamic_cast<const coDoUnstructuredGrid *>(obj);
    //check that cast was successful
    if (!inGrid){
      sendError("received wrong object type at port %s", p_inputPort_grid->getName());
      return FAILURE;
    }
  }
  
  //boundary element faces
  obj = p_inputPort_boundaryElementFaces->getCurrentObject();
  if (!obj) {
    sendInfo("did not receive object at port %s", p_inputPort_boundaryElementFaces->getName());
  }
  else {
    inBoundaryElementFaces = dynamic_cast<const coDoSet *>(obj);
    objSet = inBoundaryElementFaces->getAllElements(&nObjects);
    
    inInletElementNodes = dynamic_cast<const coDoIntArr *>(objSet[0]);
    inOutletElementNodes = dynamic_cast<const coDoIntArr *>(objSet[1]);
    inShroudElementNodes = dynamic_cast<const coDoIntArr *>(objSet[2]);
    inShroudExtElementNodes = dynamic_cast<const coDoIntArr *>(objSet[3]);
    inFrictlessElementNodes = dynamic_cast<const coDoIntArr *>(objSet[4]);
    inPsbladeElementNodes = dynamic_cast<const coDoIntArr *>(objSet[5]);
    inSsbladeElementNodes = dynamic_cast<const coDoIntArr *>(objSet[6]);
    inWallElementNodes = dynamic_cast<const coDoIntArr *>(objSet[7]);
    inSsleperiodicElementNodes = dynamic_cast<const coDoIntArr *>(objSet[8]);
    inPsleperiodicElementNodes = dynamic_cast<const coDoIntArr *>(objSet[9]);
    inSsteperiodicElementNodes = dynamic_cast<const coDoIntArr *>(objSet[10]);
    inPsteperiodicElementNodes = dynamic_cast<const coDoIntArr *>(objSet[11]);
    inRrinletElementNodes = dynamic_cast<const coDoIntArr *>(objSet[12]);
    inRroutletElementNodes = dynamic_cast<const coDoIntArr *>(objSet[13]);
    inHubAllElementNodes = dynamic_cast<const coDoIntArr *>(objSet[14]);
    inShroudAllElementNodes = dynamic_cast<const coDoIntArr *>(objSet[15]);
    
    nNodesInlet = inInletElementNodes->getDimension(1);
    nNodesOutlet = inOutletElementNodes->getDimension(1);
    nNodesWall = inWallElementNodes->getDimension(1);
    nNodesFrictless = inFrictlessElementNodes->getDimension(1);
    nNodesShroud = inShroudElementNodes->getDimension(1);
    nNodesShroudExt = inShroudExtElementNodes->getDimension(1);
    nNodesPsblade = inPsbladeElementNodes->getDimension(1);
    nNodesSsblade = inSsbladeElementNodes->getDimension(1);
    nNodesSsleperiodic = inSsleperiodicElementNodes->getDimension(1);
    nNodesPsleperiodic = inPsleperiodicElementNodes->getDimension(1);
    nNodesSsteperiodic = inSsteperiodicElementNodes->getDimension(1);
    nNodesPsteperiodic = inPsteperiodicElementNodes->getDimension(1);
    nNodesRrinlet = inRrinletElementNodes->getDimension(1);
    nNodesRroutlet = inRroutletElementNodes->getDimension(1);
    nNodesHubAll = inHubAllElementNodes->getDimension(1);
    nNodesShroudAll = inShroudAllElementNodes->getDimension(1);
    
    elem = inInletElementNodes->getAddress();
    nNodesFaceInlet = *elem;
    elem = inOutletElementNodes->getAddress();
    nNodesFaceOutlet = *elem;
    elem = inWallElementNodes->getAddress();
    nNodesFaceWall = *elem;
    elem = inFrictlessElementNodes->getAddress();
    nNodesFaceFrictless = *elem;
    elem = inShroudElementNodes->getAddress();
    nNodesFaceShroud = *elem;
    elem = inShroudExtElementNodes->getAddress();
    nNodesFaceShroudExt = *elem;
    elem = inPsbladeElementNodes->getAddress();
    nNodesFacePsblade = *elem;
    elem = inSsbladeElementNodes->getAddress();
    nNodesFaceSsblade = *elem;
    elem = inSsleperiodicElementNodes->getAddress();
    nNodesFaceSsleperiodic = *elem;
    elem = inPsleperiodicElementNodes->getAddress();
    nNodesFacePsleperiodic = *elem;
    elem = inSsteperiodicElementNodes->getAddress();
    nNodesFaceSsteperiodic = *elem;    
    elem = inPsteperiodicElementNodes->getAddress();
    nNodesFacePsteperiodic = *elem;    
    elem = inRrinletElementNodes->getAddress();
    nNodesFaceRrinlet = *elem;    
    elem = inRroutletElementNodes->getAddress();
    nNodesFaceRroutlet = *elem;    
    elem = inHubAllElementNodes->getAddress();
    nNodesFaceHubAll = *elem;  
    elem = inShroudAllElementNodes->getAddress();
    nNodesFaceShroudAll = *elem;  
    
    nElemInlet = nNodesInlet/nNodesFaceInlet;
    nElemOutlet = nNodesOutlet/nNodesFaceOutlet;
    nElemShroud = nNodesShroud/nNodesFaceShroud;
    nElemShroudExt = nNodesShroudExt/nNodesFaceShroudExt;
    nElemFrictless = nNodesFrictless/nNodesFaceFrictless;
    nElemPsblade = nNodesPsblade/nNodesFacePsblade;
    nElemSsblade = nNodesSsblade/nNodesFaceSsblade;
    nElemWall = nNodesWall/nNodesFaceWall;
    nElemSsleperiodic = nNodesSsleperiodic/nNodesFaceSsleperiodic;
    nElemPsleperiodic = nNodesPsleperiodic/nNodesFacePsleperiodic;
    nElemSsteperiodic = nNodesSsteperiodic/nNodesFaceSsteperiodic;
    nElemPsteperiodic = nNodesPsteperiodic/nNodesFacePsteperiodic;
    nElemRrinlet = nNodesRrinlet/nNodesFaceRrinlet;
    nElemRroutlet = nNodesRroutlet/nNodesFaceRroutlet;
    nElemHubAll = nNodesHubAll/nNodesFaceHubAll;
    nElemShroudAll = nNodesShroudAll/nNodesFaceShroudAll;

    //check that cast was successful
    if (!inBoundaryElementFaces){
      sendError("received wrong object type at port %s", p_inputPort_boundaryElementFaces->getName());
      return FAILURE;
    }
  }
  
  /* -------------------------------------------------------------------------*/
  /* inGrid --> unstructured grid                                             */
  /*                                                                          */
  /* start to write inGrid into cgns file                                     */ 
  /* -------------------------------------------------------------------------*/
  
  
  //write out short info about mesh
  inGrid->getGridSize(&nElem, &nConn, &nCoords);
  sendInfo("nElements = %i", nElem);
  sendInfo("nConnectivities = %i", nConn);
  sendInfo("nCoordinates = %i", nCoords);
  
  //get addresses and type list
  inGrid->getAddresses(&elem, &conn, &xCoord, &yCoord, &zCoord);
  if (inGrid->hasTypeList()) {
    inGrid->getTypeList(&tList);
  }

/* debug output ///////////////////////////////////////////////////////////// */  
//  //write out type list
//  if (inGrid->hasTypeList()){
//    inGrid->getTypeList(&tList);
//    for (ii=0;ii<nElem;ii++){
//      cout << "tList[" << ii << "]" << "=(" << *(tList+ii) << endl;
//    }
//  }
//  //write out elements
//  for (ii=0;ii<nElem;ii++){
//    cout << "elem[" << ii << "]" << "=(" << *(elem+ii) << endl;
//  }
  //write out coordinates
//  for (ii=0;ii<nCoords;ii++){
//    cout << "coord[" << ii << "]" << "=(" << *(xCoord+ii) << "," << *(yCoord+ii) << "," << *(zCoord+ii) << ")" << endl;
//  }
//  //write out conectivities
//  for (ii=0;ii<nConn;ii++){
//    cout << "conn[" << ii << "]" << "=(" << *(conn+ii) << endl;
//  }
/* ////////////////////////////////////////////////////////////////////////// */  
  
  //write cgns file
  //----------------------------------------------------------------------------
  //open cgns file**************************************************************
  ier = cg_open(cgns_filebrowser->getValue(), CG_MODE_WRITE, &index_file);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "cgns file opened ..." << endl;
  //
  // base
  //
  //write base******************************************************************
  ier = cg_base_write(index_file, "base", 3, 3, &base_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "base written ..." << endl;
  // write zones****************************************************************
  cgsize[0][0] = nCoords;
  cgsize[0][1] = nElem;
  cgsize[0][2] = 0;
  ptrToCgsize = &cgsize[0][0];        
  ier = cg_zone_write(index_file, base_i, "runner", ptrToCgsize, CGNS_ENUMV(Unstructured), &zone_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "runner written ..." << endl;
  
  // write grid coordinates*****************************************************
  ier = cg_coord_write(index_file, base_i, zone_i, CGNS_ENUMV(RealSingle), "CoordinateX", xCoord, &coord_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "coordinate x written ..." << endl;
  ier = cg_coord_write(index_file, base_i, zone_i, CGNS_ENUMV(RealSingle), "CoordinateY", yCoord, &coord_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "coordinate y written ..." << endl;
  ier = cg_coord_write(index_file, base_i, zone_i, CGNS_ENUMV(RealSingle), "CoordinateZ", zCoord, &coord_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "coordinate z written ..." << endl;
  
  //write connectivities for 8-node-hexa grid***********************************
  intHelper = new cgsize_t[nConn];
  for (ii=0;ii<nConn;ii++){
    *(intHelper+ii) = *(conn+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }
  ier = cg_section_write(index_file, base_i, zone_i, "Elem", CGNS_ENUMV(HEXA_8), \
                         gElemCounter+1, nElem, 0, intHelper, &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  gElemCounter = gElemCounter + nElem;
  delete intHelper;
  cout << "Elem written ..." << endl;   
  
  //write inlet*****************************************************************
  elem = inInletElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesInlet];
  for (ii=0;ii<nNodesInlet;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }
  ier = cg_section_write(index_file, base_i, zone_i, "inlet", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemInlet, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemInlet;
  cout << "inlet written ... " << endl;
  
  //write outlet****************************************************************
  elem = inOutletElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesOutlet];
  for (ii=0;ii<nNodesOutlet;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "outlet", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemOutlet, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemOutlet;
  cout << "outlet written ... " << endl;   
  
  //write wall******************************************************************
  elem = inWallElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesWall];
  for (ii=0;ii<nNodesWall;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "wall", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemWall, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemWall;
  cout << "wall written ... " << endl;
  
  //write frictless*************************************************************
  elem = inFrictlessElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesFrictless];
  for (ii=0;ii<nNodesFrictless;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "frictless", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemFrictless, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemFrictless;
  cout << "frictless written ... " << endl;
  
  //write shroud****************************************************************
  elem = inShroudElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesShroud];
  for (ii=0;ii<nNodesShroud;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "shroud", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemShroud, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemShroud;
  cout << "shroud written ... " << endl;
  
  //write shroudExt*************************************************************
  elem = inShroudExtElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesShroudExt];
  for (ii=0;ii<nNodesShroudExt;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "shroudExt", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemShroudExt, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemShroudExt;
  cout << "shroudExt written ... " << endl; 
  
  //write psblade***************************************************************
  elem = inPsbladeElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesPsblade];
  for (ii=0;ii<nNodesPsblade;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "psblade", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemPsblade, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemPsblade;
  cout << "psblade written ... " << endl;   
  
  //write ssblade***************************************************************
  elem = inSsbladeElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesSsblade];
  for (ii=0;ii<nNodesSsblade;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "ssblade", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemSsblade, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemSsblade;
  cout << "ssblade written ... " << endl;  
  
  //write ssleperiodic**********************************************************
  elem = inSsleperiodicElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesSsleperiodic];
  for (ii=0;ii<nNodesSsleperiodic;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "ssleperiodic", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemSsleperiodic, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemSsleperiodic;
  cout << "ssleperiodic written ... " << endl; 
  
  //write psleperiodic**********************************************************
  elem = inPsleperiodicElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesPsleperiodic];
  for (ii=0;ii<nNodesPsleperiodic;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "psleperiodic", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemPsleperiodic, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemPsleperiodic;
  cout << "psleperiodic written ... " << endl;
  
  //write ssteperiodic**********************************************************
  elem = inSsteperiodicElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesSsteperiodic];
  for (ii=0;ii<nNodesSsteperiodic;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "ssteperiodic", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemSsteperiodic, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemSsteperiodic;
  cout << "ssteperiodic written ... " << endl;
  
  //write psteperiodic**********************************************************
  elem = inPsteperiodicElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesPsteperiodic];
  for (ii=0;ii<nNodesPsteperiodic;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "psteperiodic", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemPsteperiodic, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemPsteperiodic;
  cout << "psteperiodic written ... " << endl;
  
  //write rrinlet***************************************************************
  elem = inRrinletElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesRrinlet];
  for (ii=0;ii<nNodesRrinlet;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "rrinlet", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemRrinlet, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemRrinlet;
  cout << "rrinlet written ... " << endl;
  
  //write rroutlet**************************************************************
  elem = inRroutletElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesRroutlet];
  for (ii=0;ii<nNodesRroutlet;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "rroutlet", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemRroutlet, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemRroutlet;
  cout << "rroutlet written ... " << endl; 
  
  //write hubAll****************************************************************
  elem = inHubAllElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesHubAll];
  for (ii=0;ii<nNodesHubAll;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "hubAll", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemHubAll, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemHubAll;
  cout << "hubAll written ... " << endl; 
  
  //write shroudAll*************************************************************
  elem = inShroudAllElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesShroudAll];
  for (ii=0;ii<nNodesShroudAll;ii++){
    *(intHelper+ii) = *(elem+ii);
    *(intHelper+ii) = *(intHelper+ii)+1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "shroudAll", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemShroudAll, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemShroudAll;
  cout << "shroudAll written ... " << endl; 
  
  //write periodic suction******************************************************
  elem = inSsleperiodicElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesSsleperiodic+nNodesSsteperiodic];
  tmpCounter = 0;
  for (ii=0;ii<nNodesSsleperiodic;ii++){
    *(intHelper+tmpCounter) = *(elem+ii);
    *(intHelper+tmpCounter) = *(intHelper+tmpCounter)+1;
    tmpCounter = tmpCounter + 1;
  }    
  elem = inSsteperiodicElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  for (ii=0;ii<nNodesSsteperiodic;ii++){
    *(intHelper+tmpCounter) = *(elem+ii);
    *(intHelper+tmpCounter) = *(intHelper+tmpCounter)+1;
    tmpCounter = tmpCounter + 1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "periodic_suction", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemSsleperiodic+nElemSsteperiodic, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemSsleperiodic + nElemSsteperiodic;
  cout << "peridic_suction written ... " << endl;
  
  //write periodic pressure*****************************************************
  elem = inPsleperiodicElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  intHelper = new cgsize_t[nNodesPsleperiodic+nNodesPsteperiodic];
  tmpCounter = 0;
  for (ii=0;ii<nNodesPsleperiodic;ii++){
    *(intHelper+tmpCounter) = *(elem+ii);
    *(intHelper+tmpCounter) = *(intHelper+tmpCounter)+1;
    tmpCounter = tmpCounter + 1;
  }    
  elem = inPsteperiodicElementNodes->getAddress();
  //increment nodes
  elem = elem+1;
  for (ii=0;ii<nNodesPsteperiodic;ii++){
    *(intHelper+tmpCounter) = *(elem+ii);
    *(intHelper+tmpCounter) = *(intHelper+tmpCounter)+1;
    tmpCounter = tmpCounter + 1;
  }    
  ier = cg_section_write(index_file, base_i, zone_i, "periodic_pressure", CGNS_ENUMV(QUAD_4), \
                         gElemCounter+1, gElemCounter+nElemPsleperiodic+nElemPsteperiodic, 0, intHelper, \
                         &section_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  delete intHelper;
  gElemCounter = gElemCounter + nElemPsleperiodic + nElemPsteperiodic;
  cout << "peridic pressure written ... " << endl;
  
  //
  // baseCFX
  //
  //write base******************************************************************
  ier = cg_base_write(index_file, "baseCFX", 3, 3, &baseCFX_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "baseCFX written ..." << endl;
  // write zones****************************************************************
  cgsize[0][0] = nCoords;
  cgsize[0][1] = nElem;
  cgsize[0][2] = 0;
  ptrToCgsize = &cgsize[0][0];        
  ier = cg_zone_write(index_file, baseCFX_i, "runner", ptrToCgsize, CGNS_ENUMV(Unstructured), &zone_i);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "runner written ..." << endl;  
  
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("GridCoordinates", "", "/base/runner/GridCoordinates");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "coordinates linked ..." << endl;
  
  //connectivities
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("Elem", "", "/base/runner/Elem");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "Elem linked ..." << endl;   
  
  //write inlet*****************************************************************
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("inlet", "", "/base/runner/inlet");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "inlet linked ..." << endl;   
  
  //write outlet****************************************************************
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("outlet", "", "/base/runner/outlet");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "outlet linked ..." << endl;    
  
  //write psblade***************************************************************
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("blade_pressure", "", "/base/runner/psblade");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "psblade linked ..." << endl;    
  
  //write ssblade***************************************************************
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("blade_suction", "", "/base/runner/ssblade");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "ssblade linked ..." << endl;    
  
  //write hubAll****************************************************************
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("hub", "", "/base/runner/hubAll");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "hubAll linked ..." << endl;  
  
  //write shroudAll*************************************************************
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("shroud", "", "/base/runner/shroudAll");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "shroudAll linked ..." << endl;  
  
  //write periodic suction******************************************************
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("periodic_suction", "", "/base/runner/periodic_suction");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "periodic_suction linked ..." << endl;  
  
  //write periodic pressure*****************************************************
  ier = cg_goto(index_file, baseCFX_i, "runner", 0, "end"); 
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  ier = cg_link_write("periodic_pressure", "", "/base/runner/periodic_pressure");
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  cout << "periodic_pressure linked ..." << endl;
  
  //close cgns file*************************************************************
  ier = cg_close(index_file);
  if(ier) {
    sendError("%s", cg_get_error());
    return FAILURE;
  }
  //----------------------------------------------------------------------------
  
  sendInfo("grid successfully converted to cgns");
  return SUCCESS;
}

MODULE_MAIN(IO_MODULE, writeToCgns)
