#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../General/include/flist.h"
#include "../General/include/ilist.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/nodes.h"
#include "../General/include/elements.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#include "include/rr_grid.h"
#include "include/rr_meshall.h"
#include "include/rr_bcnodes.h"
#include "include/rr_elem.h"
#include "include/rr_meshnodmisc.h"
#include "include/rr_meshwrite.h"
#ifdef SMOOTH_MESH
#include "include/rr_meshsmooth.h"
#endif

#define NPOIN_EXT 10

#ifdef GAP
#include "include/rr_gapelem.h"
#endif

#ifdef DEBUG_NODES
#define VPRINT(x) fprintf(stderr,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#define VPRINTF(x,f) fprintf(f,"%10s = [%4f, %4f, %4f]\n",#x,x[0], x[1], x[2])
#endif

int MeshRR_GridRegions(struct rr_grid *grid)
{
	int i, elreg_num, ii1;

#ifdef DEBUG_NODES
	char fn[111];
	FILE *fp;
	int j, k, l;
#endif

#ifdef GAP
	int gpoffset, itip, ishroud, nnum_regular, elnum_regular, nnum_prev;
#endif

	int *psblade      = NULL;
	int *ssblade      = NULL;
	int *psleperiodic = NULL;
	int *ssleperiodic = NULL;
	int *psteperiodic = NULL;
	int *ssteperiodic = NULL;
	int *inlet        = NULL;
	int *outlet       = NULL;

	struct Ilist *dummy_inlet = NULL;

	grid->n = AllocNodelistStruct();
	grid->e = AllocElementStruct();
	grid->wall         = AllocElementStruct();
	grid->shroud       = AllocElementStruct();
	grid->shroudext    = AllocElementStruct();
	grid->psblade      = AllocElementStruct();
	grid->ssblade      = AllocElementStruct();
	grid->psleperiodic = AllocElementStruct();
	grid->ssleperiodic = AllocElementStruct();
	grid->psteperiodic = AllocElementStruct();
	grid->ssteperiodic = AllocElementStruct();
	grid->einlet       = AllocElementStruct();
	grid->eoutlet      = AllocElementStruct();
	grid->frictless    = AllocElementStruct();
#ifdef RR_IONODES
	grid->rrinlet      = AllocElementStruct();
	grid->rroutlet     = AllocElementStruct();
#endif
	FreeBCNodesMemory(grid);
	AllocBCNodesMemory(grid);
	if(dummy_inlet) FreeIlistStruct(dummy_inlet);
	dummy_inlet = AllocIlistStruct(grid->cdis+1);

	// create nodes
	fprintf(stdout,"\nCreating nodes for %d grid elements ... ",
		grid->ge_num);fflush(stdout);
	ii1 = grid->cge[0]->reg[3]->line[1]->nump - grid->cge[0]->reg[2]->line[2]->nump;
		for(i = 0; i < grid->ge_num; i++) {
#ifdef DEBUG_NODES
		sprintf(fn,"rr_debugnodes_%02d.txt", i);
		if( (fp = fopen(fn,"w+")) == NULL) {
		       fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
			exit(-1);
		}
		fprintf(stderr," MeshRR_XXXRegion %2d",i);fflush(stderr);
#endif

		if(grid->mesh_ext)
			MeshRR_InletRegion(grid->n,grid->ge[i]->ml,dummy_inlet,
					   grid->psle, grid->ssle, 
					   grid->cge[i]->reg[0]);
		else
			MeshRR_InletRegion(grid->n,grid->ge[i]->ml,grid->inlet,
					   grid->psle, grid->ssle, 
					   grid->cge[i]->reg[0]);
		/*MeshRR_SSRegion(grid->n, grid->ge[i]->ml, grid->ssnod, 
				grid->ssle, grid->cge[i]->reg[1], 
				grid->cge[i]->reg[0]);*/
		MeshMod_SSRegion(grid->n, grid->ge[i]->ml, grid->ssnod, 
				grid->ssle, grid->cge[i]->reg[1], 
				grid->cge[i]->reg[0], ii1);
		/*MeshRR_PSRegion(grid->n, grid->ge[i]->ml, grid->psnod, 
				grid->psle, grid->cge[i]->reg[3], 
				grid->cge[i]->reg[0]);*/
		MeshMod_PSRegion(grid->n, grid->ge[i]->ml, grid->psnod, 
				 grid->psle, grid->cge[i]->reg[3], 
				 grid->cge[i]->reg[0],ii1);
		MeshRR_CoreRegion(grid->n, grid->ge[i]->ml, 
				  grid->cge[i]->reg[2], grid->cge[i]->reg[0], 
				  grid->cge[i]->reg[1],
				  grid->cge[i]->reg[3], grid->v14_angle[0]);
		MeshRR_SSTERegion(grid->n, grid->ge[i]->ml, 
				  grid->cge[i]->reg[4], grid->cge[i]->reg[1],
				  grid->ssnod, grid->sste,
				  grid->outlet);
		MeshRR_OutletRegion(grid->n, grid->ge[i]->ml, 
				    grid->cge[i]->reg[5], 
				    grid->cge[i]->reg[1], grid->cge[i]->reg[4],
				    grid->cge[i]->reg[2], grid->outlet);
		MeshRR_PSTERegion(grid->n, grid->ge[i]->ml, 
				  grid->cge[i]->reg[6], grid->cge[i]->reg[3], 
				  grid->cge[i]->reg[5],
				  grid->psnod, grid->pste, grid->outlet);
		if(grid->mesh_ext)
		    MeshRR_ExtRegion(grid->n, grid->ge[i]->ml, grid->inlet,
				     grid->psle, grid->ssle,
				     grid->cge[i]->reg[7], 
				     grid->cge[i]->reg[0]);
		if(i == 0) {
			grid->n->offset = grid->n->num;
		}

#ifdef DEBUG_NODES
		DumpNodes(&grid->n->n[i*grid->n->offset], grid->n->offset, fp);
		fprintf(fp,"grid->n->offset = %d\n",grid->n->offset);
#endif
		CalcNodeCoords(&grid->n->n[i*grid->n->offset], grid->n->offset,
			       grid->rot_clock);

#ifdef BLNODES_OUT
		PutBladeNodes(grid->n, grid->ssnod, grid->psnod, i, 
			      grid->ge[i]->para);
#endif

#ifdef DEBUG_NODES
		fprintf(fp,"i = %3d, equiv = %3d\n",i,
				EquivCheck(&grid->n->n[i*grid->n->offset],grid->n->offset));
		fclose(fp);
		fprintf(stderr," -> number of nodes: %8d\n",grid->n->num);
#endif
	} // end i
	fprintf(stdout,"done!\n%d nodes created. %d per grid element.\n",
			grid->n->num,grid->n->offset);
	FreeIlistStruct(dummy_inlet);

#ifdef GAP
	// rm clearance nodes from blade surface node lists
	grid->ssnod->num -= (grid->ssnod->num/grid->ge_num) * (grid->gp_num-1);
	grid->psnod->num -= (grid->psnod->num/grid->ge_num) * (grid->gp_num-1);
	nnum_regular  = grid->n->num;
#ifdef DEBUG_NODES
	fprintf(stderr," nnum_regular = %d\n",nnum_regular);
#endif
	ishroud = grid->ge_num - 1;
	itip    = grid->ge_num - (grid->gp_num);
	fprintf(stdout,"Creating nodes in %d tip clearance regions ... ",grid->gp_num);fflush(stdout);
	for(i = itip; i <= ishroud; i++) {
		nnum_prev = grid->n->num;
		MeshRR_SSGapRegion(grid->n, grid->ge[i]->ml, 
						   grid->sste, grid->ssnod, 
						   grid->cge[i]->reg[grid->reg_num], 
						   grid->cge[i]->reg[1], grid->cge[i]->reg[4], i-itip);
		MeshRR_PSGapRegion(grid->n, grid->ge[i]->ml, 
						   grid->pste, grid->psnod, 
						   grid->cge[i]->reg[grid->reg_num+1], 
						   grid->cge[i]->reg[3], grid->cge[i]->reg[6], i-itip);
	if(i == itip) gpoffset = grid->n->num - nnum_regular;
	if( (grid->n->num - nnum_prev) != gpoffset) {
		fprintf(stderr,"offset between tip clearance node layers not constant!\n");
		fprintf(stderr,"  gpoffset = %d, grid->n->num - nnum_prev = %d\n\n",
				gpoffset, grid->n->num - nnum_prev);
		exit(-1);
	}
	CalcNodeCoords(&grid->n->n[nnum_regular+(i-itip)*gpoffset], gpoffset);
	
#ifdef DEBUG_NODES
	fprintf(stderr," GAP %2d added: total number of nodes: %8d\n",i, grid->n->num);
#endif
	}
	fprintf(stdout,"done! %d nodes added. %d per gap grid element.\n",
			grid->n->num-nnum_regular, gpoffset);
#endif // GAP

#ifdef READJUST_PERIODIC
#ifdef GAP
	fprintf(stderr,"\n WARNING: Readjustment of periodic nodes not implemented for\n");
	fprintf(stderr,"          grid with tip clearance, yet! Continuing anyway.\n");
	fprintf(stderr,"          src: %s, line: %d\n\n",__FILE__, __LINE__);
#else
	fprintf(stdout,"\n Readjusting periodic areas ..."); fflush(stdout);
	ReadjustPeriodic(grid->n, grid->psle, grid->ssle, grid->pste, grid->sste, grid->ge_num);
	fprintf(stdout," done!\n");
#endif // GAP
#endif // READJUST_PERIODIC

	// boundary node flags
#ifdef DEBUG_NODES
	fprintf(stderr,"\n ... MeshRR_GridRegions(): creating bc flags ...");fflush(stderr);
#endif
	psblade      = GetBCNodes(grid->psnod, grid->n->num);
#ifdef DEBUG_NODES
	fprintf(stderr," psblade,");fflush(stderr);
#endif
	ssblade      = GetBCNodes(grid->ssnod, grid->n->num);
#ifdef DEBUG_NODES
	fprintf(stderr," ssblade,");fflush(stderr);
#endif
	psleperiodic = GetBCNodes(grid->psle, grid->n->num);
	ssleperiodic = GetBCNodes(grid->ssle, grid->n->num);
	psteperiodic = GetBCNodes(grid->pste, grid->n->num);
	ssteperiodic = GetBCNodes(grid->sste, grid->n->num);
	inlet        = GetBCNodes(grid->inlet, grid->n->num);
	outlet       = GetBCNodes(grid->outlet, grid->n->num);
#ifdef DEBUG_NODES
	fprintf(stderr,"\n done!\n");
#endif

	// create Elements
	fprintf(stdout,"Creating elements ... ");fflush(stdout);
	elreg_num = grid->reg_num;
	for(i = 1; i < grid->ge_num; i++) {
		CreateRR_Elements(grid->cge[(i-1)]->reg, grid->e, psblade, 
				  ssblade, psleperiodic, ssleperiodic, 
				  psteperiodic, ssteperiodic, inlet, outlet, 
				  grid->wall, grid->psblade, grid->ssblade, 
				  grid->psleperiodic, grid->ssleperiodic, 
				  grid->psteperiodic, grid->ssteperiodic, 
				  grid->einlet, grid->eoutlet, elreg_num, 
				  grid->n->offset);
	}
	fprintf(stdout,"done! %d elements created.\n",grid->e->nume);

#ifdef GAP
	elnum_regular = grid->e->nume;
#endif

	// boundary elements at hub and shroud
	i = grid->iinlet;
	GetHubElements(grid->e, grid->wall, grid->frictless, grid->shroudext, 
		       grid->n, grid->e->nume/(grid->ge_num-1), 
		       grid->ge[0]->ml->len[i],
		 grid->ge[0]->ml->len[grid->ge[0]->ml->p->nump-grid->ioutlet]);
	GetShroudElements(grid->e, grid->shroud, grid->shroudext, grid->n, 
			  grid->e->nume/(grid->ge_num-1), grid->ge_num, 
			  grid->ge[grid->ge_num-1]->ml->len[i],
			  grid->ge[grid->ge_num-1]->ml->len[grid->ge[grid->ge_num-1]->ml->p->nump-grid->ioutlet]);
#ifdef RR_IONODES
	GetRRIONodes(grid->e, grid->n, grid->ge, grid->ge_num, grid->rrinlet, 
				 grid->rroutlet, grid->cge, grid->ioutlet);
#endif

#ifdef PATRAN_SES_OUT
	WritePATRAN_SESfile(grid->n->num, grid->e->nume, grid->ge_num, 0, 0, 
			    "MERIDIANelems", "MERIDIANnodes", "meridian", 
			    "meridian_node");
#endif

	// Elements in tip clearance
#ifdef GAP
	for(i = itip; i < ishroud; i++) {
		CreateRR_GapElements(grid->cge[i]->reg, grid->cge[i+1]->reg, 
				     grid->e, psteperiodic, ssteperiodic, 
				     grid->psblade,grid->ssblade, 
				     grid->psteperiodic, grid->ssteperiodic, 
				     grid->wall, grid->shroud, grid->reg_num, 
				     grid->gpreg_num, gpoffset, i-itip, 
				     i-ishroud+1
#ifdef DEBUG_GAPELEM
							 , grid->n->n
#endif
			);
	}
#ifdef PATRAN_SES_OUT
	WritePATRAN_SESfile(grid->n->num-nnum_regular, 
			    grid->e->nume-elnum_regular, grid->gp_num, 
			    nnum_regular, elnum_regular, "MERIDIANTIPelems", 
			    "MERIDIANTIPnodes", "meridiantip",
			    "meridiantip_node");
#endif
#endif // GAP

#ifdef SMOOTH_MESH
	SmoothRR_Mesh(grid->n,grid->e,grid->ge_num,grid->psnod,grid->ssnod,
		      grid->psle,
		      grid->ssle, grid->pste, grid->sste, grid->inlet,
		      grid->outlet);
#endif

#ifdef FENFLOSS_OUT
#ifdef ONLY_GGEN
    grid->write_grid = 1;
#endif
	if(grid->write_grid) {
		if(grid->create_inbc) CreateInletBCs(grid->inlet,grid->n,
						     grid->inbc,&grid->bcval,
						     grid->alpha_const,
						     grid->turb_prof);
		WriteFENFLOSS_Geofile(grid->n, grid->e);
		WriteFENFLOSS_BCfile(grid->n, grid->e, grid->wall, 
				     grid->frictless, grid->shroud, 
				     grid->shroudext, grid->psblade, 
				     grid->ssblade, grid->psleperiodic, 
				     grid->ssleperiodic, grid->psteperiodic,
				     grid->ssteperiodic, grid->einlet, 
				     grid->eoutlet, grid->rrinlet, 
				     grid->rroutlet, grid->inlet, 
				     grid->bcval,grid->rot_ext);
		WriteFENFLOSS62x_BCfile(grid->n, grid->e, grid->wall, 
					grid->frictless,grid->shroud, 
					grid->shroudext, grid->psblade, 
					grid->ssblade, grid->psleperiodic, 
					grid->ssleperiodic, grid->psteperiodic,
					grid->ssteperiodic, grid->einlet, 
					grid->eoutlet, grid->rrinlet, 
					grid->rroutlet, grid->inlet,
					grid->bcval,grid->rot_ext);
	}
#endif

#ifdef DEBUG_NODES
	fprintf(stderr,"MeshRR_GridRegions: DumpNodes ... ");fflush(stderr);

	for(i = 0; i < grid->ge_num; i++) {
		sprintf(fn,"rr_nodes_%02d.txt", i);
		if( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
			exit(-1);
		}
		DumpNodes(&grid->n->n[i*grid->n->offset], grid->n->offset, fp);
		for(j = 0; j < grid->reg_num; j++) {
			for(k = 0; k < grid->cge[i]->reg[j]->numl; k++) {
				fprintf(fp,"\n\n");
				for(l = 0; l < grid->cge[i]->reg[j]->nodes[k]->num; l++) {
					fprintf(fp," %10d    %10.6f  %10.6f  %10.6f  %10.6f      %10.6f  %10.6f  %10.6f\n",
							grid->n->n[grid->cge[i]->reg[j]->nodes[k]->list[l]]->id,
							grid->n->n[grid->cge[i]->reg[j]->nodes[k]->list[l]]->phi,
							grid->n->n[grid->cge[i]->reg[j]->nodes[k]->list[l]]->l,
							grid->n->n[grid->cge[i]->reg[j]->nodes[k]->list[l]]->r,
							grid->n->n[grid->cge[i]->reg[j]->nodes[k]->list[l]]->arc,
							grid->n->n[grid->cge[i]->reg[j]->nodes[k]->list[l]]->x,
							grid->n->n[grid->cge[i]->reg[j]->nodes[k]->list[l]]->y,
							grid->n->n[grid->cge[i]->reg[j]->nodes[k]->list[l]]->z);
				}
			}
		}

		fclose(fp);
	}

	sprintf(fn,"rr_nodes_all.txt");
	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"Shit happened opening file '%s'!\n",fn);
		exit(-1);
	}
	DumpNodes(&grid->n->n[0], grid->n->num, fp);
	fclose(fp);
	fprintf(stderr,"done!\n");
#endif

#ifdef DEBUG_BC
	fprintf(stderr,"MeshRR_GridRegions: DumpBoundaries, DumpBCElements ... ");fflush(stderr);
	DumpBoundaries(grid->n, grid->inlet, grid->psle, grid->ssle, grid->psnod, grid->ssnod,
				   grid->pste, grid->sste, grid->outlet, grid->ge_num);
	DumpBCElements(grid->wall, grid->n, "wall");
	DumpBCElements(grid->frictless, grid->n, "frictionless");
	DumpBCElements(grid->shroud, grid->n, "shroud");
	DumpBCElements(grid->shroudext, grid->n, "shroudext");
	DumpBCElements(grid->einlet, grid->n, "inlet");
	DumpBCElements(grid->eoutlet, grid->n, "outlet");
	DumpBCElements(grid->psblade, grid->n, "psblade");
	DumpBCElements(grid->ssblade, grid->n, "ssblade");
	DumpBCElements(grid->psleperiodic, grid->n, "psleperiodic");
	DumpBCElements(grid->ssleperiodic, grid->n, "ssleperiodic");
	DumpBCElements(grid->psteperiodic, grid->n, "psteperiodic");
	DumpBCElements(grid->ssteperiodic, grid->n, "ssteperiodic");
#ifdef RR_IONODES
	DumpBCElements(grid->rrinlet, grid->n, "rrinlet");
	DumpBCElements(grid->rroutlet, grid->n, "rroutlet");
#endif
	fprintf(stderr,"done!\n");
#endif // DEBUG_BC

#ifdef DEBUG_ELEMENTS
	fprintf(stderr,"MeshRR_GridRegions: DumpElements ... ");fflush(stderr);
	DumpElements(grid->n, grid->e, grid->ge_num);
	fprintf(stderr,"done!\n");
#endif

#ifdef MESH_2DMERIDIAN_OUT
	fprintf(stdout,"MeshRR_GridRegions: Write_2DMeridianMesh ... ");
	fflush(stdout);
	Write_2DMeridianMesh(grid->n, grid->e, grid->ge_num);
	fprintf(stdout,"done!\n");
#endif
	free(psblade);
	free(ssblade);
	free(inlet);
	free(outlet);
	free(psleperiodic);
	free(ssleperiodic);
	free(psteperiodic);
	free(ssteperiodic);

	return 0;
}
