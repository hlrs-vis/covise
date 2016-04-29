#ifndef BCNODES_INCLUDE
#define BCNODES_INCLUDE

int *GetBCNodes(struct Ilist *nodes, int nnum);
int AllocBCNodesMemory(struct rr_grid *grid);
int FreeBCNodesMemory(struct rr_grid *grid);
#endif
