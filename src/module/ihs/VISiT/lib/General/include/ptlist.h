#ifndef PTLIST_INCLUDE
#define PTLIST_INCLUDE

struct ptlist {
	int num;
	int max;
	int portion;
	void **pt;
};

struct ptlist *AllocPtListStruct(void);
void FreePtListStruct(struct ptlist *ptlist);
int Add2PtList(struct ptlist *ptlist, void *pt);

#endif // PTLIST_INCLUDE
