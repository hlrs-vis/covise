#define  BSWP_MUSTSWAP  1
#define  BSWP_DONTSWAP  0

extern  int Check_Little_Endian(void);
extern  void SwapBytes(void *p, int num);
extern  void SwapByte(int *p);
