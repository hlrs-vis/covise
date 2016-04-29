#include <stdio.h>
#include <string.h>


int Check_Little_Endian(void)
{
	int a = 1;
	char *p;

	p = (char *)&a;

	return (*p ? 1 : 0);
}

void SwapBytes(void *p, int num)
{
	int i;

	for (i = 0; i < num; i++, p+=4)
		SwapByte((int *)p);
}

void SwapByte(int *p)
{
	register char h;
	register char *x = (char *)p;

	h = *x;
	*x = *(x+3);
	*(x+3) = h;

	h = *(x+1);
	*(x+1) = *(x+2);
	*(x+2) = h;
}


#ifdef	MAIN

int main(int argc, char **argv)
{
	fprintf(stderr, "Little Endian = %d\n", Little_Endian());
	return 0;
}
#endif
