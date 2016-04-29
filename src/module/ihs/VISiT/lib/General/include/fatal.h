#include <stdio.h>

void my_fatal(const char *src, int line, const char *text);

#define  fatal(x) my_fatal(__FILE__, __LINE__, (x))
