#include <stdio.h>
#include <Tacx.h>


int main(int argc, char **argv)
{
    Tacx *tacx = new Tacx();
    while(1)
    {
        tacx->update();
    }
    return 0;
}
