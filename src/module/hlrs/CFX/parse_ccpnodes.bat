@echo off

shift

myecho -n %1 > 00.txt
myecho -n *  >> 00.txt
myecho -n %2 >> 00.txt

shift
shift

:back
set arg=%1
if defined arg (
   rem myecho -n ;  >> 00.txt
   myecho -n " "  >> 00.txt
   myecho -n %1 >> 00.txt
   myecho -n *  >> 00.txt
   myecho -n %2 >> 00.txt

   shift
   shift
   goto back
)
