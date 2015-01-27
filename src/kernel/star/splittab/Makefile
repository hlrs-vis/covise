SHAPE_OBJ = Hexa.o Prism.o Pyra.o Tetra.o Shape.o

.SUFFIXES: .cpp

CXX = CC
PROGS = illSplit samm8Split

all:  $(PROGS) 

illSplit : illSplit.o  $(SHAPE_OBJ)
	${CXX} -o illSplit illSplit.o $(SHAPE_OBJ)

samm8Split : samm8Split.o
	${CXX} -o samm8Split samm8Split.o

Hexa.o   : Hexa.cpp
Prism.o  : Prism.cpp
Pyra.o   : Pyra.cpp
Tetra.o  : Tetra.cpp
Shape.o  : Shape.cpp

.cpp.o :
	${CXX} -c -g $<

clean:
	/bin/rm -f *.o $(PROGS)

depend:
	csh -f -c 'makedepend -Y *.cpp |& grep -v "cannot find include file" || /bin/true'

# DO NOT DELETE

Hexa.o: Hexa.h Shape.h Prism.h Pyra.h Tetra.h
Prism.o: Pyra.h Shape.h Prism.h Tetra.h
Pyra.o: Pyra.h Shape.h Tetra.h
Shape.o: Shape.h
Tetra.o: Tetra.h Shape.h
ill-triv.o: Hexa.h Shape.h Prism.h Pyra.h Tetra.h
illSplit.o: Hexa.h Shape.h Prism.h Pyra.h Tetra.h
