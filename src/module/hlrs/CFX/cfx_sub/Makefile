#architecture
ARCH   = linux-amd64

#object directory
OBJDIR = ./$(ARCH)

#C compiler
CC     = /usr/bin/gcc
CFLAGS = -fPIC -g -I$(CFX_HOME)/CFX/include

#C++ compiler
CPP = /usr/bin/g++

#Fortran compiler
#FC    = /mnt/raid/soft/pgi/linux86-64/7.1-6/bin/pgf90
FC    = gfortran

#Fortran compiler flags
#F77FLAGS = -fpic -Msecond_underscore -g77libs -I$(CFX_HOME)/ansys_inc/v110/CFX/include
F77FLAGS = -g -fpic -fsecond-underscore -I$(CFX_HOME)/CFX/include

#linker flags
LDFLAGS = -g -shared -fpic -g77libs #-lg2c

#source code
OBJS    = $(OBJDIR)/user_export.o $(OBJDIR)/user_covise.o $(OBJDIR)/coSimClient.o $(OBJDIR)/cfxMalloc.o 
IMPORTOBJS = $(OBJDIR)/user_import.o $(OBJDIR)/coSimClient.o $(OBJDIR)/cfxMalloc.o $(CFX_HOME)/CFX/lib/linux-amd64/libmeshimport.a $(CFX_HOME)/CFX/lib/linux-amd64/libratlas_api.a $(CFX_HOME)/CFX/lib/linux-amd64/libratlas.a $(CFX_HOME)/CFX/lib/linux-amd64/libio.a $(CFX_HOME)/CFX/lib/linux-amd64/libcclapilt.a
all: user_import user_export factor sinangular cosangular

user_import: $(IMPORTOBJS)
	$(CC) -lm -lc -o $@ $^

user_export: $(OBJS)
	$(FC) $(LDFLAGS) -shared -o ./$(OBJDIR)/libuser_export.so $(OBJS)

factor: $(OBJDIR)/factor.o
	$(FC) $(LDFLAGS) -shared -o ./$(OBJDIR)/libuser_factor.so $^
	
sinangular: $(OBJDIR)/sinangular.o
	$(FC) $(LDFLAGS) -shared -o ./$(OBJDIR)/libuser_sinangular.so $^	
	
cosangular: $(OBJDIR)/cosangular.o
	$(FC) $(LDFLAGS) -shared -o ./$(OBJDIR)/libuser_cosangular.so $^	
	

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o ./$(OBJDIR)/$@

$(OBJDIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@


$(OBJDIR)/user_export.o:	user_export.F
	$(FC) $(F77FLAGS) -c user_export.F -o $(OBJDIR)/user_export.o

$(OBJDIR)/user_covise.o:	user_covise.F
	$(FC) $(F77FLAGS) -c user_covise.F -o $(OBJDIR)/user_covise.o

$(OBJDIR)/coSimClient.o:	coSimClient.c
	$(CC) $(CFLAGS) -c coSimClient.c -o $(OBJDIR)/coSimClient.o

$(OBJDIR)/cfxMalloc.o:	cfxMalloc.c
	$(CC) $(CFLAGS) -c cfxMalloc.c -o $(OBJDIR)/cfxMalloc.o

$(OBJDIR)/factor.o:	factor.F
	$(FC) $(F77FLAGS) -c factor.F -o $(OBJDIR)/factor.o
	
$(OBJDIR)/sinangular.o:	sinangular.F
	$(FC) $(F77FLAGS) -c sinangular.F -o $(OBJDIR)/sinangular.o
	
$(OBJDIR)/cosangular.o:	cosangular.F
	$(FC) $(F77FLAGS) -c cosangular.F -o $(OBJDIR)/cosangular.o	
	

clean:
	-rm $(OBJDIR)/*.o $(OBJDIR)/*.so 
