##########################################################
## Description: Makefile Template for COVISE modules    ##
## (C) 1995 Computer Center University of Stuttgart     ##
##            Allmandring 30  70550 Stuttgart           ##  
## Author: Uwe Woessner, Andreas Wierse  Date: 29.08.97 ##
##########################################################

                          # name of the module executable
include $(COVISEDIR)/src/make.env.$(ARCHSUFFIX)
include $(COVISEDIR)/src/make.rules

LIB_FILE         = libMasterAndWorkers$(LIBSUFFIX)

APPLICATION     = $(LIB_FILE)

CATEGORY              = .

PROJECT            = General

OBJECTS         = $(OBJDIR)/MasterAndWorkers.o \
                  $(OBJDIR)/Thread.o \
                  $(OBJDIR)/ThreadInitialise.o \
                  $(OBJDIR)/ThreadTask.o \
                  $(OBJDIR)/PosixMutexAndCondition.o \
                  $(OBJDIR)/PosixThread.o \
                  $(OBJDIR)/TaskGroup.o

EXAMPLE_OBJECTS                =  $(OBJDIR)/MyThreadTask.o \
                                  $(OBJDIR)/MyTaskGroup.o \
                                  $(OBJDIR)/MyMasterAndWorkers.o \
                                  $(OBJDIR)/MyMain.o

EXAMPLE         = MyExample

MYFLAGS         = -D_REENTRANT 

ADD_CLEAN       = MyExample

#SRCFILES = $${THISDIR}/*.cpp $${THISDIR}/*.h $${THISDIR}/Makefile

all: $(PLATFORM)/bin/$(LIB_FILE) $(EXAMPLE)

$(PLATFORM)/bin/$(LIB_FILE): $(OBJECTS)
	$(MKSHARED)$(LIB_FILE) $(MYFLAGS) $(OBJDIR)/*.o $(ADDOBJECTS)
	chmod 755 $(LIB_FILE)
	cp $(LIB_FILE) $(PLATFORM)/bin
	echo $(LIB_FILE) created

$(EXAMPLE): $(EXAMPLE_OBJECTS) $(PLATFORM)/bin/$(LIB_FILE)
	$(CC) $(CPPFLAGS) $(MYFLAGS) -o $@ $(EXAMPLE_OBJECTS) \
		$(COVISE_LIBDIR) $(POSIX_THREADS_LIB) -lMasterAndWorkers

$(OBJDIR)/$(APPLICATION).o: $(APPLICATION).cpp 
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c $(APPLICATION).cpp -o $(OBJDIR)/$(APPLICATION).o
	 $(POSTCC)

$(OBJDIR)/MasterAndWorkers.o: MasterAndWorkers.cpp MasterAndWorkers.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c MasterAndWorkers.cpp -o $(OBJDIR)/MasterAndWorkers.o
	$(POSTCC)

$(OBJDIR)/Thread.o: Thread.cpp Thread.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c Thread.cpp -o $(OBJDIR)/Thread.o
	$(POSTCC)

$(OBJDIR)/ThreadInitialise.o: ThreadInitialise.cpp ThreadInitialise.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c ThreadInitialise.cpp -o $(OBJDIR)/ThreadInitialise.o
	$(POSTCC)

$(OBJDIR)/ThreadTask.o: ThreadTask.cpp ThreadTask.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c ThreadTask.cpp -o $(OBJDIR)/ThreadTask.o
	$(POSTCC)

$(OBJDIR)/PosixMutexAndCondition.o: PosixMutexAndCondition.cpp PosixMutexAndCondition.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c PosixMutexAndCondition.cpp -o $(OBJDIR)/PosixMutexAndCondition.o
	$(POSTCC)

$(OBJDIR)/PosixThread.o: PosixThread.cpp PosixThread.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c PosixThread.cpp -o $(OBJDIR)/PosixThread.o
	$(POSTCC)

$(OBJDIR)/TaskGroup.o: TaskGroup.cpp TaskGroup.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c TaskGroup.cpp -o $(OBJDIR)/TaskGroup.o
	$(POSTCC)

$(OBJDIR)/MyThreadTask.o: MyThreadTask.cpp MyThreadTask.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c MyThreadTask.cpp -o $(OBJDIR)/MyThreadTask.o
	$(POSTCC)

$(OBJDIR)/MyTaskGroup.o: MyTaskGroup.cpp MyTaskGroup.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c MyTaskGroup.cpp -o $(OBJDIR)/MyTaskGroup.o
	$(POSTCC)

$(OBJDIR)/MyMasterAndWorkers.o: MyMasterAndWorkers.cpp MyMasterAndWorkers.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c MyMasterAndWorkers.cpp -o $(OBJDIR)/MyMasterAndWorkers.o
	$(POSTCC)

$(OBJDIR)/MyMain.o: MyMain.cpp MyMasterAndWorkers.h
	$(CC) $(CPPFLAGS) $(MYFLAGS) -c MyMain.cpp -o $(OBJDIR)/MyMain.o
	$(POSTCC)

