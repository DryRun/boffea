#-----------------------------------------------------------------------------

PROGRAM = TripleGaussianPdf2

HDRS = TripleGaussianPdf2.h

OBJS = TripleGaussianPdf2.o\
 dict_$(PROGRAM).o

#-----------------------------------------------------------------------------
ObjSuf = o
SrcSuf = cc
ExeSuf = exe
DllSuf = so
OutPutOpt = -o

ROOTCFLAGS = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs)
ROOTGLIBS = $(shell root-config --glibs)
ROOTLDFLAGS = $(shell root-config --ldflags)

# Directories
HEADRDIR = .

# debug option: -g
# no exceptions: -fno-exceptions

CXX = g++
CXXFLAGS = -O3 -Wall -fPIC -g -std=c++11
LD = g++
LDFLAGS = -O3 -Wall -g 
SOFLAGS = -shared
ARCHFLAGS =

#
CXXFLAGS += $(ROOTCFLAGS) -I$(HEADRDIR)
LIBS = $(ROOTLIBS)
GLIBS = $(ROOTGLIBS)
LDFLAGS += $(ROOTLDFLAGS)


%.o: %.cc
	$(CXX) -c $(CXXFLAGS) $*.cc

$(PROGRAM).so: $(OBJS)
	@echo "Linking $(PROGRAM).so ..."
	$(LD) $(ARCHFLAGS) $(LDFLAGS) $(SOFLAGS) $(OBJS) $(ROOTLIBS) -o$(PROGRAM).so
	@echo "linking done"

dict_$(PROGRAM).cc: $(HDRS)
	@echo "Generating dictionary ..."
	rootcint -f dict_$(PROGRAM).cc -c -p -I$(HEADRDIR) $(HDRS) LinkDef.h 

clean:
	@rm -f $(OBJS) $(PROGRAM).so dict_*


#------------------------------------------------------------------------
