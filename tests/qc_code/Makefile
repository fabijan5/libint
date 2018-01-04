TOPDIR=../..
ifndef SRCDIR
  SRCDIR=$(shell pwd)
endif
-include $(TOPDIR)/src/bin/MakeVars
-include $(TOPDIR)/src/lib/libint/MakeVars.features

# include headers the object include directory
CPPFLAGS += -I$(TOPDIR)/include -I$(TOPDIR)/include/libint2 -I$(SRCDIR)/$(TOPDIR)/src/lib/libint -DSRCDATADIR=\"$(SRCDIR)/$(TOPDIR)/lib/basis\" -I/Users/fabijanpavosevic/research/calculations/test-orca/lda/xcfun/include

COMPILER_LIB = $(TOPDIR)/src/bin/libint/libINT.a
COMPUTE_LIB = -lint2 -lxcfun 
vpath %.a $(TOPDIR)/lib:$(TOPDIR)/lib/.libs /Users/fabijanpavosevic/research/calculations/test-orca/lda/xcfun/lib

OBJSUF = o
DEPSUF = d
CXXDEPENDSUF = none
CXXDEPENDFLAGS = -M


TEST1 = hartree-fock
CXXTEST1SRC = $(TEST1).cc
CXXTEST1OBJ = $(CXXTEST1SRC:%.cc=%.$(OBJSUF))
CXXTEST1DEP = $(CXXTEST1SRC:%.cc=%.$(DEPSUF))

check:: check1 

check1::

check1:: $(TEST1)
	./$^ $(SRCDIR)/h2o.xyz 

mp2.o: mp2.cc
	g++ -c -I /usr/local/include/eigen3 $^ -std=c++11

ccsd.o: ccsd.cc
	g++ -c -I /usr/local/include/eigen3 $^ -std=c++11

$(TEST1): $(CXXTEST1OBJ) mp2.o ccsd.o $(COMPILER_LIB) $(COMPUTE_LIB)
	g++ $^ -o $@
#	$(LD) $^ -o $@ $(LDFLAGS) $^ $(SYSLIBS)

# Source files for timer and tester are to be compiled using CXXGEN
$(TEST1): CXX=$(CXXGEN)
$(TEST1): CXXFLAGS=$(CXXGENFLAGS)
$(TEST1): LD=$(CXXGEN)

clean::
	-rm -rf $(TEST1) *.o *.d

distclean:: realclean
	-rm -rf $(TOPDIR)/include/libint2/boost

realclean:: clean

targetclean:: clean

$(TOPDIR)/include/libint2/boost/preprocessor.hpp: $(SRCDIR)/$(TOPDIR)/external/boost.tar.gz
	gunzip -c $(SRCDIR)/$(TOPDIR)/external/boost.tar.gz | tar -xf - -C $(TOPDIR)/include/libint2

depend:: $(CXXTEST1DEP) 

ifneq ($(CXXDEPENDSUF),none)
%.d:: %.cc $(TOPDIR)/include/libint2/boost/preprocessor.hpp
	$(CXXDEPEND) $(CXXDEPENDFLAGS) -c $(CPPFLAGS) $(CXXFLAGS) $< > /dev/null
	sed 's/^$*.o/$*.$(OBJSUF) $*.d/g' < $(*F).$(CXXDEPENDSUF) > $(@F)
	/bin/rm -f $(*F).$(CXXDEPENDSUF)
else
%.d:: %.cc $(TOPDIR)/include/libint2/boost/preprocessor.hpp
	$(CXXDEPEND) $(CXXDEPENDFLAGS) -c $(CPPFLAGS) $(CXXFLAGS) $< | sed 's/^$*.o/$*.$(OBJSUF) $*.d/g' > $(@F)
endif

-include $(CXXTEST1DEP)
