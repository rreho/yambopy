# Wannier Tools - Compilation

Be aware of Fortran type mismatches and inconsistent module handling. Compilation may require manual tweaking.
1. Clone [WannierTools repo](https://github.com/quanshengwu/wannier_tools)
```
git clone https://github.com/quanshengwu/wannier_tools.git
cd wannier_tools
```
2. Read the INSTALL file and follow those steps
3. Modules can be loaded with `module load 2022`, `module load foss/2022a`

```
Currently Loaded Modules:
  1) 2022                              19) gompi/2022a
  2) GCCcore/11.3.0                    20) FFTW.MPI/3.3.10-gompi-2022a
  3) zlib/1.2.12-GCCcore-11.3.0        21) ScaLAPACK/2.2.0-gompi-2022a-fb
  4) binutils/2.38-GCCcore-11.3.0      22) foss/2022a
  5) GCC/11.3.0                        23) Szip/2.1.1-GCCcore-11.3.0
  6) numactl/2.0.14-GCCcore-11.3.0     24) HDF5/1.12.2-gompi-2022a
  7) XZ/5.2.5-GCCcore-11.3.0           25) ELPA/2021.11.001-foss-2022a
  8) libxml2/2.9.13-GCCcore-11.3.0     26) libxc/5.2.3-GCC-11.3.0
  9) libpciaccess/0.16-GCCcore-11.3.0  27) QuantumESPRESSO/7.1-foss-2022a
 1)  hwloc/2.7.1-GCCcore-11.3.0        28) cURL/7.83.0-GCCcore-11.3.0
 2)  OpenSSL/3                         29) gzip/1.12-GCCcore-11.3.0
 3)  libevent/2.1.12-GCCcore-11.3.0    30) lz4/1.9.3-GCCcore-11.3.0
 4)  UCX/1.12.1-GCCcore-11.3.0         31) zstd/1.5.2-GCCcore-11.3.0
 5)  libfabric/1.15.1-GCCcore-11.3.0   32) bzip2/1.0.8-GCCcore-11.3.0
 6)  UCC/1.0.0-GCCcore-11.3.0          33) netCDF/4.9.0-gompi-2022a
 7)  OpenMPI/4.1.4-GCC-11.3.0          34) netCDF-Fortran/4.6.0-gompi-2022a
 8)  FlexiBLAS/3.2.0-GCC-11.3.0        35) OpenBLAS/0.3.20-GCC-11.3.0
 9)  FFTW/3.3.10-GCC-11.3.0
```
1. Here is the Makefile I used on Snellius. Go to `src` and add edit the `Makefile` file. Note the custom `FFLAGS` and `OBJS` list:

```
OBJ =  module.o sparse.o wt_aux.o math_lib.o symmetry.o readHmnR.o inverse.o proteus.o \
       eigen.o ham_qlayer2qlayer.o psi.o unfolding.o rand.o \
                 ham_slab.o ham_bulk.o ek_slab.o ek_bulk_polar.o ek_bulk.o \
       readinput.o fermisurface.o surfgreen.o surfstat.o \
                 mat_mul.o ham_ribbon.o ek_ribbon.o \
       fermiarc.o berrycurvature.o \
                 wanniercenter.o dos.o  orbital_momenta.o\
                 landau_level_sparse.o landau_level.o lanczos_sparse.o \
                 berry.o wanniercenter_adaptive.o \
                 effective_mass.o findnodes.o \
                 sigma_OHE.o sigma.o Boltz_transport_anomalous.o \
                 2D_TSC.o optic.o orbital_hall.o\
        main.o

# compiler, here mpif90 should be comipled with gfortran
F90  = mpif90 -cpp -DMPI -ffree-line-length-none

INCLUDE =
WFLAG = #-Wall -Wextra -Warray-temporaries -Wconversion -fimplicit-none \
       -fbacktrace -ffree-line-length-0 -fcheck=all -finit-real=nan \
       -ffpe-trap=zero,overflow,underflow

OFLAG = -g
FFLAG = -O2 -fallow-argument-mismatch -ffree-line-length-none
LFLAG = $(OFLAG)


# blas and lapack libraries
LIBS =  -L/sw/arch/RHEL8/EB_production/2022/software/OpenBLAS/0.3.20-GCC-11.3.0/lib -lopenblas

# Intel MKL also supports with gfortran comipler
# dynamic linking
#LIBS = -L/${MKLROOT}/lib/intel64 -lmkl_core -lmkl_sequential -lmkl_intel_lp64 -lpthread


main : $(OBJ)
        $(F90) $(LFLAG) $(OBJ) -o wt.x $(LIBS)
        cp -f wt.x ../bin

.SUFFIXES: .o .f90

.f90.o :
        $(F90) $(FFLAG) $(INCLUDE) -c $*.f90

clean :
        rm -f *.o *.mod *~ wt.x
```

4. Run `make`