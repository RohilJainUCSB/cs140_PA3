#Make sure you load these modules before compiling with this makefile
#	module load slurm
#	module load gpu/0.15.4 gcc/7.2.0 cuda/11.0.2


NVCC      = nvcc
NVCCFLAGS =  -arch=sm_70 -O3
LDFLAGS   =  -arch=sm_70 -O3  -lcublas
CFLAGS    =  


OBJECTS	  = compare.o 

TARGET		= compare 

all:  $(TARGET)

compare: compare.cu
	$(NVCC) -o $@ compare.cu $(LDFLAGS)

status:
	squeue -u `whoami`

runcompare: compare
	sbatch -v compare.sh


clean:
	rm  *.o $(TARGET)

cleanlog:
	rm  *.out
