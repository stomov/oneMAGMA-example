CPU		:= dpcpp -DMAGMA_HAVE_SYCL -O3 -fopenmp
CUDAGPU		:= clang++ -DMAGMA_HAVE_SYCL -O3 #-fopenmp
CUDAGPU         := dpcpp -DMAGMA_HAVE_SYCL -O3 
#CUDAGPUFLAGS	:= -fsycl -fsycl-targets=nvptx64-nvidia-cuda
INCLUDE 	:= -I/home/user1/anna/magmaMtxMul/dpct_output/include 
INCLUDE         := -I/home/tomov/oneAPI/mtxMtxMulCnvt/dpct_output/include
INCLUDE         := -I./ -I$MKLROOT/include -L$MKLROOT/lib/intel64

#TESTA		:= -wA=16 -wB=16 -hA=16 -hB=16
TESTB           := -wA=32 -wB=32 -hA=32 -hB=32
TESTC           := -wA=64 -wB=64 -hA=64 -hB=64
TESTD           := -wA=128 -wB=128 -hA=128 -hB=128
TESTE           := -wA=256 -wB=256 -hA=256 -hB=256
TESTF           := -wA=512 -wB=512 -hA=512 -hB=512
TESTG           := -wA=1024 -wB=1024 -hA=1024 -hB=1024
TESTH           := -wA=1536 -wB=1536 -hA=1536 -hB=1536
TESTJ		:= -wA=2048 -wB=2048 -hA=2048 -hB=2048
TESTK           := -wA=3072 -wB=3072 -hA=3072 -hB=3072
TESTL		:= -wA=4096 -wB=4096 -hA=4096 -hB=4096
TESTM           := -wA=5120 -wB=5120 -hA=5120 -hB=5120
TESTN		:= -wA=6144 -wB=6144 -hA=6144 -hB=6144
TESTP           := -wA=7168 -wB=7168 -hA=7168 -hB=7168
TESTQ		:= -wA=8192 -wB=8192 -hA=8192 -hB=8192

cuda = -DMAGMA_TUNING -DDIM_X=16 -DDIM_Y=16 -DBLK_M_nn=96 -DBLK_N_nn=96 -DBLK_K_nn=16 -DDIM_XA=32 -DDIM_YA=8  -DDIM_XB=8  -DDIM_YB=32
hip  = -DMAGMA_TUNING -DDIM_X=16 -DDIM_Y=16 -DBLK_M_nn=96 -DBLK_N_nn=64 -DBLK_K_nn=8  -DDIM_XA=32 -DDIM_YA=8  -DDIM_XB=8  -DDIM_YB=32
ker1 = -DMAGMA_TUNING -DDIM_X=16 -DDIM_Y=16 -DBLK_M_nn=64 -DBLK_N_nn=64 -DBLK_K_nn=8  -DDIM_XA=32 -DDIM_YA=8  -DDIM_XB=8  -DDIM_YB=32
ker2 = -DMAGMA_TUNING -DDIM_X=16 -DDIM_Y=16 -DBLK_M_nn=64 -DBLK_N_nn=64 -DBLK_K_nn=8  -DDIM_XA=32 -DDIM_YA=8  -DDIM_XB=8  -DDIM_YB=32
ker3 = -DMAGMA_TUNING -DDIM_X=16 -DDIM_Y=16 -DBLK_M_nn=64 -DBLK_N_nn=64 -DBLK_K_nn=16 -DDIM_XA=32 -DDIM_YA=8  -DDIM_XB=8  -DDIM_YB=32
ker4 = -DMAGMA_TUNING -DDIM_X=32 -DDIM_Y=32 -DBLK_M_nn=64 -DBLK_N_nn=64 -DBLK_K_nn=32 -DDIM_XA=32 -DDIM_YA=32 -DDIM_XB=32 -DDIM_YB=32
ker5 = -DMAGMA_TUNING -DDIM_X=32 -DDIM_Y=32 -DBLK_M_nn=96 -DBLK_N_nn=96 -DBLK_K_nn=32 -DDIM_XA=32 -DDIM_YA=32 -DDIM_XB=32 -DDIM_YB=32
ker6 = -DMAGMA_TUNING -DDIM_X=8  -DDIM_Y=8  -DBLK_M_nn=96 -DBLK_N_nn=96 -DBLK_K_nn=8  -DDIM_XA=32 -DDIM_YA=2  -DDIM_XB=2  -DDIM_YB=32
ker7 = -DMAGMA_TUNING -DDIM_X=16 -DDIM_Y=16 -DBLK_M_nn=32 -DBLK_N_nn=32 -DBLK_K_nn=8  -DDIM_XA=32 -DDIM_YA=8  -DDIM_XB=8  -DDIM_YB=32
ker8 = -DMAGMA_TUNING -DDIM_X=16 -DDIM_Y=16 -DBLK_M_nn=64 -DBLK_N_nn=64 -DBLK_K_nn=8  -DDIM_XA=32 -DDIM_YA=8 -DDIM_XB=8 -DDIM_YB=32

cpu:	
	$(CPU) -I./ -c -o sgemm_fermi.dp.o sgemm_fermi.dp.cpp $(cuda)
	$(CPU) matrixMul.dp.cpp -o intelCpuExec $(INCLUDE) sgemm_fermi.dp.o

gpu:
	$(CUDAGPU) $(CUDAGPUFLAGS) -I./ -c -o sgemm_fermi.dp.o sgemm_fermi.dp.cpp $(cuda)
	$(CUDAGPU) $(CUDAGPUFLAGS) matrixMul.dp.cpp -o cudaGpuExec $(INCLUDE) sgemm_fermi.dp.o -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

runCpu:
	#./intelCpuExec $(TESTA)
	./intelCpuExec $(TESTB)
	./intelCpuExec $(TESTC)
	./intelCpuExec $(TESTD)
	./intelCpuExec $(TESTE)
	./intelCpuExec $(TESTF)
	./intelCpuExec $(TESTG)
	./intelCpuExec $(TESTH)
	./intelCpuExec $(TESTJ)
	./intelCpuExec $(TESTK)
	./intelCpuExec $(TESTL)
	./intelCpuExec $(TESTM)
	./intelCpuExec $(TESTN)
	./intelCpuExec $(TESTP)
	./intelCpuExec $(TESTQ)

runGpu:
	#./cudaGpuExec $(TESTA)
	./cudaGpuExec $(TESTB)
	./cudaGpuExec $(TESTC)
	./cudaGpuExec $(TESTD)
	./cudaGpuExec $(TESTE)
	./cudaGpuExec $(TESTF)
	./cudaGpuExec $(TESTG)
	./cudaGpuExec $(TESTH)
	./cudaGpuExec $(TESTJ)
	./cudaGpuExec $(TESTK)
	./cudaGpuExec $(TESTL)
	./cudaGpuExec $(TESTM)
	./cudaGpuExec $(TESTN)
	./cudaGpuExec $(TESTP)
	./cudaGpuExec $(TESTQ)

clean:
	rm *Exec

