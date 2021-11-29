libspmm.so: kernel.cu kernel.h Makefile
	nvcc -arch=sm_52 -DB32 -Xptxas -O3  --use_fast_math --compiler-options '-fPIC -O3' -o libspmm.so --shared kernel.cu
clean:
	rm libspmm.so

#nvcc -arch=sm_52 -DB32 -G --use_fast_math --compiler-options '-fPIC -g' -o libspmm.so --shared kernel.cu
