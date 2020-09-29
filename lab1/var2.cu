#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define DEFAULT_ROW  16384
#define DEFAULT_COL  16384
// time stamp function in seconds
double getTimeStamp() {
	struct timeval tv ;
	gettimeofday( &tv, NULL ) ;
	return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}
// host side matrix addition
void h_addmat(float *A, float *B, float *C, int nx, int ny){ 
	float* ia = A, *ib =B, *ic =C;
	for (int iy =0; iy<ny; iy++){
		for (int ix =0; ix<nx; ix++){
			
			ic[ix] = ia[ix] + ib[ix];
			//if (iy*nx + ix == 67133440) printf("the addition in host: %.6f + %.6f = %.6f\n",ia[ix],ib[ix],ic[ix]);
			
		}
		ia += nx;
		ib += nx;
		ic += nx;
	}
 }
//host side matrix comparison
int h_compareResult(float *h_C, float *d_C, int noElems){ 
	float *host_c = h_C,*device_c = d_C;
	for (int i =0; i<noElems; i++){
		if (*(host_c) != *(device_c)){
#ifdef DEBUG

			printf("the i = %d\n", i);
			printf("the data of CPU is %.6f\n", *(host_c));
			printf("the data of GPU is %.6f\n", *(device_c));

#endif
			return 1;
		} 
		host_c++;
		device_c++;
	}
	return 0;
 }
// device-side matrix addition
__global__ void f_addmat( float *A, float *B, float *C, int nx, int ny ){
	// kernel code might look something like this
	// but you may want to pad the matrices and index into them accordingly
	int ix = threadIdx.x + blockIdx.x*blockDim.x ;
	int iy = threadIdx.y + blockIdx.y*blockDim.y ;
	int idx = iy*nx + ix ;
	if( (ix<nx) && (iy<ny) )
	C[idx] = A[idx] + B[idx] ;
	//if (idx == 0) printf("the addition in device: %.6f + %.6f = %.6f\n",A[idx],B[idx],C[idx]);
}

void initData(float* add, int noElems){
	int i;
	float a = 5.0;
	for (i=0; i< noElems; i++){
		*(add++) = ((float)rand()/(float)(RAND_MAX)) * a;
	}

}

int main(int argc, char* argv[]){

	if(argc != 3){
		printf("Error: wrong number of argument\n");
		exit(0);
	}

	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);

	int noElems = nx * ny;
	int bytes = noElems * sizeof(float);
#ifdef DEBUG
	printf("the input row # is %d\n",nx);
	printf("the input col # is %d\n",ny);
	printf("the noElems is %d\n",noElems);
	printf("the bytes is %d\n",bytes);
#endif
	// padding


	// alloc memeory host-side
	float *h_A = (float*) malloc(bytes);
	float *h_B = (float*) malloc(bytes);
	float *h_dC = (float*) malloc(bytes);	 //gpu result
	float *h_hC = (float*) malloc(bytes); // host result
	//float *h_dC;
	//cudaHostAlloc(&h_dC, bytes, 0);	


	// init matrices with random data
	initData(h_A, noElems);
	initData(h_B, noElems);

	//alloc memeory device-side
	float *d_A, *d_B, *d_C;
	cudaMalloc( &d_A, bytes);
	cudaMalloc( &d_B, bytes);
	cudaMalloc( &d_C, bytes);

	double timeStampA = getTimeStamp() ;

	//transfer data to dev
	cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice) ;
	// note that the transfers would be twice as fast if h_A and h_B
	// matrices are pinned

	double timeStampB = getTimeStamp() ;

	// invoke Kernel
	int block_x=1, block_y;
	if (ny < 1024){
		
		block_y = ny;
		while ((ny + block_y-1)/block_y > 65535){
			block_y ++;
		}
		while (block_x * block_y > 1024){
			block_x --;
		}
	}
	else{
		block_y = 1024;
	}
#ifdef DEBUG
	printf("the final block size is x = %d and y = %d \n",block_x, block_y);
	printf("the final grid dimension is x = %d and y = %d \n",(nx + block_x-1)/block_x, (ny + block_y-1)/block_y);
#endif
	dim3 block( block_x, block_y ) ; // you will want to configure this
	dim3 grid( (nx + block.x-1)/block.x, (ny + block.y-1)/block.y ) ;
	f_addmat<<<grid, block>>>( d_A, d_B, d_C, nx, ny ) ;

	cudaDeviceSynchronize() ;

	double timeStampC = getTimeStamp() ;

	//copy data back
	cudaMemcpy( h_dC, d_C, bytes, cudaMemcpyDeviceToHost ) ;
	double timeStampD = getTimeStamp() ;
	// free GPU resources
	cudaFree( d_A ) ; cudaFree( d_B ) ; cudaFree( d_C ) ;
	cudaDeviceReset() ;
	// check result
	h_addmat( h_A, h_B, h_hC, nx, ny ) ;
	// h_dC == h+hC???
	free(h_A);
	free(h_B);
#ifdef DEBUG
	float *ptr;
	ptr = h_dC;
	int n = 0;
	ptr = ptr + n;
	printf("the data of GPU at index %d before comparison is %.6f\n", n,*(ptr));
#endif
	if (h_compareResult(h_hC,h_dC,noElems) == 1){
			printf("the two results don't match\n");
	}
	else{
		printf("totoal= %.6f CPU_GPU_transfer = %.6f kernel =%.6f GPU_CPU_transfer= %.6f\n",timeStampD - timeStampA,timeStampB - timeStampA, timeStampC - timeStampB, timeStampD - timeStampC  );
		//printf("CPU_GPU_transfer_time = %.6f\n",timeStampB - timeStampA );
		//printf("kernel_time = %.6f\n",timeStampC - timeStampB );
		//printf("GPU_CPU_transfer_time = %.6f\n",timeStampD - timeStampC );
	}

	cudaFreeHost(h_hC);

}

