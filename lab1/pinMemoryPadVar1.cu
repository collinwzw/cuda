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
void h_addmat(float *A, float *B, float *C, int nx, int ny, int new_nx){ 
	float* ia = A, *ib =B, *ic =C;
	for (int iy =0; iy<ny; iy++){
		for (int ix =0; ix<new_nx; ix++){
			
			if (ix < nx)*(ic++) = *(ia++) + *(ib++);
			else{
				ia++;
				ib++;
			}
			//if (iy*nx + ix == 67133440) printf("the addition in host: %.6f + %.6f = %.6f\n",ia[ix],ib[ix],ic[ix]);
			
		}
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
	//printf("the addition at idx = %d in device: %.6f + %.6f = %.6f\n",idx, A[idx],B[idx],C[idx]);
}

void initData(float* add, int new_nx,int nx, int ny){
	int row,col;
	float a = 5.0;
	for (row=0; row< ny; row++){
		for (col=0; col< new_nx; col++){
			if (col < nx) *(add++) = ((float)rand()/(float)(RAND_MAX)) * a;
			else *(add++) = 0;
		}
	}

}

void removePading(float* h_dC, float* h_temp_dC, int nx,int ny,int new_nx){
	int row,col;
	int count=0;
	float *r_padding = h_temp_dC, *r = h_dC;
	for (row=0; row< ny; row++){
		for (col=0; col< new_nx; col++){
			
			if (col < nx){
			 r[count] = r_padding[row * new_nx + col];
			 //printf("at index %d, the value transfer is %f\n", count, r_padding[row * nx + col]);
			 count ++;


			}
		
			
		}
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
	int new_nx;
	int new_bytes;
	int new_noElems;
	// GTX memeory access bandwidth = 256 bits = 32 Bytes = 8 float element
	if (nx%8 != 0){
		
		int numberOfPaddingColAdded = 8 - nx%8;
		new_nx = nx + numberOfPaddingColAdded;
		
	}
	else{
		new_nx = nx;
	}
	printf("the nx + padding = %d\n", new_nx);
	new_noElems = new_nx * ny;
	new_bytes = new_noElems * sizeof(float);

	// alloc memeory host-side
	float *h_A;
	float *h_B;
	float *h_temp_dC;// = (float*) malloc(bytes);	 //gpu result
	float *h_hC = (float*) malloc(bytes); // host result

	cudaHostAlloc((void**)&h_A, new_bytes, 0);	
	cudaHostAlloc((void**)&h_B, new_bytes, 0);	
	cudaHostAlloc((void**)&h_temp_dC, new_bytes, 0);	
	// init matrices with random data
	initData(h_A, new_nx, nx, ny);
	initData(h_B, new_nx, nx, ny);

	//alloc memeory device-side
	float *d_A, *d_B, *d_C;
	cudaMalloc( &d_A, new_bytes);
	cudaMalloc( &d_B, new_bytes);
	cudaMalloc( &d_C, new_bytes);
	// check result
	h_addmat( h_A, h_B, h_hC, nx, ny, new_nx ) ;

	//computing minimal dimension of block size y according to the spec
	int min_blocky = 1;
	while ((ny + min_blocky-1)/min_blocky > 65535){
		min_blocky ++;
	}
	int block_x, block_y = min_blocky;
	if (nx < 1024){
		// if input nx is smaller than 1024		
		block_x = nx;
		
		while (block_x > 32 && block_x %32 !=0){
			// make the block_x in multiple of 32 (warp size)
			block_x --;
		}		
	}
	else{
		block_x = 1024;

	}

	while (block_x * block_y > 1024){
		// check if the total number of thread in a block exceed 1024 or not, if yes, subtract block x by 32
		if (block_x -32 > 0) block_x = block_x - 32;
		else block_x --;
	}
	printf("the final block size is x = %d and y = %d \n",block_x, block_y);
	printf("the final grid dimension is x = %d and y = %d \n",(new_nx + block_x-1)/block_x, (ny + block_y-1)/block_y);
	double timeStampA = getTimeStamp() ;
	
	//transfer data to dev
	cudaMemcpy( d_A, h_A, new_bytes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_B, h_B, new_bytes, cudaMemcpyHostToDevice) ;
	// note that the transfers would be twice as fast if h_A and h_B

	double timeStampB = getTimeStamp() ;

	// invoke Kernel


	dim3 block( block_x, block_y) ; // you will want to configure this
	dim3 grid( (new_nx + block.x-1)/block.x, (ny + block.y-1)/block.y ) ;
#ifdef DEBUG
	printf("the final block size is x = %d and y = %d \n",block.x, block.y);
	printf("the final grid dimension is x = %d and y = %d \n",(new_nx + block.x-1)/block.x, (ny + block.y-1)/block.y);
#endif
	f_addmat<<<grid, block>>>( d_A, d_B, d_C, new_nx, ny ) ;

	cudaDeviceSynchronize() ;

	double timeStampC = getTimeStamp() ;

	//copy data back

	cudaMemcpy( h_temp_dC, d_C, new_bytes, cudaMemcpyDeviceToHost ) ;

#ifdef DEBUG
	float *ptr;
	ptr = h_temp_dC;
	int n = 32;
	ptr = ptr + n;
	printf("the data of GPU at index %d before comparison is %.6f\n", n,*(ptr));
#endif
	double timeStampD = getTimeStamp() ;
	// free GPU resources
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	float *h_dC = (float*) malloc(bytes);
	removePading(h_dC,h_temp_dC,nx,ny,new_nx);
	cudaFree( d_A ) ; cudaFree( d_B ) ; cudaFree( d_C ) ;


	//cudaDeviceReset() ;


	// h_dC == h+hC???

#ifdef DEBUG

	ptr = h_dC;
	n = 0;
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

	cudaFreeHost(h_dC);
	free(h_hC);
	cudaDeviceReset() ;

}

