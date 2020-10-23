#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define blockX  32
#define blockY  1
#define blockZ 32

// time stamp function in seconds
double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}
// host side matrix calculation
void h_compute_result(float *A, float *B, int n){
	float* ia = A, *ib =B;

	for (int iz =1; iz<n-1; iz++){
		for (int iy =1; iy<n-1; iy++){
            for (int ix =1; ix<n-1; ix++){
			ia[iz*(n)*(n) + iy * (n) + ix] = (ib[iz*(n)*(n) + iy * n + ix + 1] + ib[iz*(n)*(n) + iy * n + ix - 1] +
                                                    ib[iz*(n)*(n) + (iy + 1) * n + ix] + ib[iz*(n)*(n) + (iy - 1) * n + ix] +
                                                    ib[(iz + 1)*(n)*(n) + iy * n + ix] + ib[(iz - 1)*(n)*(n) + iy * n + ix]);

#ifdef DEBUG
//                if (iz == 31 && ix == 1 && iy == 1){
//			    printf("the CPU additon %lf\n", ia[iz*(n)*(n) + iy * (n) + ix]);}


#endif
		}
	}
 }
}

//host side matrix comparison

int h_compareResult(float *h_A, float *d_A, int noElems){
	float *host_a = h_A,*device_a = d_A;
	for (int i =0; i<noElems; i++){
		if (*(host_a) != *(device_a)){
#ifdef DEBUG

			printf("the i = %d\n", i);
			printf("the data of CPU is %lf\n", *(host_a));
			printf("the data of GPU is %lf\n", *(device_a));

#endif
			return 1;
		}
#ifdef DEBUG
//        if (i == 30101){            printf("the i = %d\n", i);
//			printf("the data of CPU is %lf\n", *(host_a));
//			printf("the data of GPU is %lf\n", *(device_a));}


#endif
        host_a++;
        device_a++;
	}
	return 0;
 }

// device-side matrix addition
__global__ void f_jocobiRelaxation( float *A, float *B, int n,int zEnd, int debug){
	// kernel code might look something like this
	// but you may want to pad the matrices and index into them accordingly
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
	int idx = iz*n*n + iy*n + ix ;
	if( (ix<n-1) && (iy<n-1) && (iz<zEnd) &&(ix>=1) && (iy>=1) && (iz>=1)){

        A[idx] =( B[idx+1] + B[idx - 1] + B[idx + n] + B[idx - n] + B[idx + n * n] + B[idx - n * n] );
#ifdef DEBUG
        //printf("the addition at idx = %d in device: %.6f \n",idx, A[idx]);
        if (idx == 10101 && debug == 0){
            printf("the addition at idx = %d in device: %.6f \n",idx, A[idx]);
            printf("at idx = %d in device, ix = %d, iy = %d, iz = %d \n",idx, ix, iy, iz);
            printf("the  idx + 1  = %lf in device \n", B[idx+1]);
            printf("the  idx - 1  = %lf in device \n", B[idx-1]);
        }
#endif
	}

}

void initDataB(float* ib, int n){
    for (int iz =0; iz<n; iz++){
        for (int iy =0; iy<n; iy++){
            for (int ix =0; ix<n; ix++){
                ib[iz*(n)*(n) + iy * (n) + ix] =  ( ix + iy + iz);
            }
        }
    }
}
void initDataA(float* ia, int noElem){
    for (int i=0; i<noElem; i++) {
        ia[i] = 0;
    }
}
float SumDataA(float* ia, int noElem){
    float r = 0;
    for (int i=0; i<noElem; i++) {
        r += ia[i]/10000;
    }
    return r;
}

int main(int argc, char* argv[]){

	if(argc != 2){
		printf("Error: wrong number of argument\n");
		exit(0);
	}

	int n = atoi(argv[1]);


	// do the input argument check.
	if(n<=2){
		printf("Error: input arguement can't be zero or negative\n");
		exit(0);
	}

	int noElems = n * n * n;
	int bytes = noElems * sizeof(float);
    printf("number of element is %d \n", noElems);


	// alloc memeory host-side
	float *h_hA  = (float*) malloc(bytes); // host result


	float *h_B; // host result
    float *h_dA;

	//pin memory in host side
	cudaHostAlloc((void**)&h_B, bytes, 0);
    cudaHostAlloc((void**)&h_dA, bytes, 0);

	// init matrices with random data
    initDataA(h_hA, noElems);
    initDataA(h_dA, noElems);
    initDataB(h_B, n);
	//alloc memory device-side
	float *d_A, *d_B;
	cudaMalloc( &d_A, bytes);
	cudaMalloc( &d_B, bytes);



    // getting host side result
    h_compute_result( h_hA, h_B, n) ;


    int minimumBytesPerStream = n * sizeof(float) * blockZ * n;
//    while (minimumBytesPerStream < 4194304*16){	// 4194304 is when 1024(thread) * 2 (blocks/SMS) * 16 (SMS) * 4 (sizeof(Float)) * 2 (Two float number required for addition), we want data transfer is multiple of this number
//        minimumBytesPerStream = minimumBytesPerStream * 2;
//    }

    int bytesPerStream = minimumBytesPerStream;
    // calculating number of Streams according to the calculated bytesPerStream
    int NSTREAMS = bytes/((blockZ-2)*n*n*sizeof(float)) ;
    // if there is data remain where they are not multiple of bytesPerStream
    int remainZ = (n)%((blockZ-2));
    // initialize the stream array
    cudaStream_t stream[NSTREAMS+2];
    // input the pre-calculated block size and calculate the grid size
    dim3 block( blockX, 1, blockZ ) ; // you will want to configure this
    dim3 grid( (n + block.x-1)/block.x, (n + block.y-1)/block.y, 1 ) ;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    float AccumulateKernelTime = 0;
    int i,offset, debug = -1;

    double timeStampA = getTimeStamp() ;
    for( i = 1; i <=NSTREAMS; i++ ){
        // create stream
        cudaStreamCreate(&stream[i]);
        //calculating offset
        //offset = ((i-1) * bytesPerStream /4 - (i - 1) * 2 * n* n );
        offset = (i-1) * (blockZ-2) * n * n;

        //Asynch copy data from host to device
        cudaMemcpyAsync(&d_B[offset],&h_B[offset],bytesPerStream, cudaMemcpyHostToDevice, stream[i]);
        //record the timestamp before kernel invoke
        cudaEventRecord(start);
        //invoke kernel
        f_jocobiRelaxation<<<grid, block,0,stream[i]>>>( &d_A[offset], &d_B[offset], n,(blockZ-1),debug ) ;
        //record the timestamp before kernel invoke
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        // write down the difference
        cudaEventElapsedTime(&milliseconds, start, stop);
        // add this time to accumulated time
        AccumulateKernelTime += milliseconds;
        //Asynch copy data from device back to host
        cudaMemcpyAsync(&h_dA[offset ],&d_A[offset],bytesPerStream, cudaMemcpyDeviceToHost,stream[i]);

    }
    if(remainZ != 0){
        offset = ((i-1) * (blockZ-2)) * n * n;
        int remainBytes = (remainZ) *n * n * sizeof(float);
        cudaStream_t last;
        cudaStreamCreate(&last);
        dim3 block( (blockX), blockY, (remainZ) ) ; // you will want to configure this
        dim3 grid( (n + block.x-1)/block.x, (n + block.y-1)/block.y, 1 ) ;
        cudaMemcpyAsync(&d_B[offset],&h_B[offset],remainBytes, cudaMemcpyHostToDevice, last);
        cudaEventRecord(start);
        debug = 0;
        f_jocobiRelaxation<<<grid, block,0,last>>>( &d_A[offset], &d_B[offset], n,remainZ - 1,debug ) ;

        cudaEventRecord(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        AccumulateKernelTime += milliseconds;
        cudaMemcpyAsync(&h_dA[offset],&d_A[offset],remainBytes, cudaMemcpyDeviceToHost,last);
        cudaStreamSynchronize(last);
    }

    for(i = 1; i <=NSTREAMS; i++ ){
        cudaStreamSynchronize(stream[i]);
    }
    double timeStampB = getTimeStamp() ;
    cudaDeviceSynchronize() ;

    printf("time taken for kernel = %lf ms. \n", AccumulateKernelTime   );

    double timeStampF = getTimeStamp() ;
    printf("toal time taken for GPU = %lf ms. \n",(timeStampB - timeStampA)*1000);
    if (h_compareResult(h_hA, h_dA, noElems) == 1){
        printf("Error: the two results don't match\n");
    }
    else{
        printf("the result match\n");
    }
    cudaFree( d_A ) ; cudaFree( d_B );
    cudaFreeHost(h_B);
    cudaFreeHost(h_dA);
    cudaDeviceReset() ;
    free(h_hA);

/*
	int i;
	// calculating minimum bytes each Stream should take according to the calculated block_y

	int minimumBytesPerStream = nx * sizeof(float) * 4 * block_y;
	while (minimumBytesPerStream < 4194304*16){	// 4194304 is when 1024(thread) * 2 (blocks/SMS) * 16 (SMS) * 4 (sizeof(float)) * 2 (Two float number required for addition), we want data transfer is multiple of this number
		minimumBytesPerStream = minimumBytesPerStream * 2;
	}
	// yPerStream is mutiple of 4 so every thread can process 4 different y in one stream
	//int yPerStream = minimumBytesPerStream/ nx;
	// calculating bytes each Stream according to the calculated yPerStream
	//int bytesPerStream = nx * sizeof(float) * yPerStream;
	// calculating number of Streams according to the calculated bytesPerStream
	//int NSTREAMS = bytes/bytesPerStream;
	// if there is data remain where they are not multiple of bytesPerStream
	//int remainBytes = bytes%bytesPerStream;
	// initialize the stream array
	//cudaStream_t stream[NSTREAMS+1];
	// input the pre-calculated block size and calculate the grid size
	//dim3 block( block_x, block_y ) ; // you will want to configure this
	//dim3 grid( (nx + block.x-1)/block.x, (bytesPerStream/(sizeof(float) * nx) + block.y-1)/block.y ) ;

#ifdef DEBUG
	printf("the final bytesPerStream is = %d\n", bytesPerStream);

	printf("the remainBytes is = %d\n", remainBytes);
	printf("the final block size is x = %d and y = %d \n",block_x, block_y);
	printf("the final grid dimension is x = %d and y = %d \n",(nx + block_x-1)/block_x, (yPerStream + block.y-1)/block.y );
#endif
	// initialize the event for calculating accumulate kernel time.
	// NOTE: if we don't need to calculating the accumulate kernel time, the total time is at least 10% faster.
	// But  kernel time is important to show.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float timeStampA = getTimeStamp() ;
	float timeStampB= getTimeStamp() ;
	float milliseconds;
	float AccumulateKernelTime = 0;
	for(i = 1; i <=NSTREAMS; i++ ){
		// create stream
		cudaStreamCreate(&stream[i]);
		//calculating offset
		int offset = (i-1) * bytesPerStream/4;
		//Asynch copy data from host to device
		cudaMemcpyAsync(&d_A[offset],&h_A[offset],bytesPerStream, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(&d_B[offset],&h_B[offset],bytesPerStream, cudaMemcpyHostToDevice, stream[i]);
		//record the timestamp before kernel invoke
		cudaEventRecord(start);
		//invoke kernel
		f_addmat4<<<grid, block,0,stream[i]>>>( &d_A[offset], &d_B[offset], &d_C[offset], nx, bytesPerStream/(4* sizeof(float) * nx), bytesPerStream/(4* sizeof(float)) ) ;
		//record the timestamp before kernel invoke
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		// write down the difference
		cudaEventElapsedTime(&milliseconds, start, stop);
		// add this time to accumulated time
		AccumulateKernelTime += milliseconds/1000;
		//Asynch copy data from device back to host
		cudaMemcpyAsync(&h_dC[offset],&d_C[offset],bytesPerStream, cudaMemcpyDeviceToHost,stream[i]);
	}
	// if there is remaining byte, we do the process one more time
	if(remainBytes != 0){
		int remainEle = remainBytes/4;
		cudaStream_t last;
		cudaStreamCreate(&last);
		int offset = NSTREAMS * bytesPerStream/4;
		cudaMemcpyAsync(&d_A[offset],&h_A[offset],remainBytes, cudaMemcpyHostToDevice, last);
		cudaMemcpyAsync(&d_B[offset],&h_B[offset],remainBytes, cudaMemcpyHostToDevice, last);
		dim3 grid( (nx + block.x-1)/block.x, (remainEle/nx + block.y-1)/block.y ) ;
		cudaEventRecord(start);
		f_addmat<<<grid, block,0,last>>>( &d_A[offset], &d_B[offset], &d_C[offset], nx, remainEle/nx ) ;
		cudaEventRecord(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		AccumulateKernelTime += milliseconds/1000;
		cudaMemcpyAsync(&h_dC[offset],&d_C[offset],remainBytes, cudaMemcpyDeviceToHost,last);
		cudaStreamSynchronize(last);
	}

	float timeStampC = getTimeStamp() ;
	//wait for all stream finish the job
	for(i = 1; i <=NSTREAMS; i++ ){
		cudaStreamSynchronize(stream[i]);
	}

	cudaDeviceSynchronize() ;
	//time where device side jobs have been finished
	float timeStampD = getTimeStamp() ;

	// free some Host and GPU resources that are not needed anymore
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFree( d_A ) ; cudaFree( d_B ) ; cudaFree( d_C ) ;

#ifdef DEBUG
	float * ptr;
	int n;
	ptr = h_dC;
	n = 0;
	ptr = ptr + n;
	printf("the data of GPU at index %d before comparison is %.6f\n", n,*(ptr));
#endif
	//h_compareResult compares the result computed by host and result computed by device
	//if any element is not same, the function will return 1, otherwise print out the time
	if (h_compareResult(h_hC,h_dC,noElems) == 1){
			printf("Error: the two results don't match\n");
	}
	else{
		//printf(" %.6f  %.6f %.6f %.6f\n",timeStampD - timeStampA,timeStampB - timeStampA, AccumulateKernelTime, timeStampD - timeStampC  );
		printf(" %.6f  %.6f %.6f %.6f\n",timeStampD - timeStampA,timeStampB - timeStampA, AccumulateKernelTime, timeStampD - timeStampC  );
	}
	// free rest Host Side Resources
	cudaFreeHost(h_dC);
	free(h_hC);
	cudaDeviceReset();
 */
}
