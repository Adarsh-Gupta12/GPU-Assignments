#include<iostream>
#include<sys/time.h>
#include<cuda.h>
#include<stdio.h>
using namespace std;

__global__ void calculateAB(int *A, int *B, int *E, int p, int q, int r)
{
    //Using shared memory
    //As there are q write operations for calculating one element of (A*B)

	//A is coalesced, as all threads of a block is accessing same memory locations
	//it means all threads of the same warp is also accessing same memory locations
	//let's take the example of block 0
	//thread 0 will access A[0] then all the threads upto 31 will also access A[0] in the first iteration of for loop
	//then in the second iteration of for loop, all the threads will access A[1]
	//since threads are accessing same memory location, therefore A is coalesced

	//B we are accessing column wise, therefore B is also fully coalesced
	//In the first iteration of for loop, thread 0 will access first element of first column of B, 
	//thread 1 will access first element of 2nd column, which is stored next to first element of first column of B
	//likewise thread i will access first element of ith column in the first iteration of for loop, which are stored consecutively
	//similarly in the 2nd iteration of for loop, thread i will access 2nd element of column i, which are stored consecutively
	//similar thing will happen in all iterations of for loop

	//currResult is also coalesced, as we are accessing consecutive memory location across threads of the same warp

    extern __shared__ int currResult[];
    int A_RowNum = blockIdx.x;
    int B_ColNum = threadIdx.x;
    currResult[B_ColNum] = 0;
    for (int i = 0; i < q; i++)
    {
        currResult[B_ColNum] += A[A_RowNum * q + i] * B[B_ColNum+i*r];
    }
    E[A_RowNum * r+B_ColNum] = currResult[B_ColNum];
}

__global__ void calculateCDT(int *C, int *D, int *E, int p, int q, int r)
{
	//C is colasced, explaination same as A
	//currResult is also coalesced, as we are accessing consecutive memory location across threads of the same warp
    extern __shared__ int currResult1[];
    int C_RowNum = blockIdx.x;
    int D_RowNum = threadIdx.x;
    currResult1[D_RowNum] = 0;
    for (int i = 0; i < q; i++)
    {
        currResult1[D_RowNum] += C[C_RowNum * q + i] * D[D_RowNum*q+i];
    }
    E[C_RowNum * r+D_RowNum] += currResult1[D_RowNum];
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
    dim3 grid1(p, 1, 1);
    dim3 block1(r, 1, 1);
    //to calculate matrix AB
    //each thread will compute one element of A*B
    calculateAB<<<grid1, block1, r*sizeof(int)>>>(d_matrixA, d_matrixB, d_matrixE, p, q, r);
    cudaDeviceSynchronize();
    dim3 grid2(p, 1, 1);
    dim3 block2(r, 1, 1);
    calculateCDT<<<grid2, block2, r*sizeof(int)>>>(d_matrixC, d_matrixD, d_matrixE, p, q, r);
    cudaDeviceSynchronize(); 

	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
