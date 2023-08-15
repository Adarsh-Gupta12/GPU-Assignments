#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <bits/stdc++.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

struct Request
{
    int req_id;
    int req_cen;
    int req_fac;
    int req_start;
    int req_slot;
};

//My approach is to get the list of requests for each facility sorted by requestid in ascending order, for this I have used stl sort
//And once I have request list for each facility in asc order
//I am launching a thread for each facility, in each thread I'm executing all the request corresponding to that facility sequentially

//*******************************************

// Write down the kernels here

//requests array contains all the requests sorted in (ccid, fid, reqid) order, so for each facility I'm calculating what is start and end index 
// of the requests in requests array
__global__ void calculateOffset(struct Request *requests, int *offsetStart, int *offsetEnd, int R)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=R)
        return;
    int centerId = requests[id].req_cen;
    int facilityId = requests[id].req_fac;
    //offsetStart I've initialized with -1 in main function, since memset only works for 0, -1
    //but actually I want to initialize with a large number 10^6 since we are using min
    //Therefore I'm setting it to 10^6, if it is -1
    int oldVal = atomicCAS(&offsetStart[centerId*30+facilityId], -1, 1e6);

    atomicMax(&offsetEnd[centerId*30+facilityId], id+1);
    atomicMin(&offsetStart[centerId*30+facilityId], id);
}

__global__ void processRequests(struct Request *requests, int *offsetStart, int *offsetEnd, int *currCapacity, int *capacity, int n, int *succ_reqs, int *success)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n)
        return;
    if(offsetStart[id] == -1)
        return;
    //For all the requests belonging to facility number 'id'
    for(int i = offsetStart[id];i<offsetEnd[id];i++)
    {
        bool canProcessCurrRequest = true;
        for(int timeSlot = requests[i].req_start;timeSlot<(requests[i].req_start+requests[i].req_slot);timeSlot++)
        {
            //if the current capacity of facility at time slot 'timeSlot' is greater than or equal to capacity of that facility, then we can't process request 'i' 
            if(currCapacity[id*25+timeSlot] >= capacity[id])
            {
                canProcessCurrRequest = false;
                break;
            }
        }

        //if we can process the request 'i' then increment the counter of the computer center it belongs to
        if(canProcessCurrRequest)
        {
            atomicInc((unsigned *)&succ_reqs[requests[i].req_cen], INT_MAX);
            atomicInc((unsigned *)&success[0], INT_MAX);
            //Since request 'i' is successful, it will occupy 1 capacity of the facility 'id' from 'req_start[i]' to 'req_start[i]+req_slot[i]'
            for(int timeSlot = requests[i].req_start;timeSlot<(requests[i].req_start+requests[i].req_slot);timeSlot++)
                currCapacity[id*25+timeSlot]++;
        }   
    }
}

//***********************************************

//comparator for sort
static bool comp(Request &req1, Request &req2)
{
    if(req1.req_cen > req2.req_cen)
        return false;
    if(req1.req_cen == req2.req_cen)
    {
        if(req1.req_fac > req2.req_fac)
            return false;
    }
    else
        return true;
    if(req1.req_fac == req2.req_fac)
        return req1.req_id <= req2.req_id;
    return true;
}

int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      int k1=i*max_P , k2 = i*max_P;
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    struct Request *requests;

    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    requests = (struct Request *)malloc(R * sizeof(Request));
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &requests[j].req_id);
       fscanf( inputfilepointer, "%d", &requests[j].req_cen);
       fscanf( inputfilepointer, "%d", &requests[j].req_fac);
       fscanf( inputfilepointer, "%d", &requests[j].req_start);
       fscanf( inputfilepointer, "%d", &requests[j].req_slot);
       tot_reqs[requests[j].req_cen]+=1;  
    }

    struct Request *d_requests;
    int *d_offsetStart, *d_offsetEnd;
    int *d_currCapacity;
    int *d_capacity;
    int *d_succ_reqs;
    int *d_success;
    int *successCount;

    //Allocate memory on Device
    cudaMalloc(&d_requests, (R)*sizeof(Request));
    cudaMalloc(&d_offsetStart, (N*max_P)*sizeof(int));
    cudaMalloc(&d_offsetEnd, (N*max_P)*sizeof(int));
    cudaMalloc(&d_currCapacity, (N*max_P*25)*sizeof(int));
    cudaMalloc(&d_capacity, max_P*N*sizeof(int));
    cudaMalloc(&d_succ_reqs, N*sizeof(int));
    cudaMalloc(&d_success, sizeof(int));

    //Initializing device variables
    cudaMemset(d_offsetStart, -1, (N*max_P)*sizeof(int));
    cudaMemset(d_offsetEnd, 0, (N*max_P)*sizeof(int));
    cudaMemset(d_currCapacity, 0, (N*max_P*25)*sizeof(int));
    cudaMemset(d_succ_reqs, 0, (N)*sizeof(int));
    cudaMemset(d_success, 0, sizeof(int));

    //Sort the request array, first on the basis of computer center id, if they are same, then on the basis of facility id , if facility id is also same then on the basis of request id
    sort(requests, requests+R, comp);

    cudaMemcpy(d_requests, requests, (R)*sizeof(Request), cudaMemcpyHostToDevice);
    
    //*********************************
    // Call the kernels here
    dim3 grid(ceil(R/1024.0), 1, 1);
    dim3 block(min(R, 1024), 1, 1);

    calculateOffset<<<grid, block>>>(d_requests, d_offsetStart, d_offsetEnd, R);
    cudaDeviceSynchronize();
    cudaMemcpy(d_capacity, capacity, (max_P*N)*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 grid1(ceil((N*max_P)/1024.0), 1, 1);
    dim3 block1(min(N*max_P, 1024), 1, 1);
    processRequests<<<grid1, block1>>>(d_requests, d_offsetStart, d_offsetEnd, d_currCapacity, d_capacity, N*max_P, d_succ_reqs, d_success);
    cudaDeviceSynchronize();

    //********************************
    
    cudaMemcpy(succ_reqs, d_succ_reqs, (N)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(successCount, d_success, 1*sizeof(int), cudaMemcpyDeviceToHost);

    success = successCount[0];
    fail = R-success;

    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}