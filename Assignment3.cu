/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

__global__ void calculateActiveInDegree(int prevStart, int prevEnd, int* d_aid, int* d_offset, int* d_csrList, int *d_apr)
{
    //prevStart -> First node of previous level
    //prevEnd -> Last node of previous level
    //id -> node which is going to processed by current thread
    int id = blockDim.x * blockIdx.x + threadIdx.x + prevStart;
    if(id>prevEnd || d_aid[id]<d_apr[id])
        return;
    //For all the outnodes of current node, increment the active indegree of that outnode, if current node is active.
    for(int i=d_offset[id];i<d_offset[id+1];i++)
    {
      int child = d_csrList[i];
      atomicInc((unsigned *)&d_aid[child], INT_MAX);
    }
}

__global__ void calculateNumberOfActiveNodesInTheLevel(int firstNode, int lastNode, int level, int *d_activeVertex, int *d_aid, int *d_apr)
{
    //firstNode -> First node of current level
    //lastNode -> Last node of current level
    //id -> node which is going to processed by current thread
    int id = blockDim.x * blockIdx.x + threadIdx.x + firstNode;
    if(id>lastNode || d_aid[id]<d_apr[id])
        return;

    //For rule no 2
    //Rule no 2 won't be applicable for first and last node of any level, that's why not checking for first and last node.
    //For all other nodes of current level, if left neighbour and right neighbour is inactive, then mark it as inactive
    if(id>firstNode && id<lastNode)
        if(d_aid[id-1]<d_apr[id-1] && d_aid[id+1]<d_apr[id+1])
            d_aid[id] = -1;
    
    if(d_aid[id]  >= d_apr[id])
        atomicInc((unsigned *)&d_activeVertex[level], INT_MAX);
}
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();
    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();

    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
	  cudaMalloc(&d_activeVertex, L*sizeof(int));
    cudaMemset(d_activeVertex, 0, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
cudaMemset(d_aid, 0, V*sizeof(int));
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/
int firstNodeInCurrLevel = 0, lastNodeInCurrLevel = 0, i;
for(i=0;i<V;i++)
{
  if(h_apr[i] != 0)
    break;
}
lastNodeInCurrLevel = i-1;
int numberOfNodes = lastNodeInCurrLevel-firstNodeInCurrLevel+1;

//Setting the configuration for kernel launch
dim3 grid(ceil(numberOfNodes/1024.0), 1, 1);
dim3 block(min(numberOfNodes, 1024), 1, 1);

//Checking for rule 2, and then finding number of active nodes int the current level
calculateNumberOfActiveNodesInTheLevel<<<grid, block>>>(firstNodeInCurrLevel, lastNodeInCurrLevel, 0, d_activeVertex, d_aid, d_apr);
cudaDeviceSynchronize();

for(int i=1;i<L;i++)
{

    int prevLevelStart = firstNodeInCurrLevel, prevLevelEnd = lastNodeInCurrLevel;
    for(int j=firstNodeInCurrLevel;j<prevLevelEnd;j++)
        lastNodeInCurrLevel = max(lastNodeInCurrLevel, h_csrList[h_offset[j+1]-1]);
    firstNodeInCurrLevel = prevLevelEnd+1;
    numberOfNodes = prevLevelEnd-prevLevelStart+1;

    dim3 grid1(ceil(numberOfNodes/1024.0), 1, 1);
    dim3 block1(min(numberOfNodes, 1024), 1, 1);

    //Responsible for finding active indegree of nodes of next level.
    calculateActiveInDegree<<<grid1, block1>>>(prevLevelStart, prevLevelEnd, d_aid, d_offset, d_csrList, d_apr);
    cudaDeviceSynchronize(); 

    numberOfNodes = lastNodeInCurrLevel-firstNodeInCurrLevel+1;
    dim3 grid2(ceil(numberOfNodes/1024.0), 1, 1);
    dim3 block2(min(numberOfNodes, 1024), 1, 1);
  
    //Checking for rule 2, and then finding number of active nodes int the current level
    calculateNumberOfActiveNodesInTheLevel<<<grid2, block2>>>(firstNodeInCurrLevel, lastNodeInCurrLevel, i, d_activeVertex, d_aid, d_apr);
    cudaDeviceSynchronize();
}

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
