#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>


#include "platform.h"
#include "xil_printf.h"
#include "intimage.h"

#define imax 416
#define jmax 416
#define twozeroeight 208
#define onezerofour 104
#define fivetwo 52
#define twosix 26
#define onethree 13

int yolo_max_boxes=100;
int yolo_iou_threshold = 0.5;
int yolo_score_threshold = 0.5;
double yolo_anchors[2][9]={{10/416,13/416},{16/416,30/416},{33/416,23/416},{30/416,61/416},{62/416,45/416},
		{59/416,119/416},{116/416,90/416},{156/416,198/416},{373/416,326/416}};
int anchor_masks [3][3] = {{6,7,8},{3,4,5},{0,1,2}};
//when you zeropad an input image to make the output image have the same size, you do half padding
int strides=1;
bool batch_norm=true;
static int outputimage1D[174724];//the largest an image can be at the beginning with 1 zero-padding.
static int outputimage2D[418][418];


//416/13=32

float* boundingBox(int *img){
	int gridLengthX = 0;
	int gridLengthY = 32;
	int gridCoordinates[33][33];

	for(int row=33;row>0;row--){
		for (int col=0;col<33;col++){
			gridCoordinates[row][col]=iota(gridLegnthX)+iota(gridLengthY);
		}
	}

}



//Joseph's upsampling.. for 1d..? and what is c..?
float* upsample_cpu(float *in, int w, int h, int c, int batch, int stride, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    out[out_index] = scale*in[in_index];
                }
            }
        }
    }
    return out;
}
//taking length for the length of each 2d layer,
//m for a number of 2D layers,
//layer 3D for a 3D layer of interest
//N to indicate how big upsampled image you want to be. It makes layer3D bigger by N times
int* upsampling_for_3D(unsigned int length, unsigned int m, unsigned int *layer3D[m][length][length],unsigned int N){
	unsigned int upsampled[m][N*length][N*length];
	for (int th=0; th<m;th++){
		for (int row =0; row<N*(length);row++){
			for (int col = 0; col<N*(length);col++){

				upsampled[m][N*row][N*col]=layer3D[m][row][col]; //take one pixel value and put distribute it on every n th cell
				int p = layer3D[m][row][col];
				for(int smallRow = 0; smallRow<N;smallRow++){
					for(int smallCol=0; smallCol<N;smallCol++){
						upsampled[m][N*row+smallRow][N*col+smallCol]=p;
					}
				}


			}

		}
	}
	return upsampled;
}


int* upsampling_for_2D(unsigned int length, unsigned int *layer2D[length][length],unsigned int N){//N for how much you want to resize it
	unsigned int upsampled[N*length][N*length];
	for (int row =0; row<(length);row++){
		for (int col = 0; col<(length);col++){
			upsampled[N*row][N*col]=layer2D[row][col]; //take one pixel value and put distribute it on every n th cell
			int p = layer2D[row][col];
			for(int smallRow = 0; smallRow<N;smallRow++){
				for(int smallCol=0; smallCol<N;smallCol++){
					upsampled[N*row+smallRow][N*col+smallCol]=p;
				}

			}
		}

	}
	return upsampled;
}


//Softmax function has exponential function in it. This exponential function can cause an overflow when a big value is given
//To prevent overflow, we subtract the largest value from every value of a given input.
float* softmax(float *input, int n, float temp, int stride, float *output)
//what is temp for? to scale it down and then to prevent overflow?
//and is stride necessary?
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride]; //get the largest one out of the array
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);//redistribute them and put them into output array
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
    return output;
}

int* batchNorm_for_2D (unsigned int length, unsigned *layer2D[length][length], float elipson, float gamma, float beta){
	int m=length*length;

	unsigned int xValues[m];
	int index=0;
	int sum=0;
	for (int row =0; row<length;row++){
			for (int col = 0; col<length;col++){
				xValues[index++] = layer2D[row][col];
				sum += layer2D[row][col];
			}
	}
	float miniBatchMean= sum/(m);

	sum=0;
	for (int i = 0; i<m;i++){
		xValues[i]=xValues[i]-miniBatchMean;
		sum+=xValues[i];
	}

	float miniBatchVariance= sum/m;

	for (int i =0; i<m;i++){
		xValues[i]=xValues[i]/pow(miniBatchVariance+elipson,0.5);//normalize
	}

	unsigned int yValues[m];
	for (int i =0;i<m;i++){
		yValues[i]=gamma*xValues[i]+beta;//Scale and Shift, gamma and beta are learned from training....?
	}
	return yValues;
}



int* padding_for_1D_chagingto2D(unsigned int* img, int inputLength, int padding){
	int f = 0;
	//suppose
	//output image legnth = 5
	// row = 1,2,3 traveling each row
	int outputImageLength2D = inputLength+(2*padding);
	for (int row = 1; row <= inputLength;row++) {
		for (int col = 1; col <= inputLength; col++){
		//col = 1,2,3, traveling each column
		outputimage1D[row*outputImageLength2D + col] = img[f];
		f++;//iterating through the 1D array version of an 3 by 3 image
		}
	}
	int index = 0;
	for (int col = 0; col<outputImageLength2D;col++){
		for (int row = 0; row<outputImageLength2D;row++){
			outputimage2D[row][col] = outputimage1D[index++];
		}
	}
	return outputimage2D;
}

int* concatenate3d(int length, int numberOfLayers, unsigned int *layer3D[numberOfLayers][length][length], unsigned int *oldLayer3D[numberOfLayers][length][length][length] ){

	unsigned int concatenated[2*numberOfLayers][length][length];
	for (int l =0 ; l<(2*numberOfLayers); l++ ){
		for (int col = 0; col<length; col++){
			for (int row = 0; row<length;row++){
				concatenated[l][row][col];
			}
		}
	}
	return concatenated;
}

int* addition3d(int length, int numberOfLayers, unsigned int layer3D[numberOfLayers][length][length], unsigned int another3DLayer[numberOfLayers][length][length] ){

	unsigned int added[numberOfLayers][length][length];
	for (int l =0 ; l<numberOfLayers; l++ ){
		for (int col = 0; col<length; col++){
			for (int row = 0; row<length;row++){
				layer3D[l][row][col] = layer3D[l][row][col] + another3DLayer[l][row][col];
			}
		}
	}
	return layer3D;
}
int* leakyRelu(int inputLength, unsigned int inputImg[inputLength][inputLength], int alpha){
	for (int row=0;row<inputLength;row++){
			for (int col =0; col<inputLength;col++){
				if (inputImg[row][col]<0){
					inputImg[row][col]=alpha*inputImg[row][col];//alpha = 0.1
				}
			}
		}
		return inputImg;
}

int* padding_for_2D(unsigned int* inputImage2D[imax][jmax], int inputLength, int padding ){

	int outputImage2DLength = inputLength+(2*padding);
	int outputImage2D[outputImage2DLength][outputImage2DLength];
	for (int row = 0; row<inputLength;row++){
		for (int col = 0; col<inputLength;col++){
			outputImage2D[row+padding][col+padding]=inputImage2D[row][col];

			}
		}
	return outputImage2D;

}

int* addResidual2D(int inputLength, unsigned int prevLayer[inputLength][inputLength], unsigned int afterLayer[inputLength][inputLength]){
	for (int row=0;row<inputLength;row++){
		for (int col =0; col<inputLength;col++){
			afterLayer[row][col]=afterLayer[row][col]+prevLayer[row][col];
		}

	}
	return afterLayer;
}

int* averagePooling_for_2D(int kernelLength, int inputLength, unsigned int img[inputLength][inputLength], int padding, int stride){

	int outputsize = (int)floor(((inputLength+2*padding-kernelLength)/stride)+1);//round down
	int kernel[kernelLength][kernelLength];
	int out[outputsize][outputsize];

	for(int i=0; i < inputLength; ++i)              // sliding for image rows
	{
	    for(int j=0; j < inputLength; ++j)          // sliding for image columns
	    {
	        for(int r=0; r < kernelLength; ++r)     // sliding for kernel filter rows
	        {
	            //starting from the right side and going to the left
	            for(int c=0; c < kernelLength; ++c) // kernel columns
	            {
	                    out[i][j] = out[i][j]+(img[i][j] * kernel[r][c]);//add all the numbers up and put it on one grid.
	            }
	            out[i][j]=out[i][j]/(kernelLength*kernelLength);
	        }
	    j=j+stride;
	    }
	    i=i+stride;
	}
	return out;

}
int* averagePooling_for_3D(int inputLength,int M,int R,int C,int kernelLength, float I[M][inputLength][inputLength],float W[M][inputLength][kernelLength][kernelLength], float B[M],int stride, int padding){//based on Professor Milder's Code
	if ((inputLength<=0) || (M<=0) || (R<=0) || (C<=0) || (stride<=0) || (kernelLength<=0)) { // if any of the integers are negative
	        printf("ERROR: 0 or negative parameter\n");
	        return(1);
	}
	int outputsize = (int)floor(((inputLength+2*padding-kernelLength)/stride)+1);//round down

	float O[M][outputsize][outputsize]; //Outputs




    //Actual Convolution : sliding 3D inputs through 4D weights
    for (int m=0; m<M; m++) { // 4D array is a 1D of 3Ds


          for (int n=0; n<inputLength; n++) {
              for (int i=0; i<kernelLength; i++) {
                  for (int j=0; j<kernelLength; j++) {

                      for (int rr=0; rr<R; rr++) {// 2D convolution from here
                          for (int cc=0; cc<C; cc++) {
                              //S would be stride.
                              float t1 = W[m] [n][i][j] * I[n][stride*rr+i][stride*cc+j];
                              // mux: if i==0, j==0, and n==0 we need to add bias.
                              // otherwise, we accumulate
                              float t2 = (i==0 && j==0 && n==0) ? B[m] : O[m][rr][cc];
                              O[m][rr][cc] = (t1 + t2)/(kernelLength*kernelLength);
                          }
                      }
                  }
              }
          }
      }
    return O;
}
int* convolution2D(int kernelLength, int inputLength, unsigned int img[inputLength][inputLength], int padding, int stride){

	// Determining the size of output:
	// rounddown (((n + 2p - f) / s) + 1) by rounddown (((n + 2p - f) / s) + 1)
	// where
	// n is for n by n image
	// f is for f by f filter
	// p is for padding
	// s is for stride
	int outputsize = (int)floor(((inputLength+2*padding-kernelLength)/stride)+1);//round down
	int kernel[kernelLength][kernelLength];
	int out[outputsize][outputsize];
	for(int i=0; i < inputLength; ++i)              // sliding for image rows
	{
	    for(int j=0; j < inputLength; ++j)          // sliding for image columns
	    {
	        for(int r=0; r < kernelLength; ++r)     // sliding for kernel filter rows
	        {
	            //starting from the right side and going to the left
	            for(int c=0; c < kernelLength; ++c) // kernel columns
	            {
	                    out[i][j] = out[i][j]+(img[i][j] * kernel[r][c]);//add all the numbers up and put it on one grid.
	            }
	        }
	    j=j+stride;
	    }
	    i=i+stride;
	}
	return out;
}

int* convolution3D(int inputLength,int M,int R,int C,int kernelLength, float I[M][inputLength][inputLength],float W[M][inputLength][kernelLength][kernelLength], float B[M],int stride, int padding){//based on Professor Milder's Code
	if ((inputLength<=0) || (M<=0) || (R<=0) || (C<=0) || (stride<=0) || (kernelLength<=0)) { // if any of the integers are negative
	        printf("ERROR: 0 or negative parameter\n");
	        return(1);
	}
	int outputsize = (int)floor(((inputLength+2*padding-kernelLength)/stride)+1);//round down

	float O[M][outputsize][outputsize]; //Outputs




    //Actual Convolution : sliding 3D inputs through 4D weights
    for (int m=0; m<M; m++) { // 4D array is a 1D of 3Ds


          for (int n=0; n<inputLength; n++) {
              for (int i=0; i<kernelLength; i++) {
                  for (int j=0; j<kernelLength; j++) {

                      for (int rr=0; rr<R; rr++) {// 2D convolution from here
                          for (int cc=0; cc<C; cc++) {
                              //S would be stride.
                              float t1 = W[m] [n][i][j] * I[n][stride*rr+i][stride*cc+j];
                              // mux: if i==0, j==0, and n==0 we need to add bias.
                              // otherwise, we accumulate
                              float t2 = (i==0 && j==0 && n==0) ? B[m] : O[m][rr][cc];
                              O[m][rr][cc] = t1 + t2;
                          }
                      }
                  }
              }
          }
      }
    return O;
}

int l2Loss(int w[]){
	int length;
	length = sizeof(w)/sizeof(int);
	double sum=0;
	for (int i =0;i<length;i++){
		sum = sum+w[i]*w[i];
	}
	double w2norm = pow(sum,0.5);
	//0.0005*
	//loss = l2 * reduce_sum(square(x))

	return w2norm;
}
int rgbImg (int img[]){
	int r[416][416];
	int g[416][416];
	int b[416][416];

	for (int row=0; row<416; row++){
		for (int col =0; col<416;col++){
			r[row][col]=img[row+col];
			g[row][col]=img[173056+row+col];
			b[row][col]=img[346112+row+col];
		}
	}

	return r,g,b;
}

int DarknetConv(x, filters, size, strides, batch_norm){

	if (strides==1){
		char padding[] = "same";
	}
	else {
		x = topLeftHalfZeroPadding2D(x);
		char padding[] = "valid";
	}
	//x = conv2D();
	if (batch_norm==true){
		 //x = BatchNormalization()(x)
		//x = LeakyReLU(alpha=0.1)(x)

	} return x;
}

int topLeftHalfZeroPadding2D(int x[]){
	int padded[417][417];

	for (int col = 1; col<=416; col++){
		for (int row = 1; row <=416; row++){
			padded[row][col]=x[row-1];
		}
	}

	return x;
}