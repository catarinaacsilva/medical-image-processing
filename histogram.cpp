#include <iostream>


#include "opencv2/core/core.hpp"

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"


using namespace cv;

using namespace std;


Mat computeHistogram( const Mat &imagem ) {
    // Caracteristicas do histograma

    int numImages = 1;

    const int* channels = { 0 };

    // Dimensao

    int dim = 1;

    // Tamanho

    int histSize = 256;     // De 0 a 255

    // Intervalo de intensidades

    float range[] = { 0, 256 };     // Limite superior e EXCLUSIVO !!

    const float* histRange = { range };

    // Histograma uniforme

    bool uniform = true;

    // Nao ha acumulacao de dados de varias imagens

    bool accumulate = false;

    // Calcular o histograma

    Mat histograma;

    calcHist( &imagem, numImages, channels, Mat(),

              histograma, dim, &histSize, &histRange, uniform, accumulate );

     return histograma;
}


void printHistogramFeatures( const Mat & histograma ) {
    // Propriedades obtidas do histograma da imagem original

    // MIN and MAX values and their indices

    double minValue, maxValue;

    int minIndex[2], maxIndex[2];

    minMaxIdx( histograma, &minValue, &maxValue, minIndex, maxIndex );

    cout << endl;

    cout << "Menor numero de pixels : " << minValue << " - Para o nivel de cinzento : " << minIndex[0] << endl;

    cout << "Maior numero de pixels : " << maxValue << " - Para o nivel de cinzento : " << maxIndex[0] << endl;

    // MEAN and STANDARD DEVIATION

    vector<double> meanValue;

    vector<double> stdDev;

    meanStdDev( histograma, meanValue, stdDev );

    cout << "Valor medio : " << meanValue[0] << endl;

    cout << "Desvio padrao : " << stdDev[0] << endl;

    cout << endl;
}


Mat createHistogramImage( Mat histograma ) {
    // Criar uma imagem para representar o histograma

    int histImageWidth = 512;

    int histImageHeight = 512;

    Scalar backgroundIsWhite( 255 );

    Mat histImage( histImageHeight, histImageWidth, CV_8UC1, backgroundIsWhite );

    // Desenhar o histograma

    // Numero de elementos

    int histSize = histograma.total();

    // A largura de cada barra

    int binWidth = (int) ((float) histImageWidth / histSize + 0.5f);

    // Normalizar os valores para [ 0, histImageHeight ]

    Mat histNormalizado;

    normalize( histograma, histNormalizado, 0, histImageHeight, NORM_MINMAX );

    // Desenhar as barras do histograma normalizado

    // ATENCAO : ccordenada Y !!

    for(int i = 0; i < histSize; ++i )
    {
        rectangle( histImage,
                   Point( i * binWidth, histImageHeight ),
                   Point( ( i + 1 ) * binWidth, histImageHeight - (int) (histNormalizado.at<float>(i))),
                   Scalar( 0 ), -1);
    }

    return histImage;
}


int main( int argc, char** argv )
{
    if( argc != 2)
    {
        cout <<"Falta o nome do ficheiro da imagem !!" << endl;

        return -1;
    }

	Mat imagem;

	imagem = imread( argv[1], IMREAD_UNCHANGED);

	if( ! imagem.data )
	{
	    // Leitura SEM SUCESSO

	    cout << "Ficheiro nao foi aberto ou localizado !!" << endl;

	    return -1;
	}

	if( imagem.channels() > 1 )
	{
	    cout << "A imagem carregada tem mais do que UM CANAL !!" << endl;

	    return -1;
	}

    // Janela

    namedWindow( "Imagem Original", WINDOW_AUTOSIZE );

    // Visualizar

    imshow( "Imagem Original", imagem );

    // HISTOGRAMA

    Mat histograma = computeHistogram( imagem );

    // Propriedades obtidas do histograma da imagem original

    printHistogramFeatures( histograma );

    // Criar uma imagem para representar o histograma

    Mat histImage = createHistogramImage( histograma );

    // Visualizar o histograma

    namedWindow( "Histograma", WINDOW_AUTOSIZE );

    imshow( "Histograma", histImage );


    waitKey( 0 );

	destroyAllWindows();

	return 0;
}
