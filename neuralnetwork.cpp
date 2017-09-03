/*
SRI RAMA JEYAM
LaudroBot
NeuralNetwork interface class
20/03/26
V Vinokanth
vinokanth92@gmail.com | vinokanth.velu@gmail.com
*/

#include "neuralnetwork.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv2/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <QVector>
#include <QStringList>
#include <QString>
#include <QStringList>
#include <QDebug>
#include <stdlib.h>
#include <QDir>
#include <QFileInfoList>
#include <QFile>
#include <QTextStream>
#include <QElapsedTimer>
#include <QVariant>
#include <QPair>

using namespace std;
using namespace cv;

NeuralNetwork::NeuralNetwork(QString name, int networkOutputs)
    //Memeber initializer list
    :NETWORK_OUTPUTS(networkOutputs), HIDDEN_LAYERS(5), NAME(name), CLUSTERS(1000),
     savePath("/Users/Vino/Documents/My Stuff/My Robotics Projects/LaundroBotQt/"),
     ANN_savePath(NeuralNetwork::savePath + NeuralNetwork::NAME + "_ANN.yaml"),
     vocabulary_savePath(NeuralNetwork::savePath + NeuralNetwork::NAME + "_vocabulary.yaml"),
     FLANN_savePath(NeuralNetwork::savePath + NeuralNetwork::NAME + "_FLANN.yaml") ,
     className_savePath(NeuralNetwork::savePath + NeuralNetwork::NAME + "_class_names.txt")
{
    qDebug() << "Neural network: " + name.toUpper() + " is created";

    //Reserve the size of NeuralNetwork::trLabels to the number of classes of the training data
    //This means the number of the classes identified by the neural network is the number of outputs
    NeuralNetwork::trClassNames.reserve(networkOutputs);

    //Create a new ANN
    NeuralNetwork::ANN = ml::ANN_MLP::create();
}

void NeuralNetwork::setTrDataPath(QString path)
{
    //Setting path provided during object instantiazation to private variable trDataPath
    if(path.isEmpty() == false)
    {
        NeuralNetwork::trDataPath = path;
    }
    else
    {
        qDebug() << "Invalid data path";
    }
}

void NeuralNetwork::test(Mat src)
{
    //This member function test the given image for any know class and prints the out put on the console
    //We needa descriptor matcher algorithm to match the descriptors returned by the KAZE algorithm to
    //match with the already predifined feature bins/clusters of KMeans

    //For test image we use FLANN based matcher to extract the features defined by the
    //NeuralNetwork::vocabObj (KMeans)

   //Load the existing FLANN object
   //NeuralNetwork::FLANN.load<FlannBasedMatcher>(NeuralNetwork::FLANN_savePath.toStdString());

   Mat testDescriptors = NeuralNetwork::getDescriptors(src); //Get the descriptors for the test image
   std::vector<DMatch> testMatches; //vector to store FLANN output
   Mat outputArray; //ANN inputs

   NeuralNetwork::FLANN.match(testDescriptors,testMatches,cv::noArray());

   //Loop through the matches and get the index of the test decriptor corresponding to the training set of descriptors
   for(auto i = 0; i < testMatches.size(); i++)
   {
       int index = testMatches[i].trainIdx;
       outputArray.at<float>(index)++; //Increment the value at index by 1
   }

   //Normalize outputArray
   cv::normalize(outputArray,outputArray,0,1,cv::NORM_MINMAX,-1,cv::Mat());

   Mat predicition; //Neural network prediciton
   NeuralNetwork::ANN->predict(outputArray,predicition,0);

   qDebug() << "Predicted class: " << NeuralNetwork::getPredictedClass(predicition);

}

QPair<Mat,QVector<NeuralNetwork::imageData>> NeuralNetwork::readData()
{
    qDebug() << "Func: readData()";

    //This function reads the following data from the train data set
    // - Image
    // - File name
    // - Class name
    // - Index (position of image in the dataset vector)

    int totalDescriptors = 0; //total count of descriptors

    //This is the vector of imageData structs
    QVector<NeuralNetwork::imageData> imageDataVector;
    Mat descriptorSet; //A Mat data structure which holds all descriptor data

    QFileInfoList fileNamesList;
    fileNamesList = NeuralNetwork::getFileNames();
    NeuralNetwork::setClassLabels(fileNamesList);
    NeuralNetwork::setClassNames();

    for(int i = 0; i < fileNamesList.size(); i++)
    {
        //Prepare input image path variable
        QString fileName = fileNamesList.at(i).fileName();
        QString path = NeuralNetwork::trDataPath;
        path.append(fileName); //Absolute path of input image

        Mat input = cv::imread(path.toStdString(),0);
        NeuralNetwork::imageData imgStruct;

        //Console output for debug
        qDebug() << "Reading file: " << fileName << " #" << i;

        //While the read image is not empty
        if(input.empty() == false)
        {
            //Compute the descriptors and append them to the Mat

            //This folowing code is for RAM optimisation
            int desSetSizeIntital = descriptorSet.rows;
            descriptorSet.push_back(NeuralNetwork::getDescriptors(input));
            int desSetSizeAfter = descriptorSet.rows;
            imgStruct.descriptorRowStart = totalDescriptors;
            imgStruct.descriptorRowEnd = totalDescriptors + (desSetSizeAfter-desSetSizeIntital); //End of index is (start + size)
            //Add the size of this descriptor mat to the totalDescriptors to mark the position
            totalDescriptors += desSetSizeAfter - desSetSizeIntital;

            input.release();
        }

        //Append the imageStruct obj
        imageDataVector.append(imgStruct);
    }

    //Prepare return QPair object
    QPair<Mat,QVector<NeuralNetwork::imageData>> pair;
    pair.first = descriptorSet;
    pair.second = imageDataVector;

    qDebug() << "EXIT Func: readData()";
    return pair;
}

QFileInfoList NeuralNetwork::getFileNames()
{
    qDebug() << "Func: getFileNames()";

    //Create QDir object and return all filenames
    QDir sourcePath(NeuralNetwork::trDataPath);
    QFileInfoList fileInfoList = sourcePath.entryInfoList(QDir::Files,QDir::Name);

    qDebug() << "EXIT Func: getFileNames()";
    return fileInfoList;
}

Mat NeuralNetwork::getDescriptors(Mat src)
{
    //This function extracts the descriptors or points of interest in the train image set
    //to create bag-of-words
    //This function uses KAZE algorithm to extract local features
    //This is an overloaded function

    //Function parameters
    std::vector<cv::KeyPoint> keypoints; //This stores the detected keypoints of the image
    Mat outputDescriptors;  //Mat array which stores the descriptors

    cv::Ptr<cv::KAZE> Kaze = cv::KAZE::create();
    Kaze->detectAndCompute(src,cv::noArray(),keypoints,outputDescriptors,false);
    return outputDescriptors;
}


void NeuralNetwork::setClassLabels(QFileInfoList fileNameList)
{
    qDebug() << "Func: getClassLabels()";

    //This function return the class name list
    //This list contains the QString::split fileName; fileNameSplit[0] is the class name
    //because the image name format is "class_name.image_number.jpg"

    QStringList fileNameSplit;

    for(int i = 0; i < fileNameList.size(); i++)
    {
        //Extract the fileName from QFileInfoList fileNames and split the QString
        //to extract the class name
        fileNameSplit = fileNameList.at(i).fileName().split(".",QString::SkipEmptyParts);
        NeuralNetwork::trClassList.append(fileNameSplit.at(0)); //Extract the first element
    }

    qDebug() << "EXIT Func: getClassLabels()";
}

void NeuralNetwork::setClassNames()
{
    qDebug() << "Func: setClassNames()";

    //This function creates a vector of class names for the given neural network
    //Loop through NeuralNetwork::trClassList and extract unique IDs
    QString temp = NeuralNetwork::trClassList.at(0); //Extract the first element and store in a temp variable
    NeuralNetwork::trClassNames.append(temp.toUpper()); //Append the first ID

    for(int i = 1; i < NeuralNetwork::trClassList.size(); i++)
    {
        QString currentItem = NeuralNetwork::trClassList.at(i);

        //If comparison is a NOT an exact match, append ID and replace temp
        if(temp.compare(currentItem,Qt::CaseInsensitive) != 0)
        {
            NeuralNetwork::trClassNames.append(currentItem.toUpper());
            temp = currentItem;
        }
    }
}

QPair<Mat, QVector<NeuralNetwork::imageData>> NeuralNetwork::kMeans()
{
    //This function clusters the input descriptors into a given number of clusters
    //The output of this function is struct vocabulary, defined in the class definition of NeuralNetwork
    //struct vocab contains 2 memebers, namely labels and centroids of the KMeans

    qDebug() << "Func: KMeans()";

    //Run readData
    QPair<Mat, QVector<NeuralNetwork::imageData>> pairInput = NeuralNetwork::readData();

    //Parameters for KMeans
    //samples -> contains each data point in a row. The number of coloumns is equal to dimension of the
    //data point; samples are requied to be in type CV-32F as required by the algorithm.
    //labels -> Contains the centroid-index for each data point. Same dim as samples.
    //centres -> Coordinates of centroids

    //Preparing KMeans parameters
    Mat labels;
    Mat centres;

    kmeans(pairInput.first,NeuralNetwork::CLUSTERS,labels,
           TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),1,KMEANS_PP_CENTERS,centres);

    //Prepare vocabulary struct object
    NeuralNetwork::vocabObj.centroids = centres;
    NeuralNetwork::vocabObj.labels = labels;

    //This function returns both the readData vector and the KMeans labels output to build the histogram of
    //images for the neural network input
    QPair <Mat, QVector<NeuralNetwork::imageData> >pairOutput;
    pairOutput.first = labels;
    pairOutput.second = pairInput.second; //readData() ouput pair second element -> NeuralNetwork::imageData

    qDebug() << "EXIT Func: KMeans()";
    return pairOutput;
}

void NeuralNetwork::saveModel()
{
    //Save all models at a predefined location
    NeuralNetwork::ANN->save(NeuralNetwork::ANN_savePath.toStdString()); //Save ANN
    NeuralNetwork::FLANN.save(NeuralNetwork::FLANN_savePath.toStdString()); //Save FLANN

    cv::FileStorage fileStorage(NeuralNetwork::vocabulary_savePath.toStdString(),cv::FileStorage::WRITE);
    fileStorage << "vocabulary" << NeuralNetwork::vocabObj.centroids;
    fileStorage.release();

    //Create text file of class names
    QFile classNamesFile(NeuralNetwork::className_savePath);

    if(classNamesFile.open(QIODevice::ReadWrite))
    {
        QTextStream outputText(&classNamesFile);
        for(int i = 0; i < NeuralNetwork::trClassNames.size(); i++)
        {
            outputText << NeuralNetwork::trClassNames.at(i) << "\n";
        }
    }
    classNamesFile.close();
}

Mat NeuralNetwork::getNumericLabel()
{
    qDebug() << "Func: getNumericLabel()";

    //This function returns numeric labels for Neural Networks
    //For example, if the NN has 3 outputs (classes),
    //the outputs (coloumn vectors) will be defined as follows;
    //Class 1 -> 100
    //Class 2 -> 010
    //Class 3 -> 001

    //Number of bits -> number of outputs
    //Single bit change generate numeric label

    Mat numericLabels;
    qDebug() << "Training data classes: " << this->trClassNames;

    for(int i = 0; i < NeuralNetwork::trClassList.size(); i++)
    {
        //Iniitalize a zero row vector
        Mat tempZero = Mat::zeros(1,NeuralNetwork::NETWORK_OUTPUTS,CV_32F);
        QString currentItem = NeuralNetwork::trClassList.at(i);

        //For each elemnt of list, check the index value and set that element bit of numericLabels to 1
        for(int x = 0; x < NeuralNetwork::trClassNames.size(); x++)
        {
            //if exact match, get the index
            if(currentItem.compare(NeuralNetwork::trClassNames.at(x),Qt::CaseInsensitive) == 0)
            {
                tempZero.at<float>(0,x) = 1; //Set the corresponding bit to 1
            }
        }

        //Append tempZero numericLabels
        numericLabels.push_back(tempZero);
    }

    qDebug() << "EXIT Func: getNumericLabels()";
    return numericLabels;
}

QString NeuralNetwork::getPredictedClass(Mat prediction)
{
    //Find the index of the max element of prdeiction mat
    int current(0), temp(0), index(0);

    for(int i = 0; i < prediction.rows; i++)
    {
        current = prediction.at<int>(i,0);
        if(temp < current)
        {
            temp = current;
            index = i;
        }
    }

    return NeuralNetwork::trClassNames.at(index);
}

Mat NeuralNetwork::getTrainingData()
{
    qDebug() << "Func: getTrainingData()";
    //This function returns the histogram data input for the neural network

    //Run KMeans algorithm
    QPair<Mat, QVector<NeuralNetwork::imageData>> pair = NeuralNetwork::kMeans();
    Mat labels = pair.first; //Labels output of KMeans
    QVector<NeuralNetwork::imageData>readData = pair.second; //Vector imageData struct

    //This Mat is the histograms for the all images based on the KAZE-descriptors and KMeans-clusters
    //This will be actual input to the Neural Network (normalized)
    Mat histogramSet; //Function output

    for(int i = 0; i < readData.size(); i++)
    {
        int startIndex = readData.at(i).descriptorRowStart; //Start of image-descriptor index in labels
        int endIndex = readData.at(i).descriptorRowEnd; //End of image-descriptor index in labels

        //qDebug() << "Start Index: " << startIndex << " End index: " << endIndex;

        //Define the size of the histogram for a given image
        //Histogram is a row vector
        //Number of rows = CLUSTERS of the KMeans
        //The format of cv::Size is cv::Size(colomns,rows); Hence, Size(CLUSTERS,1)
        //The input type for cv::ANN is CV_32F
        Mat histogram = Mat::zeros(Size(NeuralNetwork::CLUSTERS,1),CV_32F);

        for(int x = startIndex; x < endIndex; x++)
        {
            int clusterIndex = labels.at<int>(x,0);
            histogram.at<int>(0,clusterIndex)++; //Increment that index position by 1
        }

        histogramSet.push_back(histogram); //Append the histogram to the Mat
    }

    //Safe to clearup RAM-intensive data strutures
    readData.clear();

    //Normalize the data
    normalize(histogramSet,histogramSet,0,1,cv::NORM_MINMAX,-1,cv::Mat());

    qDebug() << "EXIT Func: getTrainingData()";
    return histogramSet;
}

bool NeuralNetwork::isTrained()
{
    //Checks if the ANN is trained
    return this->ANN->isTrained();
}

bool NeuralNetwork::isFound(Query query)
{
    //This is an utility function which searches system for the query item enum
    if (query == Query::NEURAL_NETWORK)
    {
        //Check for saved neural network model
        QDir testDir(NeuralNetwork::ANN_savePath);
        return testDir.exists();
    }
    else if(query == Query::VOCABULARY)
    {
        //Check for saved neural network model
        QDir testDir(NeuralNetwork::vocabulary_savePath);
        return testDir.exists();
    }
    else if(query == Query::MATCHER)
    {
        //Check for saved neural network model
        QDir testDir(NeuralNetwork::FLANN_savePath);
        return testDir.exists();
    }

    else if(query == Query::ALL)
    {
        QDir ANN_Dir(NeuralNetwork::ANN_savePath);
        QDir VOC_Dir(NeuralNetwork::vocabulary_savePath);
        QDir MATCH_Dir(NeuralNetwork::FLANN_savePath);
        return (ANN_Dir.exists() && VOC_Dir.exists() && MATCH_Dir.exists());
    }
    else
    {
        return false;
    }
}

void NeuralNetwork::train()
{
    qDebug() << "Training neural network: " << this->NAME.toUpper();
    QElapsedTimer timer;
    timer.start(); //Start timer

    //Get normalized training data to feed the neural network
    Mat trainingData = NeuralNetwork::getTrainingData();

    //Set parameters of ANN
    //Pass the labels memeber of vocabObj struct obj to NeuralNetwork::genNumericLabel() to obtain
    //labeled numeric response
    Mat trainingDataResponse = NeuralNetwork::getNumericLabel();

    //cout << "NumericLabels: " << trainingDataResponse.rows << endl;

    //Prepare ANN layer size
    vector<int> layerSizes = {NeuralNetwork::CLUSTERS,NeuralNetwork::HIDDEN_LAYERS,NeuralNetwork::NETWORK_OUTPUTS};
    NeuralNetwork::ANN->setLayerSizes(layerSizes);
    NeuralNetwork::ANN->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM); //Activation function is SIGMOID
    NeuralNetwork::ANN->train(trainingData,ml::ROW_SAMPLE,trainingDataResponse);

    qDebug() << "Completed training neural network: " << this->NAME.toUpper();
    qDebug() << "Training FLANN Matcher";

    //Train the FLANN matcher
    NeuralNetwork::FLANN.add(NeuralNetwork::vocabObj.centroids); //Add the descriptor bins of KMeans
    NeuralNetwork::FLANN.train();

    //Save the KMeans vocabulary and ANN for later usage
    qDebug() << "Saving models at: " << this->savePath;
    NeuralNetwork::saveModel();

    qDebug() << "Total time elapsed: " << ((timer.elapsed()/1000)/60) << " minutes";
}

bool NeuralNetwork::load()
{
    //Load neural network models
    if(NeuralNetwork::isFound(NeuralNetwork::Query::ALL))
    {
        NeuralNetwork::ANN = Algorithm::load<ml::ANN_MLP>(this->ANN_savePath.toStdString(),"ANN");
        //NeuralNetwork::FLANN.load<cv::FlannBasedMatcher>(this->FLANN_savePath.toStdString(),"FLANN");

        //Load class names in to NeuralNetwork::trClassNames
        QFile classNames(this->className_savePath);
        if(classNames.open(QIODevice::ReadWrite))
        {
            QTextStream in(&classNames);

            while(!in.atEnd())
            {
                //Read line and append to trClassNames
                NeuralNetwork::trClassNames.append(in.readLine());
            }

            classNames.close(); //Close file
        }

        qDebug() << "Successfully loaded neural network models";
        return true;
    }
    else
    {
        qDebug() << "Failed to load neural network models. Check status using NeuralNetwork::isFound()";
        return false;
    }
}



