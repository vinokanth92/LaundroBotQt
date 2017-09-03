//V Vinokanth
//15/03/17
//LaundroBot Project
//Neural Network Class

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <QVector>
#include <QStringList>
#include <QString>
#include <QStringList>
#include <QFileInfoList>

using namespace std;
using namespace cv;

class NeuralNetwork
{

public:

    //Public Enums
    enum Query
    {
        NEURAL_NETWORK,
        VOCABULARY,
        MATCHER,
        ALL
    };

    NeuralNetwork(QString name, int networkOutputs);
    void setTrDataPath(QString path);
    void train();
    bool load();
    void load(QString path);
    void test(Mat src); //Public interface to test Neural Network; returns class names
    void test(QVector<Mat> src); //Overloaded test function for an array of images

    //Utility member functions
    bool isTrained(); //Returns TRUE if specified ANN is trained
    bool isFound(Query query); //Returns TRUE if query is found

private:
    QVector<QString> trClassNames; //Vector of types of classes {class_1,class_2,...,class_n}
    QVector<QString> trClassList;  //Class list of all training data images {image_1_class,image_2_class,...,image_n_class}
    Ptr<ml::ANN_MLP> ANN; //OpenCV ANN object
    cv::FlannBasedMatcher FLANN; //FLANN is the algorithm of choice to match descriptors
    const QString NAME; //name of the Neural Network object to be reffered later
    const int HIDDEN_LAYERS;
    const int NETWORK_OUTPUTS;
    int CLUSTERS; //KMeans clusters
    QString trDataPath;

    //Defining the struct of NeuralNetwork::KMeans() output
    struct vocabulary
    {
        Mat centroids;
        Mat labels;
    } vocabObj;

    //Struct holding the data of the read image
    struct imageData
    {
        int descriptorRowStart; //starting row value of decsriptor in the total descriptor Mat (KMeans::descriptorSet)
        int descriptorRowEnd; //ending row value of decsriptor in the total descriptor Mat (KMeans::descriptorSet)
    };

    //Private member functions

    //Path varaible for models
    const QString savePath;
    const QString ANN_savePath;
    const QString vocabulary_savePath;
    const QString FLANN_savePath;
    const QString className_savePath;

    QPair<Mat,QVector<NeuralNetwork::imageData>> readData(); //Returns descriptor set, and imageData vector
    QFileInfoList getFileNames();
    Mat getDescriptors(Mat src);
    void setClassLabels(QFileInfoList fileNameList);
    void setClassNames(); //Sets the class names to trClassNames
    //Generate K-clusters or BoW for all descriptors for a given set of images
    QPair<Mat, QVector<NeuralNetwork::imageData> > kMeans();
    void saveModel(); //Save KMeans BoW and FLANN based matcher for the given vocabulary
    Mat getNumericLabel(); //Returns numeric labels to train Neural Network
    void genHashTable();
    QString getPredictedClass(Mat prediction); //Returns the class-string of prediction
    Mat getTrainingData(); //Returns the histogram for a given set of descriptors

};

#endif // NEURALNETWORK_H
