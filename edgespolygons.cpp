#include "edgespolygons.h"
#include <opencv2/opencv.hpp>
#include "math.h"

using namespace std;
using namespace cv;

vector<Point> EdgesPolygons::approxOuterPolygon(Mat edge)
{
    int maxIndex(0), tempArea(0);
    vector<Vec4i> hierarchy;
    vector<vector<Point>> contour;
    vector<Point> approxPolygon;
    float area;

    //Threshold it as binary for better results
    cv::threshold(edge, edge, 10, 255, CV_THRESH_BINARY);

    //Find and draw the contour woth largest area
    findContours(edge, contour, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    Scalar color(150,15,33);

    for (int i = 0; i >= 0; i = hierarchy[i][0])
    {
        area = contourArea(contour[i]);
        if (tempArea < area)
        {
            tempArea = area;
            maxIndex = i;
        }
    }

    approxPolyDP(contour[maxIndex], approxPolygon, 5, true);
    return approxPolygon;
}

Mat EdgesPolygons::detectEdges(Mat src)
{
    cout << "- Edge detection" << endl;
    Mat blur, edgeX, edgeY, edge, edgeMaxClone, edgeMax, segment;
    vector<Mat> edge_CH(3);
    Size kernel(5,5);

    //Blur the raw image to remove noise
    GaussianBlur(src, blur, kernel, 2);

    //Run sobel edge detector
    Sobel(blur, edgeX, CV_32F, 1, 0);
    Sobel(blur, edgeY, CV_32F, 0, 1);

    //Finding Euclidean distance of edgeX and edgeY
    magnitude(edgeX, edgeY, edge);

    //Convert back to unsigned 3-channel image
    edge.convertTo(edge, CV_8UC3);

    //S[lit each channel into edge_CH[i]
    split(edge, edge_CH);

    //Find the max intensity from 0-255 out of the three channel, dominant edges
    max(edge_CH[0], edge_CH[1], edgeMax);
    max(edge_CH[2], edgeMax, edgeMax);

    return edgeMax;
}
