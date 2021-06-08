#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"
#include "geometry_msgs/Twist.h"
#include <math.h>

// using namespace std;
ros::Publisher command_pub;
bool fire_flag;

void draw_lines(const sensor_msgs::ImageConstPtr &msg);
int avg_index(const cv::Mat img);

int main(int argc, char **argv)
{

    ros::init(argc, argv, "line_detector");
    ros::NodeHandle nh;

    fire_flag = 0;
    
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber img_sub = it.subscribe("/camera/color/image_raw", 1, draw_lines);

    command_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    ros::spin();
    cv::destroyWindow("view");

    return 0;
}

int avg_index(const cv::Mat img)
{

    int rows = img.rows;
    int cols = img.cols;

    int sum = 0;
    int count = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (img.at<uchar>(i, j) > 0)
            {
                sum += j;
                count++;
            }
        }
    }

    if (count <= 0)
    {
        return 0;
    }
    else
    {
        return sum / count;
    }
}

struct POLY_FITTING_COST
{
    POLY_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    template <typename T>
    bool operator()(const T *const abc, T *residual) const
    {
        residual[0] = T(_y) - (abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-(ax^2+bx+c)
        return true;
    }
    const double _x, _y;
};

void draw_lines(const sensor_msgs::ImageConstPtr &msg)
{

    using namespace cv;
    using namespace std;

    Mat src = cv_bridge::toCvShare(msg, "bgr8")->image;
    int rows = src.rows;
    int cols = src.cols;
    // cout << "rows is " << rows;
    // cout << "cols is " << cols;

    // roi
    Mat gray = Mat::zeros(Size(cols, rows), CV_8UC3);
    Mat mask = Mat::zeros(Size(cols, rows), CV_8UC3);
    vector<Point> points;
    Point p1(0, int(0.5 * rows));
    Point p2(cols - 1, int(0.5 * rows));
    Point p3(cols - 1, rows - 1);
    Point p4(0, rows - 1);
    points.push_back(p1);
    points.push_back(p2);
    points.push_back(p3);
    points.push_back(p4);
    vector<vector<Point>> contor;
    contor.push_back(points);
    //
    fillPoly(mask, contor, Scalar(255, 255, 255));
    bitwise_and(src, mask, gray);
    // rgb to gray
    cvtColor(gray, gray, COLOR_BGR2GRAY);
    // gray to black and white
    threshold(gray, gray, 0, 255, THRESH_OTSU);

    points.empty();
    contor.empty();
    p1 = Point(0, 0);
    p2 = Point(cols - 1, 0);
    p3 = Point(cols - 1, int(0.5 * rows));
    p4 = Point(0, int(0.5 * rows));
    points.push_back(p1);
    points.push_back(p2);
    points.push_back(p3);
    points.push_back(p4);
    contor.push_back(points);
    fillPoly(gray, contor, Scalar(255, 255, 255));

    // Mat gray = Mat::zeros(Size(cols, rows), CV_8UC3);
    // cvtColor(src, gray, COLOR_BGR2GRAY);
    // threshold(gray, gray, 0, 255, THRESH_OTSU);

    // black to white
    bitwise_not(gray, gray);

    int window_height = rows / 10;
    int window_width = cols / 4;

    vector<int> s;
    reduce(gray, s, 0, REDUCE_SUM);
    int index = max_element(s.begin(), s.end()) - s.begin();

    Mat img = src.clone();
    int x1 = rows - 1 - window_height;
    int x2 = rows - 1;
    int y1 = int(index - window_width / 2);
    int y2 = int(y1 + window_width);
    y1 = y1 > 0 ? y1 : 0;
    y2 = y2 < cols ? y2 : cols - 1;

    Point pt1(y1, x1);
    Point pt2(y2, x2);
    rectangle(img, pt1, pt2, Scalar(0, 255, 0));
    Mat sub_mat = Mat(gray, Rect(y1, x1, abs(y2 - y1), abs(x2 - x1)));

    vector<Point> point_lists;

    // sliding window
    for (int i = 0; i < 5; i++)
    {
        Point pt1(y1, x1);
        Point pt2(y2, x2);
        rectangle(img, pt1, pt2, Scalar(0, 255, 0));
        Mat sub_mat = Mat(gray, Rect(y1, x1, abs(y2 - y1), abs(x2 - x1)));

        index = avg_index(sub_mat);
        index = y1 + index;

        Point p(index, (x1 + x2) / 2);
        point_lists.push_back(p);
        // circle(img, p, 1, Scalar(0, 0, 255), 4);

        x1 = rows - 1 - window_height * (i + 2);
        x2 = rows - 1 - window_height * (i + 1);
        y1 = int(index - window_width / 2);
        y2 = int(y1 + window_width);
        y1 = y1 > 0 ? y1 : 0;
        y2 = y2 < cols ? y2 : cols - 1;
    }
    double abc[3] = {0, 0, 0};
    ceres::Problem problem;
    for (int i = 0; i < (int)point_lists.size(); i++)
    {
        auto fun = new ceres::AutoDiffCostFunction<POLY_FITTING_COST, 1, 3>(new POLY_FITTING_COST(point_lists[i].y, point_lists[i].x));
        problem.AddResidualBlock(fun, nullptr, abc);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    vector<int> x;
    vector<int> y;
    for (int i = 0; i < rows; i++)
    {
        x.push_back(i);
        y.push_back(int(abc[0] * i * i + abc[1] * i + abc[2]));
    }

    for (int i = 0; i < rows - 1; i++)
    {
        line(img, Point(y[i], x[i]), Point(y[i + 1], x[i + 1]), Scalar(255, 0, 0), 5);
    }
    double x_sum = 0.0;
    double y_sum = 0.0;
    for (int i = 0; i < (int)point_lists.size(); i++)
    {
        circle(img, point_lists[i], 1, Scalar(0, 0, 255), 4);
        x_sum += point_lists[i].x;
        y_sum += point_lists[i].y;
    }
    geometry_msgs::Twist cmd;
    // send vel cmd
    cmd.linear.x = 0.1;

    double x_ave = x_sum / 5;
    double y_ave = y_sum / 5;
    double kp = 1;

    cmd.angular.z = -kp * atan((x_ave - cols / 2.0) / (rows - y_ave));
    cout << cmd.angular.z << endl;
    command_pub.publish(cmd);

    imshow("view", img);
    waitKey(30);
}
