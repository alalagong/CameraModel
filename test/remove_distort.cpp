#include "abstract_camera.hpp"
#include "MEI_camera.hpp"
#include "pinhole_camera.hpp"
#include <chrono>
#include <string>
#include <algorithm>
#include <opencv2/ccalib/omnidir.hpp>

using namespace camera_model;

// random engine
static std::mt19937_64 rd;
static std::uniform_real_distribution<double> distribution(0.0, std::nextafter(1, std::numeric_limits<double>::max()));

inline int Rand(int min, int max)
{ return (((double)distribution(rd) * (max - min + 1))) + min;}

// record points to Log
void writePointsToLog(const std::vector<cv::Point2f> &input_point, const std::string message)
{
    LOG(INFO) << message << " : " << std::endl;    
    for_each(input_point.begin(), input_point.end(), [](const cv::Point2f& it){ 
        LOG(INFO) << "(" << (it).x << ", "<<(it).y << ")";
    });
}

// By Opencv for Omnidirection camera (MEI)
void undistortOmniByOpencv(cv::Mat& src, std::vector<cv::Point2f> &input_point, 
                        const cv::Mat &K, const cv::Mat &D, const double xi, const int width, const int height)
{
    //! rectify point to undistort
    std::vector<cv::Point2f> pts_undist;
    
    cv::omnidir::undistortPoints(input_point, pts_undist, K, D, xi, cv::Mat());

    std::string message_undist = "Undistorted Point by Opencv";
    writePointsToLog(pts_undist, message_undist);

    
    //! rectify image to undistort
    cv::Mat dst; // (src.size(), src.type());
    cv::Mat map1, map2, K_new = K;

//! 这个参数太不好搞了
    // RECTIFY_PERSPECTIVE 
    K_new.at<double>(0,0) = width;
    K_new.at<double>(0,2) = width/2;
    K_new.at<double>(1,1) = height;
    K_new.at<double>(1,2) = height/2;

    // RECTIFY_CYLINDRICAL
    // K_new.at<double>(0,0) = 3*width/1.6;
    // K_new.at<double>(0,2) = 0;
    // K_new.at<double>(1,1) = 2*height/1.6;
    // K_new.at<double>(1,2) = 0;

    cv::omnidir::initUndistortRectifyMap(K, D, xi, cv::Mat(), K_new, cv::Size(2*width, height), CV_16SC2, map1, map2, cv::omnidir::RECTIFY_CYLINDRICAL);
    cv::remap(src, dst, map1, map2, cv::INTER_LINEAR, BORDER_CONSTANT);
    cv::imshow("UndistByOpencv", dst);
    cv::waitKey(30);

    // RECTIFY_PERSPECTIVE：矫正成透视图像，会损失一些视角 [width/4, height/4, width/2, height/2]
    // RECTIFY_CYLINDRICAL：矫正成圆柱图像，保留所有视图    [width/3.1415, height/3.1415, 0, 0] 下同
    // RECTIFY_STEREOGRAPHIC：矫正成立体图像，会损失一些视角
    // RECTIFY_LONGLATI：矫正成类似世界地图的经纬图，适合立体重建。
}


int main(int argc, char *argv[])
{
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = true;
    FLAGS_log_dir = std::string(getcwd(NULL,0))+"/log/";
    google::InitGoogleLogging(argv[0]);

    LOG_ASSERT(argc == 3) << "\n Usage : ./test calib_file image_file";

    std::string calib_file = argv[1], img_file = argv[2]; // path
    LOG_ASSERT(!calib_file.empty()) << "Empty Calibration file input !!!";
    LOG_ASSERT(!img_file.empty()) << "Empty Image file input !!!";

    std::string model_name;

    AbstractCamera::Ptr camera;

    AbstractCamera::Model model = AbstractCamera::checkCameraModel(calib_file);
    if(AbstractCamera::Model::PINHOLE == model)
    {
        PinholeCamera::Ptr pinhole_camera = PinholeCamera::create(calib_file);
        // 运行时期多态, 虚函数会根据指针来决定运行哪个
        camera = std::static_pointer_cast<AbstractCamera>(pinhole_camera); 
        model_name = "pinhole";
    }
    else if(AbstractCamera::Model::MEI == model)
    {
        MEICamera::Ptr mei_camera = MEICamera::create(calib_file);
        camera = std::static_pointer_cast<AbstractCamera>(mei_camera);
        model_name = "mei";
    }

    cv::Mat src = cv::imread(img_file, CV_8UC1);
    cv::Mat dst;
    std::vector<cv::Point2f> pts_dist, pts_undist;

    // test undistort points with 10 random pixels on image
    for(int i=0; i<10; i++)
    {
       pts_dist.push_back(cv::Point2i(Rand(10, camera->width()-10), Rand(10, camera->height()-10) ) );        
    }
    
    // write log
    std::string message_distpoint = "pixel coordinate generate randomly";
    writePointsToLog(pts_dist, message_distpoint);

    // by camera model
    camera->undistortPoints(pts_dist, pts_undist);
    camera->undistortMat(src, dst);

    std::string message_undistpoint = "Undistorted Point by Mine";
    writePointsToLog(pts_undist, message_undistpoint);

#ifdef _DEBUG_MODE_
    if(AbstractCamera::Model::MEI == model)
    {
        undistortOmniByOpencv(src, pts_dist, camera->K(), camera->D(), camera->xi(), camera->width(), camera->height());
    }
#endif

    size_t found = img_file.find_last_of(".");
    std::string name_front, name_back;
    if(found + 1 != img_file.size())
    {
        name_front = img_file.substr(0, found);
        name_back = img_file.substr(found);
    }


    std::string name_out = name_front + "_undist_"+model_name+name_back;
#ifdef _DEBUG_MODE_    
    std::cout<<"front" <<name_front <<", back" << name_back<<std::endl;;
#endif

    cv::imshow("OriginalImage", src);
    cv::imshow("UndistByMine", dst);
    cv::imwrite(name_out.c_str(),dst); // 输出图片
    cv::waitKey(0);
}
