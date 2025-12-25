
#include <stdint.h>
#include <vector>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <omp.h>
#include "rknn_api.h"
#include <vector>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;
using namespace std::chrono;

struct LightGlueMatches {
    vector<pair<int, int>> pairs;          // (left_idx, right_idx)
    vector<Point2i> kpts0_orig;             // matched points in img0 (original coord)
    vector<Point2i> kpts1_orig;             // matched points in img1
};


class Matches
{
public:
    Matches(const char* filename1, const char* filename2);
    ~Matches();
    unsigned char* load_data(FILE* fp, size_t ofst, size_t sz);
    unsigned char* load_model(const char* filename);
    void dump_tensor_attr(rknn_tensor_attr* attr);
    Mat letterbox_resize(const Mat& image, const Size& size, int bg_color, bool first);
    Mat rgb_to_grayscale(const Mat& image);
    Mat data_process(Mat& image, bool first = true);
    pair<vector<Point2f>, Mat> superpoint_infer(const std::vector<float>& img);
    pair<vector<Point2f>, Mat> superpoint_infer(const cv::Mat& img);
    LightGlueMatches lightglue_infer(const vector<Point2f>& keypoints0, const Mat& desc0, const vector<Point2f>& keypoints1, const Mat& desc1);
    bool load_float_data(const char* filename, std::vector<float>& data, size_t expected_size);

    pair<vector<Point2f>, vector<float>>
postprocessKeypoints(const Mat& scores_onnx, int nms_radius = 4, int top_k = 512);
    Mat simpleNMS(const Mat& scores_batch, int nms_radius);
    Mat maximumFilter(const Mat& input, int radius);
    Mat postprocessDescriptors(const vector<Point2f>& keypoints, const Mat& descriptors_4d, int s = 8);
    LightGlueMatches postprocessLightGlue(
        const Mat& scores,                           // (513, 513), CV_32F
        const vector<Point2f>& kpts0_norm,      // 512 points, normalized [-1,1]
        const vector<Point2f>& kpts1_norm, // for img1
        float threshold = 0.01f);

private:

    double ar1, ar2;
    int ox1, ox2;
    int oy1, oy2;
    bool fixed_input = true;
    int input_size = 512;
    int top_nums = 512;

    rknn_context superpoint_ctx;
    rknn_context lightglue_ctx;
    int model_data_size;
    unsigned char* superpoint_model_data;
    unsigned char* lightglue_model_data;
    rknn_input_output_num superpoint_io_num;    // 输入输出 tensor 个数
    rknn_input_output_num lightglue_io_num;
    rknn_input superpoint_inputs[1];
    rknn_input lightglue_inputs[4]; // 模型的一个数据输入，用来作为参数传入给 rknn_inputs_set 函数

    rknn_tensor_attr superpoint_output_attrs[2];
    rknn_tensor_attr lightglue_output_attrs[1];
};





