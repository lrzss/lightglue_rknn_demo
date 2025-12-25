#include "mathes.h"
#include <cstdlib>

int getSecondDim(const char* path) {
    std::string s(path);
    size_t pos1 = s.find_last_of('_');
    if (pos1 == std::string::npos) return -1;
    size_t pos2 = s.find_last_of('_', pos1 - 1);
    if (pos2 == std::string::npos) return -1;

    std::string num_str = s.substr(pos2 + 1, pos1 - pos2 - 1);
    return std::atoi(num_str.c_str()); // 返回 256
}

inline float fp16_to_fp32(uint16_t h) {
    uint32_t t1 = h & 0x7fff;
    uint32_t t2 = h & 0x8000;
    uint32_t t3 = h & 0x7c00;
    t1 <<= 13;
    t2 <<= 16;
    t1 += 0x38000000;
    t1 = (t3 == 0 ? 0 : t1);
    t1 |= t2;
    float f;
    std::memcpy(&f, &t1, sizeof(f));
    return f;
}

inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint16_t sign = (x >> 16) & 0x8000;
    uint16_t expo = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (expo <= 0) {
        // Subnormal or zero
        mant = (mant | 0x800000) >> (1 - expo);
        return sign | (mant >> 13);
    } else if (expo >= 31) {
        // Infinity or NaN
        return sign | 0x7c00;
    } else {
        // Normal value
        return sign | (expo << 10) | (mant >> 13);
    }
}

void printPoints(const std::string& name, const std::vector<cv::Point2f>& points) {
    std::cout << name << " (" << points.size() << " points):\n";
    for (size_t i = 0; i < std::min(points.size(), size_t(10)); ++i) { // 只打前10个，避免刷屏
        std::cout << "  [" << i << "] (" << points[i].x << ", " << points[i].y << ")\n";
    }
    if (points.size() > 10) {
        std::cout << "  ... (total " << points.size() << " points)\n";
    }
}


void save_intermediate_result(const string& filename, const Mat& mat) {
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "result" << mat;
    fs.release();
}

bool save_float_data(const char* filename, const float* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    return true;
}


bool Matches::load_float_data(const char* filename, std::vector<float>& data, size_t expected_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t expected_bytes = expected_size * sizeof(float);
    if (file_size != expected_bytes) {
        std::cerr << "File size mismatch: " << file_size << " vs " << expected_bytes << std::endl;
        return false;
    }
    data.resize(expected_size);
    file.read(reinterpret_cast<char*>(data.data()), expected_bytes);
    file.close();
    return true;
}

Matches::Matches(const char* filename1, const char* filename2) {
    int ret;
    top_nums = getSecondDim(filename1);
    printf("top_nums = %d\n", top_nums);
    rknn_core_mask core_mask;
    printf("Loading superpoint mode...\n");
    superpoint_model_data = load_model(filename1);  // 加载模型

    ret = rknn_init(&superpoint_ctx, superpoint_model_data, model_data_size, 0, NULL);  // 初始化函数将创建 rknn_context 对象、加载 RKNN 模型以及根据 flag 和 rknn_init_extend 结构体执行特定的初始化行为

    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    core_mask = RKNN_NPU_CORE_0;
    ret = rknn_set_core_mask(superpoint_ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        exit(-1);
    }

    rknn_sdk_version version;  // 用来表示 RKNN SDK 的版本信息
    // 能够查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、SDK 版本、内存占用信息、用户自定义字符串等信息
    ret = rknn_query(superpoint_ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);


    ret = rknn_query(superpoint_ctx, RKNN_QUERY_IN_OUT_NUM, &superpoint_io_num, sizeof(superpoint_io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    printf("superpoint model input num: %d, output num: %d\n", superpoint_io_num.n_input, superpoint_io_num.n_output);

    rknn_tensor_attr superpoint_input_attrs[superpoint_io_num.n_input];  // 表示模型的 tensor 的属性   输入
    memset(superpoint_input_attrs, 0, sizeof(superpoint_input_attrs));   // 初始化为0
    for (int i = 0; i < superpoint_io_num.n_input; i++) {
        superpoint_input_attrs[i].index = i;
        ret = rknn_query(superpoint_ctx, RKNN_QUERY_INPUT_ATTR, &(superpoint_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
        dump_tensor_attr(&(superpoint_input_attrs[i]));  // 输出模型tensor属性（输入）
    }

    //（输出）
    memset(superpoint_output_attrs, 0, sizeof(superpoint_output_attrs));
    for (int i = 0; i < superpoint_io_num.n_output; i++) {
        superpoint_output_attrs[i].index = i;
        ret = rknn_query(superpoint_ctx, RKNN_QUERY_OUTPUT_ATTR, &(superpoint_output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(superpoint_output_attrs[i]));
    }

    // 这里的设置不要动，不然推理结果不对
    memset(superpoint_inputs, 0, sizeof(superpoint_inputs));
    superpoint_inputs[0].index = 0;
    superpoint_inputs[0].type = RKNN_TENSOR_FLOAT32;     // RKNN_TENSOR_FLOAT16 RKNN_TENSOR_UINT8
    superpoint_inputs[0].size = input_size * input_size * 1 * 4;
    superpoint_inputs[0].fmt = RKNN_TENSOR_NHWC;
    superpoint_inputs[0].pass_through = 0;


    printf("Loading lightglue mode...\n");
    lightglue_model_data = load_model(filename2);  // 加载模型

    ret = rknn_init(&lightglue_ctx, lightglue_model_data, model_data_size, 0, NULL);  // 初始化函数将创建 rknn_context 对象、加载 RKNN 模型以及根据 flag 和 rknn_init_extend 结构体执行特定的初始化行为

    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    core_mask = RKNN_NPU_CORE_0_1_2;
    ret = rknn_set_core_mask(lightglue_ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        exit(-1);
    }

    ret = rknn_query(lightglue_ctx, RKNN_QUERY_IN_OUT_NUM, &lightglue_io_num, sizeof(lightglue_io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    printf("lightglue model input num: %d, output num: %d\n", lightglue_io_num.n_input, lightglue_io_num.n_output);

    rknn_tensor_attr lightglue_input_attrs[lightglue_io_num.n_input];  // 表示模型的 tensor 的属性   输入
    memset(lightglue_input_attrs, 0, sizeof(lightglue_input_attrs));   // 初始化为0
    for (int i = 0; i < lightglue_io_num.n_input; i++) {
        lightglue_input_attrs[i].index = i;
        ret = rknn_query(lightglue_ctx, RKNN_QUERY_INPUT_ATTR, &(lightglue_input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
        dump_tensor_attr(&(lightglue_input_attrs[i]));  // 输出模型tensor属性（输入）
    }

    //（输出）
    memset(lightglue_output_attrs, 0, sizeof(lightglue_output_attrs));
    for (int i = 0; i < lightglue_io_num.n_output; i++) {
        lightglue_output_attrs[i].index = i;
        ret = rknn_query(lightglue_ctx, RKNN_QUERY_OUTPUT_ATTR, &(lightglue_output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(lightglue_output_attrs[i]));
    }

    // 这里的设置不要动，不然推理结果不对
    memset(lightglue_inputs, 0, sizeof(lightglue_inputs));
    lightglue_inputs[0].index = 0;
    lightglue_inputs[0].type = RKNN_TENSOR_FLOAT32;     // RKNN_TENSOR_FLOAT16 RKNN_TENSOR_UINT8
    lightglue_inputs[0].size = top_nums * 2 * 4;
    lightglue_inputs[0].fmt = RKNN_TENSOR_NHWC;
    lightglue_inputs[0].pass_through = 0;
    lightglue_inputs[1].index = 1;
    lightglue_inputs[1].type = RKNN_TENSOR_FLOAT32;     // RKNN_TENSOR_FLOAT16 RKNN_TENSOR_UINT8
    lightglue_inputs[1].size = top_nums * 2 * 4;
    lightglue_inputs[1].fmt = RKNN_TENSOR_NHWC;
    lightglue_inputs[1].pass_through = 0;
    lightglue_inputs[2].index = 2;
    lightglue_inputs[2].type = RKNN_TENSOR_FLOAT32;     // RKNN_TENSOR_FLOAT16 RKNN_TENSOR_UINT8
    lightglue_inputs[2].size = top_nums * 256 * 4;
    lightglue_inputs[2].fmt = RKNN_TENSOR_NHWC;
    lightglue_inputs[2].pass_through = 0;
    lightglue_inputs[3].index = 3;
    lightglue_inputs[3].type = RKNN_TENSOR_FLOAT32;     // RKNN_TENSOR_FLOAT16 RKNN_TENSOR_UINT8
    lightglue_inputs[3].size = top_nums * 256 * 4;
    lightglue_inputs[3].fmt = RKNN_TENSOR_NHWC;
    lightglue_inputs[3].pass_through = 0;
}

// Mat Matches::rgb_to_grayscale(const Mat& image) {
//     // 确保输入图像是3通道的彩色图像
//     if(image.channels() != 3) {
//         cerr << "Input image is not RGB!" << endl;
//         return Mat();
//     }
//     // 分离通道
//     vector<Mat> channels;
//     split(image, channels);
//     // 定义灰度转换系数
//     array<double, 3> scale = {0.299, 0.587, 0.114};
//     // 计算加权和以得到灰度图像
//     Mat grayscale_image(image.rows, image.cols, CV_32FC1);
//     grayscale_image = scale[0] * channels[0] + // R
//                       scale[1] * channels[1] + // G
//                       scale[2] * channels[2];  // B
//     return grayscale_image;
// }

Mat Matches::rgb_to_grayscale(const Mat& image) {
    // 确保输入图像是3通道的彩色图像
    if(image.channels() != 3) {
        std::cerr << "Input image is not BGR!" << std::endl;
        return cv::Mat();
    }

    // 分离通道（BGR顺序）
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // BGR顺序：[0]=B, [1]=G, [2]=R
    // 灰度转换系数：Y = 0.299*R + 0.587*G + 0.114*B
    const double r_weight = 0.114;   // R分量权重
    const double g_weight = 0.587;   // G分量权重
    const double b_weight = 0.299;   // B分量权重

    // 创建灰度图像（单通道，浮点类型）
    cv::Mat grayscale_image;

    // 方法1：使用addWeighted（效率较高，推荐）
    cv::Mat temp;
    // 先计算 0.299*R + 0.587*G
    cv::addWeighted(channels[2], r_weight, channels[1], g_weight, 0.0, temp, CV_32F);
    // 再加上 0.114*B
    cv::addWeighted(temp, 1.0, channels[0], b_weight, 0.0, grayscale_image, CV_32F);
    cv::Mat grayscale_3d = grayscale_image.reshape(1, grayscale_image.rows);
    return grayscale_3d;
}

Mat Matches::letterbox_resize(const Mat& image, const Size& size, int bg_color, bool first) {
    // 获取输入图像尺寸
    int h = image.rows;
    int w = image.cols;
    int c = image.channels();

    // 缩放比例
    double ar = std::min(static_cast<double>(size.width) / w, static_cast<double>(size.height) / h);
    if(first){
        ar1 = ar;
    }
    else{
        ar2 = ar;
    }

    int new_w = static_cast<int>(w * ar);
    int new_h = static_cast<int>(h * ar);

    // resize
    Mat resized_double, resized;
    resize(image, resized_double, Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    resized_double.convertTo(resized, CV_8U);

    // 创建背景画布
    Mat canvas(size.height, size.width, CV_8UC1, Scalar(bg_color));

    // 计算偏移
    int ox = (size.width - new_w) / 2;
    int oy = (size.height - new_h) / 2;
    if(first){
        ox1 = ox;
        oy1 = oy;
    }
    else{
        ox2 = ox;
        oy2 = oy;
    }
//     printf("ar=%f ox=%d oy=%d \n", ar, ox, oy);

    // 贴图
    Rect roi(ox, oy, new_w, new_h);
    resized.copyTo(canvas(roi));

    return canvas;
}

Mat Matches::data_process(Mat& image, bool first){
    Mat infer_img;
//     Mat gray = rgb_to_grayscale(image);  // 转为灰度几种方式都可以
    Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
//     save_intermediate_result("cpp_gray.yml", gray);
    gray = letterbox_resize(gray, cv::Size(input_size,input_size), 114, first);
//     save_intermediate_result("cpp_letterbox.yml", gray);
    gray.convertTo(infer_img, CV_32F, 1.0f / 255.0f);
//     save_intermediate_result("cpp_infer_img.yml", infer_img);
//     save_float_data("cpp_infer_img.bin", (const float*)infer_img.data, input_size*input_size);
    return infer_img;
}


unsigned char* Matches::load_data(FILE* fp, size_t ofst, size_t sz)
{
    unsigned char* data;
    int            ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);  // 开辟空间
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

unsigned char* Matches::load_model(const char* filename)
{
    FILE* fp;
    unsigned char* data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);  // 指向文件尾
    int size = ftell(fp);  // 获取文件大小

    data = load_data(fp, 0, size);  // 获取数据

    fclose(fp);

    model_data_size = size;
    return data;
}

void Matches::dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
        "zp=%d, scale=%f\n",
        attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
        attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// ----------------------------
// Fast maximum filter using cv::dilate (equivalent to scipy's maximum_filter with constant=0)
// Input: single-channel CV_32F Mat (H, W)
// Output: same size, local max in (2r+1)x(2r+1) window, padded with 0
// ----------------------------
Mat Matches::maximumFilter(const Mat& input, int radius) {
    int ksize = 2 * radius + 1;
    // Use rectangular structuring element for max filter
    Mat kernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
    Mat output;
    // BORDER_CONSTANT with value 0.0 is default in dilate when anchor=-1 and borderType=cv::BORDER_CONSTANT
    dilate(input, output, kernel, Point(-1, -1), 1, cv::BORDER_CONSTANT, Scalar(0.0));
    return output;
}

// ----------------------------
// Simple Non-Maximum Suppression (as in SuperPoint paper)
// Input: scores_onnx — shape (1, H, W), type CV_32F
// Output: suppressed scores — shape (H, W), type CV_32F
// ----------------------------
Mat Matches::simpleNMS(const Mat& scores_batch, int nms_radius) {
    CV_Assert(scores_batch.type() == CV_32F);
    CV_Assert(scores_batch.dims == 3 && scores_batch.size[0] == 1);

    // Extract the (H, W) plane
    int H = scores_batch.size[1];
    int W = scores_batch.size[2];
    Mat scores(scores_batch.size[1], scores_batch.size[2], CV_32F,
                   (void*)(scores_batch.ptr<float>(0))); // no copy

    Mat zeros = Mat::zeros(H, W, CV_32F);

    // Step 1: initial local maxima
    Mat local_max = maximumFilter(scores, nms_radius);
    Mat max_mask = (scores == local_max); // CV_8U

    // Two refinement iterations (as in original SuperPoint)
    for (int iter = 0; iter < 2; ++iter) {
        // Convert boolean mask to float, then dilate to mark suppression zones
        Mat max_mask_f;
        max_mask.convertTo(max_mask_f, CV_32F);
        Mat supp_mask_f = maximumFilter(max_mask_f, nms_radius);
        Mat supp_mask = (supp_mask_f > 0); // CV_8U

        // Zero out suppressed regions: keep only where NOT suppressed
        Mat supp_scores;
        scores.copyTo(supp_scores, ~supp_mask);

        // Find new local maxima in non-suppressed area
        Mat new_local_max = maximumFilter(supp_scores, nms_radius);
        Mat new_max_mask = (supp_scores == new_local_max) & (~supp_mask);

        // Update: keep old maxima OR add new valid ones
        max_mask = max_mask | new_max_mask;
    }

    // Apply final mask to scores
    Mat result;
    scores.copyTo(result, max_mask);
    return result; // (H, W), CV_32F
}

// ----------------------------
// Main postprocessing function
// Input: scores_onnx — (1, H, W), CV_32F
// Output: {keypoints (x,y), scores}
// ----------------------------
pair<vector<Point2f>, vector<float>> Matches::postprocessKeypoints(const Mat& scores_onnx, int nms_radius, int top_k)
{
    // Validate input
    CV_Assert(scores_onnx.type() == CV_32F);
    CV_Assert(scores_onnx.dims == 3 && scores_onnx.size[0] == 1);

    // Step 1: NMS
    cv::Mat scores_nms = simpleNMS(scores_onnx, nms_radius); // (H, W)

    int H = scores_nms.rows;
    int W = scores_nms.cols;
    int total = H * W;

    if (top_k > total) top_k = total;

    // Step 2: Flatten and get top-k indices by score
    std::vector<std::pair<float, int>> score_idx;
    score_idx.reserve(total);

    const float* ptr = scores_nms.ptr<float>(0);
    for (int i = 0; i < total; ++i) {
        score_idx.emplace_back(ptr[i], i);
    }

    // Partial sort to get top_k largest scores
    std::nth_element(score_idx.begin(), score_idx.begin() + (total - top_k), score_idx.end());
    auto top_begin = score_idx.end() - top_k;
    std::sort(top_begin, score_idx.end(), [](const auto& a, const auto& b) {
        return a.first > b.first; // descending order
    });

    // Step 3 & 4: Convert flat index to (x, y) = (w, h)
    std::vector<cv::Point2f> keypoints;
    std::vector<float> scores;
    keypoints.reserve(top_k);
    scores.reserve(top_k);

    for (auto it = top_begin; it != score_idx.end(); ++it) {
        int flat_idx = it->second;
        int h = flat_idx / W; // row
        int w = flat_idx % W; // col
        keypoints.emplace_back(w, h); // (x=w, y=h)
        scores.push_back(it->first);
    }

    return {keypoints, scores};
}

template<typename T>
inline T clamp(T v, T low, T high) {
    return std::max(low, std::min(v, high));
}

Mat Matches::postprocessDescriptors(const vector<Point2f>& keypoints, const Mat& descriptors_4d, int s)
{
    // Validate input shape: must be 4D with batch=1
    CV_Assert(descriptors_4d.dims == 4);
    CV_Assert(descriptors_4d.size[0] == 1); // batch
    int C = descriptors_4d.size[1];
    int H = descriptors_4d.size[2];
    int W = descriptors_4d.size[3];
    CV_Assert(descriptors_4d.type() == CV_32F);

    int N = static_cast<int>(keypoints.size());
    if (N == 0) {
        return Mat(0, C, CV_32F);
    }
    // Step 1: Transform keypoints: kpts = kpts - s/2 + 0.5
    vector<Point2f> kpts(N);
    for (int i = 0; i < N; ++i) {
        kpts[i].x = keypoints[i].x - s / 2.0f + 0.5f;
        kpts[i].y = keypoints[i].y - s / 2.0f + 0.5f;
    }
    // Normalize by effective grid size (SuperPoint-specific)
    float norm_x = static_cast<float>(W * s - s / 2.0 - 0.5);
    float norm_y = static_cast<float>(H * s - s / 2.0 - 0.5);

    // Avoid division by zero (though shouldn't happen in practice)
    if (norm_x <= 0 || norm_y <= 0) {
        Mat empty;
        empty.create(0, C, CV_32F);
        return empty;
    }
    // Map to [0, W-1] and [0, H-1]
    vector<float> x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x[i] = (kpts[i].x / norm_x) * (W - 1);
        y[i] = (kpts[i].y / norm_y) * (H - 1);
    }

    // Nearest integer coords
    vector<int> x0(N), x1(N), y0(N), y1(N);
    for (int i = 0; i < N; ++i) {
        x0[i] = static_cast<int>(std::floor(x[i]));
        x1[i] = x0[i] + 1;
        y0[i] = static_cast<int>(std::floor(y[i]));
        y1[i] = y0[i] + 1;

        // Clip to valid range [0, W-1] / [0, H-1]
        x0[i] = clamp(x0[i], 0, W - 1);
        x1[i] = clamp(x1[i], 0, W - 1);
        y0[i] = clamp(y0[i], 0, H - 1);
        y1[i] = clamp(y1[i], 0, H - 1);
    }
    // Bilinear weights
    vector<float> wa(N), wb(N), wc(N), wd(N);
    for (int i = 0; i < N; ++i) {
        wa[i] = (x1[i] - x[i]) * (y1[i] - y[i]);
        wb[i] = (x1[i] - x[i]) * (y[i] - y0[i]);
        wc[i] = (x[i] - x0[i]) * (y1[i] - y[i]);
        wd[i] = (x[i] - x0[i]) * (y[i] - y0[i]);
    }
    // Access descriptor data: (1, C, H, W) → we take plane [0]
    // Data layout: [batch][C][H][W] → contiguous: index = ((c * H + h) * W + w)
    const float* desc_data = descriptors_4d.ptr<float>(0); // pointer to first element
    // Output matrix: (N, C)
    Mat descriptors_out(N, C, CV_32F);
    float* out_ptr = descriptors_out.ptr<float>(0);
    // For each keypoint, compute bilinear interpolation across all C channels
    for (int i = 0; i < N; ++i) {
        int idx_a = y0[i] * W + x0[i];
        int idx_b = y1[i] * W + x0[i];
        int idx_c = y0[i] * W + x1[i];
        int idx_d = y1[i] * W + x1[i];
        float w_a = wa[i];
        float w_b = wb[i];
        float w_c = wc[i];
        float w_d = wd[i];

        // For each channel c
        for (int c = 0; c < C; ++c) {
            // Compute base offset for channel c: c * H * W
            int base = c * H * W;
            float val = w_a * desc_data[base + idx_a] +
                        w_b * desc_data[base + idx_b] +
                        w_c * desc_data[base + idx_c] +
                        w_d * desc_data[base + idx_d];
            out_ptr[i * C + c] = val;
        }
    }
    // L2 normalize each row (descriptor)
    for (int i = 0; i < N; ++i) {
        float* row = descriptors_out.ptr<float>(i);
        double norm = 0.0;
        for (int c = 0; c < C; ++c) {
            norm += static_cast<double>(row[c]) * row[c];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            float inv_norm = static_cast<float>(1.0 / norm);
            for (int c = 0; c < C; ++c) {
                row[c] *= inv_norm;
            }
        }
        // else: leave as-is (already zero)
    }
    return descriptors_out; // (N, C), CV_32F
}

LightGlueMatches Matches::postprocessLightGlue(
    const Mat& scores,                           // (top_nums+1, top_nums+1), CV_32F
    const vector<Point2f>& kpts0_norm,      // top_nums points, normalized [-1,1]
    const vector<Point2f>& kpts1_norm, // for img1
    float threshold)
{
    const int M = top_nums; // = 513 - 1
    const int N = top_nums;

    // --- 1. Softmax along axis=1 (each row) ---
    Mat prob = Mat::zeros(top_nums+1, top_nums+1, CV_32F);
    for (int i = 0; i < top_nums+1; ++i) {
        Mat row = scores.row(i); // (1, top_nums+1)
        float max_val = *std::max_element(row.begin<float>(), row.end<float>());
        Mat shifted;
        cv::subtract(row, Scalar(max_val), shifted);
        cv::exp(shifted, shifted);
        float sum = cv::sum(shifted)[0];
        if (sum > 1e-12f) {
            cv::divide(shifted, cv::Scalar(sum), prob.row(i));
        } else {
            prob.row(i) = 0.0f;
        }
    }
    // --- 2. match0: argmax over all top_nums+1 cols (including unmatched) ---
    vector<int> match0(M);
    vector<float> mscores0(M);
    for (int i = 0; i < M; ++i) {
        Mat row = prob.row(i);
        double max_val;
        Point max_loc;
        minMaxLoc(row, nullptr, &max_val, nullptr, &max_loc);
        match0[i] = max_loc.x;           // 0..top_nums
        mscores0[i] = static_cast<float>(max_val);
    }
    // --- 3. match1: argmax over first top_nums cols, then transpose ---
    Mat prob_sub = prob(Rect(0, 0, N, M)); // rows [0,top_nums), cols [0,top_nums)
    Mat prob_sub_T;
    cv::transpose(prob_sub, prob_sub_T); // (top_nums, top_nums)
    vector<int> match1(N);
    vector<float> mscores1(N);
    for (int j = 0; j < N; ++j) {
        Mat col = prob_sub_T.row(j);
        double max_val;
        Point max_loc;
        minMaxLoc(col, nullptr, &max_val, nullptr, &max_loc);
        match1[j] = max_loc.x;           // 0..511
        mscores1[j] = static_cast<float>(max_val);
    }
    // --- 4. Score thresholding ---
    vector<bool> valid0_score(M, false);
    vector<bool> valid1_score(N, false);
    for (int i = 0; i < M; ++i) valid0_score[i] = (mscores0[i] > threshold);
    for (int j = 0; j < N; ++j) valid1_score[j] = (mscores1[j] > threshold);
    // --- 5. Mutual consistency ---
    vector<bool> valid0_mutual(M, false);
    for (int i = 0; i < M; ++i) {
        if (match0[i] < N) { // not unmatched
            int j = match0[i];
            if (match1[j] == i) {
                valid0_mutual[i] = true;
            }
        }
    }
    vector<bool> valid1_mutual(N, false);
    for (int j = 0; j < N; ++j) {
        int i = match1[j];
        if (i >= 0 && i < M && match0[i] == j) {
            valid1_mutual[j] = true;
        }
    }
    // --- 6. Final matches ---
    vector<int> final_match0(M, -1);
    for (int i = 0; i < M; ++i) {
        if (valid0_score[i] && valid0_mutual[i]) {
            final_match0[i] = match0[i];
        }
    }
    // --- 7. Build pairs ---
    vector<pair<int, int>> pairs;
    for (int i = 0; i < M; ++i) {
        if (final_match0[i] >= 0) {
            pairs.emplace_back(i, final_match0[i]);
        }
    }
    // --- 8. Denormalize keypoints to original image coordinates ---
    auto denormalize = [this](const vector<Point2f>& kpts_norm,
                          float scale, int ox, int oy) {
        vector<Point2i> out;
        out.reserve(kpts_norm.size());
        for (const auto& kp : kpts_norm) {
            float x_512 = (kp.x + 1.0f) * (float)this->input_size / 2.0f;
            float y_512 = (kp.y + 1.0f) * (float)this->input_size / 2.0f;
            x_512 -= ox;
            y_512 -= oy;
            int x_orig = static_cast<int>(std::round(x_512 / scale));
            int y_orig = static_cast<int>(std::round(y_512 / scale));
            out.emplace_back(x_orig, y_orig);
        }
        return out;
    };
//     printPoints("kpts0_norm", kpts0_norm);   // 打印归一化后的关键点
//     printPoints("kpts1_norm", kpts1_norm);
    auto kpts0_plot = denormalize(kpts0_norm, ar1, ox1, oy1);   // 反归一化，还原坐标
    auto kpts1_plot = denormalize(kpts1_norm, ar2, ox2, oy2);

    // --- 9. Extract matched points ---
    vector<Point2i> kpts0_match, kpts1_match;
    for (const auto& pair : pairs) {
        int i = pair.first;
        int j = pair.second;
        kpts0_match.push_back(kpts0_plot[i]);
        kpts1_match.push_back(kpts1_plot[j]);
    }
    cout << "Total matches: " << pairs.size() << endl;
    // 打印出前5个点
//     if (!pairs.empty()) {
//         cout << "Example pairs: ";
//         for (int idx = 0; idx < std::min(5, (int)pairs.size()); ++idx) {
//             cout << "(" << pairs[idx].first << "," << pairs[idx].second << ") ";
//         }
//         cout << endl;
//     }
    return {pairs, kpts0_match, kpts1_match};
}

pair<vector<Point2f>, Mat> Matches::superpoint_infer(const std::vector<float>& img) {
    fixed_input = true;
    superpoint_inputs[0].buf = const_cast<void*>(static_cast<const void*>(img.data()));

    rknn_inputs_set(superpoint_ctx, superpoint_io_num.n_input, superpoint_inputs);  // 可以设置模型的输入数据

    rknn_output superpoint_outputs[2];
    memset(superpoint_outputs, 0, sizeof(superpoint_outputs));
    for (int i = 0; i < superpoint_io_num.n_output; i++) {
        superpoint_outputs[i].want_float = 1;
        //superpoint_outputs[i].buf = nullptr;      // 👈 关键！确保 buf 初始为 null
        //superpoint_outputs[i].size = 0;
        superpoint_outputs[i].is_prealloc = 0;
    }

    //cout << "rknn_run start!" << endl;
    int ret = rknn_run(superpoint_ctx, NULL);  // 将执行一次模型推理，调用之前需要先通过 rknn_inputs_set 函数或者零拷贝的接口设置输入数据
    rknn_outputs_get(superpoint_ctx, 2, superpoint_outputs, nullptr);
    // Save superpoint_outputs as float32 将输出保存到文件
//     size_t elem_count[2] = {top_nums * top_nums, 256 * input_size/8 * input_size/8};          // 或 64*64，根据你的模型
//     for (int i = 0; i < 2; ++i) {
//         const float* out_ptr = reinterpret_cast<const float*>(superpoint_outputs[i].buf);
//         save_float_data(("cpp_output_" + std::to_string(i) + ".bin").c_str(), out_ptr, elem_count[i]);
//         std::cout << "Saved cpp_output_" << i << ".bin (" << elem_count[i] << " floats)\n";
//     }

    int s_sizes[] = {1, input_size, input_size};
    Mat pred_scores(3, s_sizes, CV_32F, superpoint_outputs[0].buf);
    int d_sizes[] = {1, 256, input_size/8, input_size/8}; // example
    Mat descriptors_4d(4, d_sizes, CV_32F, superpoint_outputs[1].buf);

    auto result = postprocessKeypoints(pred_scores, 4, top_nums);
    vector<Point2f>& keypoints = result.first;
    vector<float>& scores = result.second;
    Mat desc = postprocessDescriptors(keypoints, descriptors_4d, 8);

//     save_float_data("cpp_keypoints.bin", (const float*)keypoints.data(), top_nums*2);
//     save_float_data("cpp_desc.bin", (const float*)desc.data, top_nums*256);

    rknn_outputs_release(superpoint_ctx, 2, superpoint_outputs);
    return {keypoints, desc};

}

pair<vector<Point2f>, Mat> Matches::superpoint_infer(const cv::Mat& img) {
    fixed_input = false;
    superpoint_inputs[0].buf = img.data;

    rknn_inputs_set(superpoint_ctx, superpoint_io_num.n_input, superpoint_inputs);  // 可以设置模型的输入数据

    rknn_output superpoint_outputs[2];
    memset(superpoint_outputs, 0, sizeof(superpoint_outputs));
    for (int i = 0; i < superpoint_io_num.n_output; i++) {
        superpoint_outputs[i].want_float = 1;
        superpoint_outputs[i].is_prealloc = 0;
    }
    auto start = chrono::steady_clock::now();
    int ret = rknn_run(superpoint_ctx, NULL);  // 将执行一次模型推理，调用之前需要先通过 rknn_inputs_set 函数或者零拷贝的接口设置输入数据
    rknn_outputs_get(superpoint_ctx, 2, superpoint_outputs, nullptr);
    auto end = chrono::steady_clock::now();
    auto tt = chrono::duration_cast<microseconds>(end - start);
    cout << "superpoint infer time is = " << tt.count() << "us" << endl;

    // Save superpoint_outputs as float32
//     size_t elem_count[2] = {input_size * input_size, 256 * 64 * 64};  // 或 64*64，根据你的模型
//     for (int i = 0; i < 2; ++i) {
//         const float* out_ptr = reinterpret_cast<const float*>(superpoint_outputs[i].buf);
//         save_float_data(("cpp_output_" + std::to_string(i) + ".bin").c_str(), out_ptr, elem_count[i]);
//         std::cout << "Saved cpp_output_" << i << ".bin (" << elem_count[i] << " floats)\n";
//     }

    int s_sizes[] = {1, input_size, input_size};
    Mat pred_scores(3, s_sizes, CV_32F, superpoint_outputs[0].buf);
    int d_sizes[] = {1, 256, input_size/8, input_size/8}; // example
    Mat descriptors_4d(4, d_sizes, CV_32F, superpoint_outputs[1].buf);

    auto result = postprocessKeypoints(pred_scores, 4, top_nums);
    vector<Point2f>& keypoints = result.first;
    vector<float>& scores = result.second;
    Mat desc = postprocessDescriptors(keypoints, descriptors_4d, 8);

    rknn_outputs_release(superpoint_ctx, 2, superpoint_outputs);
    return {keypoints, desc};
}


LightGlueMatches Matches::lightglue_infer(const vector<Point2f>& keypoints0, const Mat& desc0, const vector<Point2f>& keypoints1, const Mat& desc1){
    assert(keypoints0.size() == top_nums);
    assert(keypoints1.size() == top_nums);
    CV_Assert(desc0.type() == CV_32F && desc0.rows == top_nums && desc0.cols == 256 && desc0.isContinuous());
    CV_Assert(desc1.type() == CV_32F && desc1.rows == top_nums && desc1.cols == 256 && desc1.isContinuous());

    // Directly use data pointers — no copy!
    lightglue_inputs[0].buf = const_cast<void*>(static_cast<const void*>(keypoints0.data()));
    lightglue_inputs[1].buf = const_cast<void*>(static_cast<const void*>(keypoints1.data()));
    lightglue_inputs[2].buf = desc0.data;
    lightglue_inputs[3].buf = desc1.data;

    rknn_inputs_set(lightglue_ctx, lightglue_io_num.n_input, lightglue_inputs);  // 可以设置模型的输入数据

    rknn_output lightglue_outputs[1];
    memset(lightglue_outputs, 0, sizeof(lightglue_outputs));
    for (int i = 0; i < superpoint_io_num.n_output; i++) {
        lightglue_outputs[i].want_float = 1;
        //lightglue_outputs[i].buf = nullptr;      // 👈 关键！确保 buf 初始为 null
        //lightglue_outputs[i].size = 0;
        lightglue_outputs[i].is_prealloc = 0;
    }

    auto start = chrono::steady_clock::now();
    int ret = rknn_run(lightglue_ctx, NULL);  // 将执行一次模型推理，调用之前需要先通过 rknn_inputs_set 函数或者零拷贝的接口设置输入数据
    ret = rknn_outputs_get(lightglue_ctx, 1, lightglue_outputs, nullptr);
    auto end = chrono::steady_clock::now();
    auto tt = chrono::duration_cast<microseconds>(end - start);
    cout << "lightglue infer time is = " << tt.count() << "us" << endl;

    // Save lightglue_outputs as float32 保存到文件
//     size_t elem_count = (top_nums+1) * (top_nums+1);          // 或 64*64，根据你的模型
//     const float* out_ptr = reinterpret_cast<const float*>(lightglue_outputs[0].buf);
//     save_float_data("cpp_output_0.bin", out_ptr, elem_count);
//     cout << "Saved cpp_output_0.bin (" << elem_count << " floats)" << endl;

    if(fixed_input){
        ar1 = 0.4;
        ar2 = 0.4;
        ox1 = 0;
        ox2 = 0;
        oy1 = 51;
        oy2 = 51;
    }

    float* src = static_cast<float*>(lightglue_outputs[0].buf);
    Mat scores_flat = Mat((top_nums+1) * (top_nums+1), 1, CV_32F, src).clone();
    Mat scores = scores_flat.reshape(1, {(top_nums+1), (top_nums+1)});
    auto result = postprocessLightGlue(scores, keypoints0, keypoints1, 0.01f);
    rknn_outputs_release(lightglue_ctx, 1, lightglue_outputs);
    return result;
}

Matches::~Matches() {
    cout << "~~~xigou~~~" << endl;
    int ret;
    ret = rknn_destroy(superpoint_ctx);  // 将释放传入的 rknn_context 及其相关资源
    ret = rknn_destroy(lightglue_ctx);  // 将释放传入的 rknn_context 及其相关资源
    if (superpoint_model_data) {
        free(superpoint_model_data);
    }
    if (lightglue_model_data) {
        free(lightglue_model_data);
    }
}


