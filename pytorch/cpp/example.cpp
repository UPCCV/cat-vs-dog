#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "mrutil.h"

//https://pytorch.org/tutorials/advanced/cpp_export.html
std::string CLASSES[] = {"cat","dog"};

int main(int argc, const char* argv[]) {
    auto module = torch::jit::load("../mnasnet_dogcat.pt");
    std::string imgpath = "../data/1.jpg";
    if(argc > 1){
        imgpath = argv[1];
    }
    auto image = cv::imread(imgpath);
    if(image.data == NULL){
        std::cout << imgpath << " cannot read" << std::endl;
        return -1;
    }
    cv::Mat image_transfomed;
    cv::resize(image, image_transfomed, cv::Size(224, 224));
    cv::cvtColor(image_transfomed, image_transfomed, cv::COLOR_BGR2RGB);
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, {image_transfomed.rows, image_transfomed.cols,3},torch::kByte);
    tensor_image = tensor_image.permute({2,0,1});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.unsqueeze(0);
    at::Tensor output = module.forward({tensor_image}).toTensor();
    auto max_result = output.max(1,true);
    auto max_index = std::get<1>(max_result).item<float>();
    auto score = std::get<0>(max_result).item<float>();
    std::string display=CLASSES[int(max_index)]+tostring(score);
    cv::putText(image, display, cv::Point(100, 50), 1, 1,cv::Scalar(0, 0, 255));
    cv::imshow("image", image);
    cv::waitKey(0);
    //cv::imwrite("./result.jpg", image);
}