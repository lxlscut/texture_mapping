//如果要生成python库文件，需要在属性->常规->配置类型中将其改为dll，生成后在release文件夹下将文件名改为pyd即可

#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<Eigen/Dense>
#include<Eigen/SVD>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
namespace py = pybind11;
using namespace std;
#define MAXFLOAT std::numeric_limits<double>::max()

template <typename T>
cv::Point_<T> applyTransform2x3(T x, T y, const cv::Mat& matT) {
	return cv::Point_<T>((matT.at<double>(0, 0) * x + matT.at<double>(0, 1) * y + matT.at<double>(0, 2)),
		(matT.at<double>(1, 0) * x + matT.at<double>(1, 1) * y + matT.at<double>(1, 2)));
}


py::array_t<uint> texture_mapping(py::array_t<float>& origin_vertice, py::array_t<float> transformed_vertices, py::array_t<uint>& pre_align_picture) {
	//由于需要用到opencv的fillpoly，applyTransform2x3，getAffineTransform函数，因此先把数据转换为opencv的格式
	auto r1 = origin_vertice.unchecked<3>();
	auto r2 = transformed_vertices.unchecked<3>();
	auto r3 = pre_align_picture.unchecked<3>();


	//图片转换到opencv
	cv::Mat origin(cv::Size(pre_align_picture.shape()[0], pre_align_picture.shape()[1]),CV_8UC3);
	for (int row = 0; row < pre_align_picture.shape()[0]; row++) {
		for (int col = 0; col < pre_align_picture.shape()[1]; col++) {
			origin.at<cv::Vec3b>(row, col) = cv::Vec3b(r3(row, col, 0), r3(row, col, 1), r3(row, col, 2));
		}
	}


	//获取3角形，一共有方块的个数乘以二个
	vector<vector<cv::Point2f>> origin_triangles;
	origin_triangles.reserve((origin_vertice.shape()[0] - 1) * (origin_vertice.shape()[1] * 2));
	for (int row = 0; row < origin_vertice.shape()[0]-1; row++) {
		for (int col = 0; col < origin_vertice.shape()[1] - 1; col++) {
			vector<cv::Point2f> triangle1(3), triangle2(3);
			triangle1[0] = cv::Point2f(r1(row, col, 0), r1(row, col, 1));
			triangle1[1] = cv::Point2f(r1(row, col + 1, 0), r1(row, col + 1, 1));
			triangle1[2] = cv::Point2f(r1(row + 1, col, 0), r1(row + 1, col, 1));
			origin_triangles.emplace_back(triangle1);
			triangle2[0] = cv::Point2f(r1(row, col+1, 0), r1(row, col+1, 1));
			triangle2[1] = cv::Point2f(r1(row+1, col + 1, 0), r1(row+1, col + 1, 1));
			triangle2[2] = cv::Point2f(r1(row + 1, col, 0), r1(row + 1, col, 1));
			origin_triangles.emplace_back(triangle2);
		}
	}


	//求取画布的信息
	float minx = MAXFLOAT, miny = MAXFLOAT, maxy = -MAXFLOAT, maxx = -MAXFLOAT;
	for (int row = 0; row < transformed_vertices.shape()[0]; row++) {
		for (int col = 0; col < transformed_vertices.shape()[1]; col++) {
			minx = min(r2(row, col, 0), minx);
			miny = min(r2(row, col, 1), miny);
			maxx = max(r2(row, col, 0), maxx);
			maxy = max(r2(row, col, 1), maxy);
		}
	}
	cv::Rect canvas;
	canvas.x = round(minx);
	canvas.y = round(miny);
	canvas.height = round(maxy - miny);
	canvas.width = round(maxx - minx);
	
	cout << "get canvas  success success ..." << "the start of canvas is:" << "x:" << canvas.x << "y:" << canvas.y << "the height is " <<canvas.height << "the width is :" << canvas.width << endl;

	//与上面一样，只不过采取的顶点集不一样，除此之外，为了让大小合适，取出包含图像的部分
	vector<vector<cv::Point2f>> transformed_triangles;
	transformed_triangles.reserve((origin_vertice.shape()[0] - 1) * (origin_vertice.shape()[1] * 2));
	for (int row = 0; row < origin_vertice.shape()[0] - 1; row++) {
		for (int col = 0; col < origin_vertice.shape()[1] - 1; col++) {
			vector<cv::Point2f> triangle1(3), triangle2(3);
			triangle1[0] = cv::Point2f(r2(row, col, 0)-canvas.x, r2(row, col, 1)-canvas.y);
			triangle1[1] = cv::Point2f(r2(row, col + 1, 0) - canvas.x, r2(row, col + 1, 1) - canvas.y);
			triangle1[2] = cv::Point2f(r2(row + 1, col, 0) - canvas.x, r2(row + 1, col, 1) - canvas.y);
			transformed_triangles.emplace_back(triangle1);
			triangle2[0] = cv::Point2f(r2(row, col + 1, 0) - canvas.x, r2(row, col + 1, 1) - canvas.y);
			triangle2[1] = cv::Point2f(r2(row + 1, col + 1, 0) - canvas.x, r2(row + 1, col + 1, 1) - canvas.y);
			triangle2[2] = cv::Point2f(r2(row + 1, col, 0) - canvas.x, r2(row + 1, col, 1) - canvas.y);
			transformed_triangles.emplace_back(triangle2);
		}
	}



	//对于每一个三角形获取其变换
	vector<cv::Mat> affine_transforms;
	for (int i = 0; i < origin_triangles.size(); i++) {
		cv::Point2f origin[3];
		cv::Point2f tranformed[3];
		for (int j = 0; j < 3; j++) {
			origin[j] = origin_triangles[i][j];
			tranformed[j] = transformed_triangles[i][j];
		}
		cv::Mat matrix = cv::getAffineTransform(tranformed,origin);
		cv::Point2f dst = applyTransform2x3<float>(tranformed[0].x, tranformed[0].y, matrix);
		cout << "origin" << origin[0].x << origin[0].y << "dst" << dst.x << dst.y << endl;
		affine_transforms.emplace_back(matrix);		
	}



	//创建画布，并按顺序填充三角形
	cv::Mat my_mask(cv::Size(canvas.width, canvas.height), CV_32SC1, cv::Scalar::all(-1));
	for (int i = 0; i < transformed_triangles.size(); i++) {
		cv::Point2i my_contour[3];
		for (int j = 0; j < 3; j++) {
			my_contour[j] = transformed_triangles[i][j];
		}
		cv::fillConvexPoly(my_mask, my_contour, 3, i, 16, 0);
	}
	



	//进行投影
	cv::Mat result(cv::Size(canvas.width,canvas.height), CV_8UC3);
	for (int row = 0; row < result.rows; row++) {
		for (int col = 0; col < result.cols; col++) {
			if (my_mask.at<int>(row, col) != -1) {
				int index = my_mask.at<int>(row, col);
				cv::Mat affine_matrix = affine_transforms[index];
				cv::Point2f dst = applyTransform2x3<float>(col, row, affine_matrix);
				if ((dst.x >= 0 && round(dst.x) < pre_align_picture.shape()[1]) && (dst.y >= 0 && round(dst.y) < pre_align_picture.shape()[0])) {
					result.at<cv::Vec3b>(row, col) = cv::Vec3b(r3(round(dst.y), round(dst.x), 0),
						r3(round(dst.y), round(dst.x), 1),
						r3(round(dst.y), round(dst.x), 2));
			}
			}
		}
	}



	py::array_t<uint> output = py::array_t<uint>(canvas.height * canvas.width * 3);
	output.resize({ canvas.height, canvas.width,3 });
	auto r8 = output.mutable_unchecked<3>();
	// 491 736 736 491 491 736
	cout << result.cols << result.rows << output.shape()[1] << output.shape()[0] << canvas.height << canvas.width << endl;
	for (int row = 0; row < output.shape()[0]; row++) {
		for (int col = 0; col < output.shape()[1]; col++) {
			for (int channel = 0; channel < output.shape()[2]; channel++) {
				uint value = result.at<cv::Vec3b>(row, col)[channel];
				r8(row, col, channel) = value;
			}
		}
	}

	return output;
}


PYBIND11_MODULE(texture_mapping,m) {
	m.def("texture_mapping", &texture_mapping);
}
