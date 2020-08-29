#include <dv-sdk/module.hpp>
#include <dv-sdk/processing.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "pinball.hpp"
#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <vector>


using namespace Eigen;


class FeastDV : public dv::ModuleBase {
private:
	// user selectable refractory period in microseconds
	int n_neurons;
	float thresholdClose;
	float thresholdOpen;
	float eta;
	float tau;
	int radius;

	int rows;
	int cols;

	

	cv::Mat w;
	cv::Mat threshold;
	dv::TimeMat lastFiringTimes;
	cv::Mat roiSurface;
    cv::Mat dotProduct;
public:
	static void initInputs(dv::InputDefinitionList &in) {
		in.addEventInput("events");
	}

	static void initOutputs(dv::OutputDefinitionList &out) {
		out.addEventOutput("events");
	}

	static const char *initDescription() {
		return ("The Feast Layer Module");
	}

	static void initConfigOptions(dv::RuntimeConfig &config) {
		config.add("n_neurons",
			dv::ConfigOption::intOption("Number of neurons", 10, 2, 100));
        config.add("thresholdClose",
                   dv::ConfigOption::floatOption("Threshold Close", 0.01, 0, 1));
        config.add("thresholdOpen",
                   dv::ConfigOption::floatOption("Threhsold Open", 0.01, 0, 1));
        config.add("eta",
                   dv::ConfigOption::floatOption("Eta", 0.001, 0, 1));
        config.add("tau",
                   dv::ConfigOption::floatOption("Tau", 1000, 1, 2000));
        config.add("roi_x",
                   dv::ConfigOption::intOption("ROI X", 10, 2, 100));
        config.add("roi_y",
                   dv::ConfigOption::intOption("ROI Y", 10, 2, 100));
        config.add("radius",
                   dv::ConfigOption::intOption("ROI radius", 10, 2, 100));



//        config.setPriorityOptions({"tau"});
	}

	FeastDV() : n_neurons(10), thresholdClose(0.01), thresholdOpen(0.02), eta(0.01), tau(10000),radius(10),lastFiringTimes(inputs.getEventInput("events").size()) {
		outputs.getEventOutput("events").setup(1,n_neurons,"Output feast events");
//        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE );

        w = cv::Mat(static_cast<int>((2*radius+1)*(2*radius+1)),n_neurons,CV_32F,cv::Scalar(0.0));
        cv::randu(w,cv::Scalar(0.0),cv::Scalar(1.0));
        for(int i = 0; i<n_neurons; i++){
            w.col(i) = w.col(i)/cv::norm(w.col(i),cv::NormTypes::NORM_L2);
        }
        roiSurface = cv::Mat( 1,static_cast<int>((2*radius+1)*(2*radius+1)),CV_32F,cv::Scalar(0.0));
        dotProduct = cv::Mat(1,n_neurons,CV_32F,cv::Scalar(0.0));
        threshold = cv::Mat(1,n_neurons,CV_32F,cv::Scalar(0.0));
        rows = inputs.getEventInput("events").sizeX();
        cols = inputs.getEventInput("events").sizeY();

	}

	void run() override {
		auto input  = inputs.getEventInput("events");
		auto output = outputs.getEventOutput("events");
//        dv::EventStore store = inputs.getEventInput("events").events();
//        dv::TimeSurface TimeSurface(rows, cols);
//        dv::Accumulator frameAccumulator;
        int event_x;
        int event_y;
        double roi_norm = 0;
        double w_norm;
//        int roi_surface_h = roiSurface.t().size().height;
//        int roi_surface_w = roiSurface.t().size().width;
//        int w_h = w.col(0).size().height;
//        int w_w = w.col(0).size().width;
//        int threshold_h = threshold.size().height;
//        int threshold_w = threshold.size().width;

        for (const auto &event : input.events()){
            event_x = event.x();
            event_y = event.y();
            lastFiringTimes.at(event.y(), event.x()) = event.timestamp();
            int winnerNeuron = -1;
            int maxDotProduct = 0;
            if(event.x()-radius < 0 || event.x()+radius > rows -1 || event.y()-radius < 0 || event.y()+radius > cols-1){
                continue;
            } else{

                for(int x=event_x-radius;x<=event_x+radius;x++){
                    for(int y=event_y-radius;y<=event_y+radius;y++){
                       roiSurface.at<float>(cv::Point(0,(x-event_x+radius)*radius + y-event_y+radius)) = static_cast<float>(expf(
                               -(static_cast<float>(event.timestamp() - lastFiringTimes.at(y,x))) / tau));
                    }
                }
                roi_norm = cv::norm(roiSurface,cv::NormTypes::NORM_L2);
                if (roi_norm > 0) {
                    roiSurface /= roi_norm;
                }

//                cv::normalize(roiSurface,roiSurface);
                dotProduct = roiSurface*w;

                for(int n = 0; n < n_neurons; n++){
                    if(dotProduct.at<float>(cv::Point(0,n)) > maxDotProduct && dotProduct.at<float>(cv::Point(0,n))  > threshold.at<float>(cv::Point(0,n))){
                        winnerNeuron = n;
                        maxDotProduct = dotProduct.at<float>(cv::Point(0,n)) ;
                    }
                }
                if (winnerNeuron > -1){
                    w.col(winnerNeuron) = w.col(winnerNeuron)*(1-eta) + roiSurface.t()*eta;
                    w_norm = cv::norm(w.col(winnerNeuron),cv::NormTypes::NORM_L2);
                    w.col(winnerNeuron) /= w_norm;
//                    cv::normalize(w.col(winnerNeuron), w.col(winnerNeuron));
                    threshold.col(winnerNeuron) += thresholdClose;
                    output << dv::Event(event.timestamp(), 0,winnerNeuron,true);
                }else{
                    threshold -= thresholdOpen;
                }

            }
		}
        output << dv::commit;

	}

	void configUpdate() override {
        n_neurons = config.getInt("n_neurons");
        tau = config.getFloat("tau");
        thresholdOpen = config.getFloat("thresholdOpen");
        thresholdClose= config.getFloat("thresholdClose");
        radius = config.getInt("radius");
        outputs.getEventOutput("events").setup(1,n_neurons,"Output feast events");


        w.release();
        roiSurface.release();
        dotProduct.release();
        threshold.release();
        w = cv::Mat(static_cast<int>((2*radius+1)*(2*radius+1)),n_neurons,CV_32F,cv::Scalar(0.0));
        cv::randu(w,cv::Scalar(0.0),cv::Scalar(1.0));
        for(int i = 0; i<n_neurons; i++){
            w.col(i) = w.col(i)/cv::norm(w.col(i),cv::NormTypes::NORM_L2);
        }
        roiSurface = cv::Mat( 1,static_cast<int>((2*radius+1)*(2*radius+1)),CV_32F,cv::Scalar(0.0));
        dotProduct = cv::Mat(1,n_neurons,CV_32F,cv::Scalar(0.0));
        threshold = cv::Mat(1,n_neurons,CV_32F,cv::Scalar(0.0));
        rows = inputs.getEventInput("events").sizeX();
        cols = inputs.getEventInput("events").sizeY();

    }
};

registerModuleClass(FeastDV)
