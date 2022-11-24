#ifndef TRACKER_H
#define TRACKER_H


#include <vector>

#include "kalmanfilter.h"
#include "track.h"
#include "nn_matching.h"


using namespace std;

class NearNeighborDisMetric;

class tracker
{
public:
	NearNeighborDisMetric* metric;
	float max_iou_distance;
	int max_age;
	int n_init;
	KalmanFilterhc* kf;
	int _next_idx;
public:


	std::vector<Track> tracks;
	tracker(/*NearNeighborDisMetric* metric,*/
		float max_cosine_distance, int nn_budget,
		float max_iou_distance = 0.7,
		int max_age = 70, int n_init = 3);
	void _initiate_track(const TrackBoxConvert& detection);
	void predict();
	void update(const DETECTIONS& detections);
	void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);


	typedef DYNAMICM(tracker::* GATED_METRIC_FUNC)(
		std::vector<Track>& tracks,
		const DETECTIONS& dets,
		const std::vector<int>& track_indices,
		const std::vector<int>& detection_indices);

	DYNAMICM gated_matric(
		std::vector<Track>& tracks,
		const DETECTIONS& dets,
		const std::vector<int>& track_indices,
		const std::vector<int>& detection_indices);
	DYNAMICM iou_cost(
		std::vector<Track>& tracks,
		const DETECTIONS& dets,
		const std::vector<int>& track_indices,
		const std::vector<int>& detection_indices);
	Eigen::VectorXf iou(DETECTBOX& bbox,
		DETECTBOXSS &candidates);
};

#endif // TRACKER_H
