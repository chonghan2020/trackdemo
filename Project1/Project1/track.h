#ifndef TRACK_H
#define TRACK_H


#include "kalmanfilter.h"
#include "util.h"

class Track
{
	enum TrackState { Tentative = 1, Confirmed, Deleted };
public:
	Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id,
		int n_init, int max_age, const FEATURE& feature);
	Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id,
		int n_init, int max_age, const FEATURE& feature, int cls);
	int cls;
	KAL_MEAN mean;
	KAL_COVA covariance;
	int track_id;
	int hits;
	int age;
	int _n_init;
	int _max_age;
	FEATURESS features;
	int time_since_update;
	TrackState state;
	void mark_missed();
	DETECTBOX to_tlwh();
	bool is_confirmed();
	bool is_deleted();
	bool is_tentative();
	void predit(KalmanFilterhc* kf);
	void update(KalmanFilterhc* const kf, const TrackBoxConvert &detection);
private:
	void featuresAppendOne(const FEATURE& f);
};

#endif
