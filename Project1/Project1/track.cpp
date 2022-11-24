#include "track.h"

Track::Track(KAL_MEAN & mean, KAL_COVA & covariance, int track_id, int n_init, int max_age, const FEATURE & feature)
{
	this->mean = mean;
	this->covariance = covariance;
	this->track_id = track_id;
	this->hits = 1;
	this->age = 1;
	this->time_since_update = 0;
	this->state = TrackState::Tentative;
	features = FEATURESS(1, SORT_SHAPE);
	features.row(0) = feature; 
	this->_n_init = n_init;
	this->_max_age = max_age;
}

void Track::predit(KalmanFilterhc * kf)
{

	kf->predict(this->mean, this->covariance);
	this->age += 1;
	this->time_since_update += 1;
}

void Track::update(KalmanFilterhc * const kf, const TrackBoxConvert & detection)
{
	KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
	this->mean = pa.first;
	this->covariance = pa.second;

	featuresAppendOne(detection.feature);
	this->hits += 1;
	this->time_since_update = 0;
	if (this->state == TrackState::Tentative && this->hits >= this->_n_init) {
		this->state = TrackState::Confirmed;
	}
}

void Track::featuresAppendOne(const FEATURE & f)
{
	int size = this->features.rows();
	FEATURESS newfeatures = FEATURESS(size + 1, SORT_SHAPE);
	newfeatures.block(0, 0, size, SORT_SHAPE) = this->features;
	newfeatures.row(size) = f;
	features = newfeatures;
}

void Track::mark_missed()
{
	if (this->state == TrackState::Tentative) {
		this->state = TrackState::Deleted;
	}
	else if (this->time_since_update > this->_max_age) {
		this->state = TrackState::Deleted;
	}
}

bool Track::is_confirmed()
{
	return this->state == TrackState::Confirmed;
}

bool Track::is_deleted()
{
	return this->state == TrackState::Deleted;
}

bool Track::is_tentative()
{
	return this->state == TrackState::Tentative;
}


DETECTBOX Track::to_tlwh()
{
	DETECTBOX ret = mean.leftCols(4);
	ret(2) *= ret(3);
	ret.leftCols(2) -= (ret.rightCols(2) / 2);
	return ret;
}


