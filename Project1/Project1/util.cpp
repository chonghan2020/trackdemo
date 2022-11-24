#include"util.h"

void printHelpInfo()
{
	std::cout
		<< "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
		<< std::endl;
	std::cout << "--help          Display help information" << std::endl;
	std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
		"multiple times to add multiple directories. If no data directories are given, the default is to use "
		"(data/samples/mnist/, data/mnist/)"
		<< std::endl;
	std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
		"where n is the number of DLA engines on the platform."
		<< std::endl;
	std::cout << "--int8          Run in Int8 mode." << std::endl;
	std::cout << "--fp16          Run in FP16 mode." << std::endl;
}
