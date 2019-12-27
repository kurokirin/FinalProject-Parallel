#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <functional>
#include <set>
#include <ctime>

using namespace std;
map<int, vector<int>> inlinks;
map<int, vector<int>> outlinks;
vector<int> sink;
map<int, double> pr_score;
int countConverge = 0;
double d = 0.85; //pagerank damping/teleportation factor in this project we use 0.85
double previous_perplex = 0;
int allPages = 0;

void pageArr(vector<std::string> str) {
	vector<int> pageIDs;
	vector<int> out;
	int count = 0, firstCol;
	for (string s : str) {
		stringstream page(s);
		int x = 0;
		page >> x;
		if (count == 0) {
			firstCol = x;
		}
		else {
			pageIDs.push_back(x);
		}
		count++;
	}
	inlinks.insert({ firstCol,pageIDs });
	outlinks.insert({ firstCol, out });
	allPages++;
}
std::vector<std::string> split(std::string strToSplit, char delimeter)
{
    std::stringstream ss(strToSplit);
    std::string item;
    std::vector<std::string> splittedStrings;
    while (std::getline(ss, item, delimeter))
    {
       splittedStrings.push_back(item);
    }
    return splittedStrings;
}
void readfile() {
	std::ifstream myfile;
	myfile.open("D:\\Parallel\\Parallel\\x64\\Debug\\citeseer.dat");
	string myText;
	cout << "Reading from the file" << endl;
	while (getline(myfile, myText)) {
		//cout << myText <<endl;
		pageArr(split(myText,' '));
	}
	myfile.close();
	cout << "finish reading from file" << endl;
	//cin.get();
}

void findOutlinks() {
	cout << "find outlinks" << endl;
	for (auto& x: inlinks) {
		for (int i : x.second) {
			try {
				/*if (outlinks.at(i).empty()) {
					continue;
				}*/
				outlinks[i].push_back(x.first);
			}
			catch (const std::out_of_range& oor) {
				std::cerr << "Out of Range error: " << oor.what() << '\n';
				//cout << "pid: " << i << endl;
			}
		}
	}
}

void findSinkNode() {
	cout << "find sink node" << endl;
	for (auto& x : outlinks) {
		if (x.second.size() == 0) {
			sink.push_back(x.first);
		}
	}
}

void initialize() {
	cout << "initialize" << endl;
	double init_val = 1/allPages;
	for (auto& x : inlinks) {
		pr_score.insert({ x.first, init_val });
	}

}

double getPerplexity() {
	cout << "get perplexity" << endl;
	double sum = 0;
	for (auto& x : pr_score) {
		if (x.second != 0) {
			sum += (x.second*log2(x.second));
		}
		if (!isnormal(sum)) {
			cout << sum;
		}
	}
	if (!isnormal(sum)) {
		cout << sum;
	}
	sum = pow(2, -(sum));
	return sum;
}

bool isConverge() {
	if (countConverge == 3) return true;
	else return false;
}

void runPageRank() {
	cout << "run pagerank" << endl;
	double sinkPR, temp = 0;
	double newPR;
	//vector<double> perplexities;
	for (int pid : sink) {
		try {
			temp += pr_score[pid];
		}
		catch (const std::out_of_range& oor) {
			std::cerr << "Out of Range error: " << oor.what() << '\n';
			//cout << "(sink)pid: " << pid << endl;
		}
	}
	while (!isConverge()) {
		//pr score of each page
		for (auto& pid : inlinks) {
			sinkPR = temp;
			newPR = (1 - d) / inlinks.size();
			newPR += (d*(sinkPR / inlinks.size()));
			//cout << "newPR1 = " << newPR << endl;
			for (int i : pid.second) {
				if (!outlinks[i].size()==0) {
					newPR += (d*(pr_score[i] / outlinks[i].size()));
				}
				/*try {
					newPR += (double)(d*(pr_score[i] / outlinks[i].size()));
				}
				catch (const std::out_of_range& oor) {
					std::cerr << "Out of Range error: " << oor.what() << '\n';
					//cout << "pid: " << i << endl;
				}*/
			}
			//cout << "newPR2 = " << newPR << endl;
			pr_score[pid.first] = newPR;

		}
		double ceilPrePerPlex = ceil(getPerplexity());
		if (previous_perplex == ceilPrePerPlex) countConverge++;
		else {
			previous_perplex = ceilPrePerPlex;
			countConverge = 0;
		}
		//perplexities.push_back(getPerplexity());
		//cout << "perplexity = " << getPerplexity() << endl;
		//cout << "end round" << endl;
	}
}

vector<int> getRankedPages(int K) {
	cout << "get rank" << endl;
	map<double, int> rankmap;
	multimap<int, double> ::iterator iter;
	for (iter = pr_score.begin(); iter != pr_score.end(); iter++)
	{
		rankmap.insert({ (*iter).second, (*iter).first });
		//cout << iter->second << ": " << iter->first << endl;
	}

	vector<int> rank;
	int count = 0;
	for (auto& x: rankmap) {
		rank.push_back(x.second);
		count++;
		if (count == (K)) break;
	}
	return rank;
}

int main() {
	std::clock_t start,stop;
	start = std::clock();

		readfile();
		findOutlinks();
		findSinkNode();
		initialize();
		runPageRank();
		vector<int> rankpages = getRankedPages(100);

	stop = std::clock();
	double duration = stop - start/1000.0;

	cout << "Top 100 pages are\n";
	for (int i = 0; i < 100; i++) {
		cout << "Rank#" << i << " PageID: " << rankpages[i] << "\n";
	}
	cout << "time: " << duration << endl;
	cin.get();
	return 0;
}